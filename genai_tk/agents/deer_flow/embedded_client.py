"""Embedded Python client for Deer-flow agent system.

Loads DeerFlow in-process via ``DEER_FLOW_PATH/backend`` — no LangGraph Server or
Gateway API processes needed. A ``SqliteSaver`` checkpointer at
``data/kv_store/deerflow_checkpoints.db`` keeps multi-turn state within a session.

Also defines the typed streaming event dataclasses (``TokenEvent``, ``NodeEvent``,
``ToolCallEvent``, ``ToolResultEvent``, ``ErrorEvent``) that the CLI consumes.

Usage:
    ```python
    from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

    client = EmbeddedDeerFlowClient(config_path="/path/to/config.yaml")
    thread_id = "my-session"
    async for event in client.stream_message(thread_id, "What is RAG?", mode="pro"):
        if isinstance(event, TokenEvent):
            print(event.data, end="", flush=True)
    ```
"""

from __future__ import annotations

import asyncio
import builtins
import os
import queue
import re
import sys
import threading
import warnings
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

# Suppress Pydantic serialization warnings emitted by LangGraph's checkpointer
# when it serialises RunnableConfig.context (which is always a dict, not None).
# NOTE: warnings.filterwarnings uses re.match() (not re.search), so the pattern
# MUST match from the start of the (possibly multi-line) warning text.
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
)

# Ensure localhost/127.0.0.1 bypass any corporate HTTP proxy (e.g. Zscaler).
# DeerFlow's sandbox health-check uses ``requests.get("http://localhost:<port>/…")``,
# which honours ``HTTP_PROXY`` — a proxy will intercept the request and the
# sandbox appears unreachable even though the container is healthy.
_NO_PROXY_HOSTS = "localhost,127.0.0.1"
for _var in ("no_proxy", "NO_PROXY"):
    existing = os.environ.get(_var, "")
    if _NO_PROXY_HOSTS not in existing:
        os.environ[_var] = f"{existing},{_NO_PROXY_HOSTS}" if existing else _NO_PROXY_HOSTS

# ---------------------------------------------------------------------------
# Typed events yielded by stream_message()
# ---------------------------------------------------------------------------


@dataclass
class TokenEvent:
    """Full AI response text for a completed message."""

    kind: str = "token"
    data: str = ""


@dataclass
class NodeEvent:
    """A graph node became active (Planner, Researcher, Coder, Reporter …)."""

    kind: str = "node"
    node: str = ""
    state: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = {}


@dataclass
class ToolCallEvent:
    """The model is calling a tool."""

    kind: str = "tool_call"
    tool_name: str = ""
    args: dict = None  # type: ignore[assignment]
    call_id: str = ""

    def __post_init__(self) -> None:
        if self.args is None:
            self.args = {}


@dataclass
class ToolResultEvent:
    """A tool returned a result."""

    kind: str = "tool_result"
    tool_name: str = ""
    content: str = ""
    call_id: str = ""


@dataclass
class ErrorEvent:
    """The run produced an error."""

    kind: str = "error"
    message: str = ""


# Union type alias matching the HTTP client's public API
StreamEvent = TokenEvent | NodeEvent | ToolCallEvent | ToolResultEvent | ErrorEvent


# ---------------------------------------------------------------------------
# Mode → flag helpers  (duplicated from cli_commands to avoid circular import)
# ---------------------------------------------------------------------------

_MODE_FLAGS: dict[str, dict[str, bool]] = {
    "flash": {"thinking_enabled": False, "is_plan_mode": False, "subagent_enabled": False},
    "thinking": {"thinking_enabled": True, "is_plan_mode": False, "subagent_enabled": False},
    "pro": {"thinking_enabled": True, "is_plan_mode": True, "subagent_enabled": False},
    "ultra": {"thinking_enabled": True, "is_plan_mode": True, "subagent_enabled": True},
}


def _mode_flags(mode: str) -> dict[str, bool]:
    return _MODE_FLAGS.get(mode.lower(), _MODE_FLAGS["flash"])


# ---------------------------------------------------------------------------
# sys.path injection
# ---------------------------------------------------------------------------


def _ensure_deer_flow_on_path() -> Path:
    """Add ``DEER_FLOW_PATH/backend`` to :data:`sys.path` if absent.

    Returns:
        Resolved backend directory path.
    """
    df_path = os.environ.get("DEER_FLOW_PATH", "")
    if not df_path:
        raise RuntimeError("DEER_FLOW_PATH is not set. Set it to the root of your deer-flow clone.")
    backend_path = Path(df_path) / "backend"
    if not backend_path.exists():
        raise RuntimeError(f"DEER_FLOW_PATH/backend not found: {backend_path}")
    backend_str = str(backend_path)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)
        logger.debug(f"Added {backend_str} to sys.path")
    _patch_deer_flow_print()
    _suppress_deer_flow_logging()
    return backend_path


def _suppress_deer_flow_logging() -> None:
    """Raise DeerFlow's standard-library loggers to CRITICAL.

    DeerFlow modules use ``logging.getLogger(__name__)`` under the ``src``
    namespace. Their INFO/ERROR messages (e.g. sandbox file-not-found 404s,
    agent internals) are not actionable for CLI users and clutter the output.
    Tool errors are already surfaced via ``ToolResultEvent``.
    """
    import logging as _std_logging

    _std_logging.getLogger("src").setLevel(_std_logging.CRITICAL)


_DEER_FLOW_SUPPRESS_PREFIXES = (
    "Lazy acquiring sandbox",
    "Acquiring sandbox",
    "Memory update timer",
    "Memory update queued",
    "Generated thread title",
    "Failed to read file",
    "Failed to list directory",
)


def _patch_deer_flow_print() -> None:
    """Redirect noisy DeerFlow print messages to :func:`logger.trace`.

    Called once; subsequent calls are no-ops (guarded by ``builtins._deer_flow_print_patched``).
    """
    if getattr(builtins, "_deer_flow_print_patched", False):
        return
    _orig_print = builtins.print

    def _filtered_print(*args: object, **kwargs: object) -> None:
        text = " ".join(str(a) for a in args)
        if any(text.startswith(p) for p in _DEER_FLOW_SUPPRESS_PREFIXES):
            logger.trace(f"[DeerFlow] {text}")
        else:
            _orig_print(*args, **kwargs)

    builtins.print = _filtered_print  # type: ignore[assignment]
    builtins._deer_flow_print_patched = True  # type: ignore[attr-defined]
    logger.debug("DeerFlow print suppression patch applied")


def _get_checkpointer_db_path() -> Path:
    """Resolve the SQLite checkpointer path using genai-tk's config paths."""
    try:
        from genai_tk.utils.config_mngr import global_config

        kv_dir = global_config().get_dir_path("paths.data") / "kv_store"
    except Exception:
        kv_dir = Path("data") / "kv_store"
    kv_dir.mkdir(parents=True, exist_ok=True)
    return kv_dir / "deerflow_checkpoints.db"


# ---------------------------------------------------------------------------
# Embedded client
# ---------------------------------------------------------------------------


class EmbeddedDeerFlowClient:
    """In-process adapter over DeerFlow's embedded ``DeerFlowClient``.

    Injects ``DEER_FLOW_PATH/backend`` to ``sys.path``, creates a
    ``SqliteSaver`` checkpointer for session-scoped multi-turn memory, and
    translates DeerFlow ``StreamEvent`` objects into genai-tk typed events.

    No LangGraph Server or Gateway API processes are required.
    """

    def __init__(
        self,
        config_path: str | Path,
        *,
        model_name: str | None = None,
    ) -> None:
        """Initialize the embedded client.

        Args:
            config_path: Path to the deer-flow ``config.yaml`` written by
                ``setup_deer_flow_config()``.
            model_name: Default model name override passed to DeerFlow.
        """
        _ensure_deer_flow_on_path()

        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import]

            db_path = _get_checkpointer_db_path()
            # Create connection directly — from_conn_string() is a context
            # manager whose generator would close the connection on GC.
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            checkpointer = SqliteSaver(conn)
            logger.debug(f"Using SqliteSaver checkpointer at {db_path}")
        except ImportError:
            from langgraph.checkpoint.memory import MemorySaver  # type: ignore[import]

            checkpointer = MemorySaver()
            logger.warning(
                "langgraph-checkpoint-sqlite not installed — using in-memory checkpointer. "
                "Multi-turn state will be lost when the process exits. "
                "Install with: uv add langgraph-checkpoint-sqlite"
            )

        from src.client import DeerFlowClient as _DeerFlowClient  # type: ignore[import]

        self._checkpointer = checkpointer
        self._client = _DeerFlowClient(
            config_path=str(config_path),
            checkpointer=self._checkpointer,
            model_name=model_name,
        )
        logger.debug(f"EmbeddedDeerFlowClient ready — config={config_path}")

    def clear_thread(self, thread_id: str) -> None:
        """Delete all checkpointer data for a thread so the next run starts fresh.

        Useful in single-shot mode where a stable ``thread_id`` is reused for
        Docker sandbox container reuse, but conversation state should not carry
        over between separate CLI invocations.
        """
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import]

            if isinstance(self._checkpointer, SqliteSaver):
                conn = self._checkpointer.conn
                conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
                conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
                conn.commit()
                logger.debug(f"Cleared checkpointer state for thread {thread_id}")
        except Exception as e:
            logger.debug(f"Could not clear checkpointer thread (non-critical): {e}")

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream_message(
        self,
        thread_id: str,
        user_input: str,
        *,
        model_name: str | None = None,
        mode: str = "flash",
        subagent_enabled: bool | None = None,
        plan_mode: bool | None = None,
    ) -> AsyncIterator[TokenEvent | NodeEvent | ToolCallEvent | ToolResultEvent | ErrorEvent]:
        """Stream a single conversation turn, yielding typed events.

        Bridges the synchronous ``DeerFlowClient.stream()`` generator to an
        async generator via a background thread + ``queue.Queue``.

        Args:
            thread_id: Conversation thread ID (scopes checkpointer memory).
            user_input: User message text.
            model_name: Override model for this turn.
            mode: Agent mode (flash | thinking | pro | ultra).
            subagent_enabled: Override subagent flag; falls back to mode default.
            plan_mode: Override plan_mode flag; falls back to mode default.

        Yields:
            Typed event objects: ``TokenEvent``, ``ToolCallEvent``,
            ``ToolResultEvent``, or ``ErrorEvent``.
        """
        flags = _mode_flags(mode)
        kwargs: dict[str, Any] = {
            "thinking_enabled": flags["thinking_enabled"],
            "plan_mode": plan_mode if plan_mode is not None else flags["is_plan_mode"],
            "subagent_enabled": subagent_enabled if subagent_enabled is not None else flags["subagent_enabled"],
        }
        if model_name:
            kwargs["model_name"] = model_name

        event_queue: queue.Queue = queue.Queue()
        _SENTINEL = object()

        def _run_sync() -> None:
            try:
                for ev in self._client.stream(user_input, thread_id=thread_id, **kwargs):
                    event_queue.put(ev)
            except Exception as exc:
                event_queue.put(exc)
            finally:
                event_queue.put(_SENTINEL)

        threading.Thread(target=_run_sync, daemon=True).start()

        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, event_queue.get)
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                yield ErrorEvent(message=str(item))
                break
            for translated in _translate_event(item):
                yield translated

    # ------------------------------------------------------------------
    # Skills / memory / MCP delegation
    # ------------------------------------------------------------------

    def list_skills(self) -> list[dict]:
        """Return all skills with their enabled state."""
        return self._client.list_skills().get("skills", [])

    def update_skill(self, skill_name: str, *, enabled: bool) -> None:
        """Enable or disable a named skill."""
        self._client.update_skill(skill_name, enabled=enabled)

    def get_memory_status(self) -> dict:
        """Return memory config and data."""
        return self._client.get_memory_status()

    def update_mcp_config(self, mcp_config: dict) -> None:
        """Replace the MCP server configuration."""
        self._client.update_mcp_config(mcp_config)

    def reset_agent(self) -> None:
        """Force agent recreation on the next ``stream_message`` call."""
        self._client.reset_agent()


# ---------------------------------------------------------------------------
# Event translation
# ---------------------------------------------------------------------------


# Cached reference to DeerFlow's StreamEvent class (populated on first use).
_DFStreamEvent: type | None = None


def _translate_event(ev: Any) -> list[StreamEvent]:
    """Translate one DeerFlow ``StreamEvent`` into zero or more genai-tk events.

    Args:
        ev: Raw event from ``DeerFlowClient.stream()``.

    Returns:
        List of typed event objects (may be empty for ``values`` / ``end``).
    """
    global _DFStreamEvent  # noqa: PLW0603
    if _DFStreamEvent is None:
        try:
            from src.client import StreamEvent as _cls  # type: ignore[import]

            _DFStreamEvent = _cls
        except ImportError:
            return []

    if not isinstance(ev, _DFStreamEvent):
        return []

    if ev.type == "messages-tuple":
        data = ev.data
        msg_type = data.get("type", "")

        if msg_type == "ai":
            results: list[StreamEvent] = []
            content = data.get("content", "")
            if content:
                results.append(TokenEvent(data=content))
            for tc in data.get("tool_calls", []):
                results.append(
                    ToolCallEvent(
                        tool_name=tc.get("name", ""),
                        args=tc.get("args", {}),
                        call_id=tc.get("id", ""),
                    )
                )
            return results

        if msg_type == "tool":
            return [
                ToolResultEvent(
                    tool_name=data.get("name", ""),
                    content=str(data.get("content", ""))[:500],
                    call_id=data.get("tool_call_id", ""),
                )
            ]

    # "values" and "end" events carry no displayable incremental info
    return []


# ---------------------------------------------------------------------------
# Response text cleanup
# ---------------------------------------------------------------------------

# Regex that matches model-generated reasoning/commentary markers.
# Some models (e.g. gpt-oss-120b) embed chain-of-thought tags like
# ``assistantanalysis``, ``assistantcommentary to=functions.X code{ … }``,
# ``assistantfinal`` directly in their text output.  Only the text after
# the last ``assistantfinal`` marker is the user-facing response.

_FINAL_MARKER_RE = re.compile(r"assistantfinal", re.IGNORECASE)


def strip_reasoning_markers(text: str) -> str:
    """Extract the user-facing response from model output that contains reasoning markers.

    If the text contains ``assistantfinal`` markers, returns only the text that
    follows the *last* such marker.  Otherwise returns the original text unchanged.
    """
    matches = list(_FINAL_MARKER_RE.finditer(text))
    if matches:
        last_match = matches[-1]
        return text[last_match.end() :].strip()
    return text
