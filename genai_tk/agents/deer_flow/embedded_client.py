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

    # Access any upstream DeerFlowClient API directly:
    print(client.client.list_skills())
    print(client.client.get_memory_status())
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


@dataclass
class ClarificationEvent:
    """DeerFlow paused and is asking the user a clarifying question.

    The agent has called ``ask_clarification`` and halted.  The caller should
    display ``question`` to the user, collect a reply, and send it as the next
    message on the same ``thread_id``.
    """

    kind: str = "clarification"
    question: str = ""
    clarification_type: str = "missing_info"
    context: str = ""


# Union type alias matching the HTTP client's public API
StreamEvent = TokenEvent | NodeEvent | ToolCallEvent | ToolResultEvent | ErrorEvent | ClarificationEvent


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
    """Add the DeerFlow package directory to :data:`sys.path` if absent.

    Supports both the legacy layout (``DEER_FLOW_PATH/backend/src/``) and the
    modern workspace layout (``DEER_FLOW_PATH/backend/packages/harness/``).

    Returns:
        Resolved backend directory path.
    """
    df_path = os.environ.get("DEER_FLOW_PATH", "")
    if not df_path:
        raise RuntimeError("DEER_FLOW_PATH is not set. Set it to the root of your deer-flow clone.")
    backend_path = Path(df_path) / "backend"
    if not backend_path.exists():
        raise RuntimeError(f"DEER_FLOW_PATH/backend not found: {backend_path}")

    # Modern layout: backend/packages/harness/deerflow/
    harness_path = backend_path / "packages" / "harness"
    if harness_path.exists():
        pkg_str = str(harness_path)
        if pkg_str not in sys.path:
            sys.path.insert(0, pkg_str)
            logger.debug(f"Added {pkg_str} to sys.path (modern layout)")
    else:
        # Legacy layout: backend/src/
        backend_str = str(backend_path)
        if backend_str not in sys.path:
            sys.path.insert(0, backend_str)
            logger.debug(f"Added {backend_str} to sys.path (legacy layout)")

    _patch_deer_flow_print()
    _suppress_deer_flow_logging()
    _patch_deer_flow_config_caching()
    _check_deer_flow_compatibility()
    return backend_path


def _patch_deer_flow_config_caching() -> None:
    """Fix upstream config caching so explicit config_path is honoured.

    When ``reload_app_config(config_path)`` is called with an explicit path,
    the subsequent ``get_app_config()`` call must NOT re-resolve from CWD.
    Upstream sets ``_app_config_is_custom = False`` unconditionally — we
    patch ``_load_and_cache_app_config`` to set it to ``True`` when an
    explicit path is given.

    This is a no-op if the upstream code is already fixed.
    """
    try:
        try:
            import deerflow.config.app_config as _cfg_mod  # type: ignore[import]
        except ImportError:
            import src.config as _cfg_mod  # type: ignore[import]

        _orig = getattr(_cfg_mod, "_load_and_cache_app_config", None)
        if _orig is None or getattr(_cfg_mod, "_genai_tk_patched", False):
            return

        def _patched(config_path: str | None = None):  # type: ignore[override]
            result = _orig(config_path)
            if config_path is not None:
                _cfg_mod._app_config_is_custom = True
            return result

        _cfg_mod._load_and_cache_app_config = _patched
        _cfg_mod._genai_tk_patched = True
        logger.debug("Patched deer-flow _load_and_cache_app_config for explicit config_path support")
    except Exception as exc:
        logger.debug(f"Could not patch deer-flow config caching (non-critical): {exc}")


def _suppress_deer_flow_logging() -> None:
    """Raise DeerFlow's standard-library loggers to CRITICAL.

    DeerFlow modules use ``logging.getLogger(__name__)`` under the ``src``
    or ``deerflow`` namespace. Their INFO/ERROR messages (e.g. sandbox
    file-not-found 404s, agent internals) are not actionable for CLI users
    and clutter the output. Tool errors are already surfaced via
    ``ToolResultEvent``.
    """
    import logging as _std_logging

    _std_logging.getLogger("src").setLevel(_std_logging.CRITICAL)
    _std_logging.getLogger("deerflow").setLevel(_std_logging.CRITICAL)


_DEER_FLOW_SUPPRESS_PREFIXES = (
    "Lazy acquiring sandbox",
    "Acquiring sandbox",
    "Memory update timer",
    "Memory update queued",
    "Generated thread title",
    "Failed to read file",
    "Failed to list directory",
    "[ClarificationMiddleware]",
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
# Compatibility checks
# ---------------------------------------------------------------------------

_compat_checked = False


def _check_deer_flow_compatibility() -> None:
    """Run once to detect common deer-flow compatibility issues and warn early.

    Checks performed:
    - Clone age (warn if git HEAD is older than 30 days)
    - DeerFlowClient constructor parameters (middlewares, available_skills)
    - Module layout consistency (DEER_FLOW_PATH points to modern vs. legacy)
    - AgentMiddleware availability (required for custom middleware injection)

    All issues are logged as warnings — nothing is fatal here.
    """
    global _compat_checked  # noqa: PLW0603
    if _compat_checked:
        return
    _compat_checked = True

    df_path = os.environ.get("DEER_FLOW_PATH", "")
    if not df_path:
        return

    issues: list[str] = []
    root = Path(df_path)

    # --- Clone freshness ---
    try:
        import subprocess

        result = subprocess.run(
            ["git", "log", "-1", "--format=%ct"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            import time

            commit_ts = int(result.stdout.strip())
            age_days = (time.time() - commit_ts) / 86400
            if age_days > 30:
                issues.append(
                    f"Deer-flow clone is {int(age_days)} days old. "
                    "Run 'make deer-flow-install' to pull the latest version."
                )
    except Exception:
        pass

    # --- Layout detection ---
    harness_path = root / "backend" / "packages" / "harness"
    legacy_src = root / "backend" / "src"
    is_modern = harness_path.exists()
    is_legacy = legacy_src.exists() and not is_modern

    if is_legacy:
        issues.append(
            "Deer-flow clone uses the legacy backend/src/ layout. "
            "The modern layout (backend/packages/harness/) adds middleware and "
            "skill filtering support. Run 'make deer-flow-install' to upgrade."
        )

    # --- DeerFlowClient API checks ---
    try:
        if is_modern:
            from deerflow.client import DeerFlowClient as _Cls  # type: ignore[import]
        else:
            from src.client import DeerFlowClient as _Cls  # type: ignore[import]

        import inspect

        params = set(inspect.signature(_Cls.__init__).parameters)
        for expected in ("middlewares", "available_skills"):
            if expected not in params:
                issues.append(
                    f"DeerFlowClient.__init__ is missing '{expected}' parameter. "
                    "Tests using this feature will be skipped. "
                    "Pull the latest deer-flow or check the PR that adds it."
                )
    except ImportError:
        issues.append(
            "Could not import DeerFlowClient — deer-flow dependencies may not be installed. "
            "Run 'make deer-flow-install'."
        )

    # --- AgentMiddleware availability ---
    try:
        from langchain.agents.middleware import AgentMiddleware  # noqa: F401
    except ImportError:
        issues.append(
            "langchain.agents.middleware.AgentMiddleware is not available. "
            "Custom middleware injection requires langchain >= 0.3.x. "
            "Run 'uv sync --group deer-flow' to update."
        )

    # --- Config module prefix consistency ---
    # Detect if config_bridge would use the wrong prefix
    try:
        from genai_tk.agents.deer_flow.config_bridge import _deer_flow_module_prefix

        pfx = _deer_flow_module_prefix()
        if is_modern and pfx == "src":
            issues.append(
                "Module prefix mismatch: deer-flow uses the modern layout but config_bridge "
                "resolved 'src' prefix. Generated config.yaml tool paths will be wrong."
            )
        elif is_legacy and pfx == "deerflow":
            issues.append(
                "Module prefix mismatch: deer-flow uses the legacy layout but config_bridge "
                "resolved 'deerflow' prefix. Generated config.yaml tool paths will be wrong."
            )
    except ImportError:
        pass

    if issues:
        header = "Deer-flow compatibility check detected potential issues:"
        details = "\n".join(f"  • {issue}" for issue in issues)
        logger.warning(f"{header}\n{details}")
    else:
        logger.debug("Deer-flow compatibility check passed (no issues detected)")


# ---------------------------------------------------------------------------
# Embedded client
# ---------------------------------------------------------------------------


class EmbeddedDeerFlowClient:
    """In-process adapter over DeerFlow's embedded ``DeerFlowClient``.

    Injects ``DEER_FLOW_PATH/backend`` to ``sys.path``, creates a
    ``SqliteSaver`` checkpointer for session-scoped multi-turn memory, and
    translates DeerFlow ``StreamEvent`` objects into genai-tk typed events.

    No LangGraph Server or Gateway API processes are required.

    The underlying :class:`src.client.DeerFlowClient` instance is
    available via the :attr:`client` property for direct access to all
    upstream APIs (memory, skills, MCP config, file uploads, etc.).
    """

    def __init__(
        self,
        config_path: str | Path,
        *,
        model_name: str | None = None,
        middlewares: list | None = None,
        available_skills: set[str] | None = None,
    ) -> None:
        """Initialize the embedded client.

        Args:
            config_path: Path to the deer-flow ``config.yaml`` written by
                ``setup_deer_flow_config()``.
            model_name: Default model name override passed to DeerFlow.
            middlewares: Optional list of instantiated middleware objects to
                inject into the agent (forwarded to ``DeerFlowClient``).
            available_skills: Optional set of skill names to make available.
                ``None`` means all discovered skills are available (default).
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

        try:
            from deerflow.client import DeerFlowClient as _DeerFlowClient  # type: ignore[import]
        except ImportError:
            from src.client import DeerFlowClient as _DeerFlowClient  # type: ignore[import]

        self._checkpointer = checkpointer

        # Build kwargs dynamically — the upstream DeerFlowClient evolves; pass
        # only parameters whose names appear in the constructor signature so
        # this wrapper works with both older and newer deer-flow installs.
        import inspect

        _supported = set(inspect.signature(_DeerFlowClient.__init__).parameters)
        _upstream_kwargs: dict[str, Any] = {
            "config_path": str(config_path),
            "checkpointer": self._checkpointer,
            "model_name": model_name,
        }
        if "middlewares" in _supported:
            _upstream_kwargs["middlewares"] = middlewares or []
        elif middlewares:
            logger.warning(
                "This deer-flow version does not support the 'middlewares' parameter — ignoring. "
                "Update your deer-flow clone to enable middleware injection."
            )
        if "available_skills" in _supported:
            _upstream_kwargs["available_skills"] = available_skills
        elif available_skills is not None:
            logger.warning(
                "This deer-flow version does not support the 'available_skills' parameter — ignoring. "
                "Update your deer-flow clone to enable skill filtering."
            )

        self._client = _DeerFlowClient(**_upstream_kwargs)
        self._middlewares_supported = "middlewares" in _supported
        self._available_skills_supported = "available_skills" in _supported
        logger.debug(f"EmbeddedDeerFlowClient ready — config={config_path}")

    @property
    def client(self):
        """Direct access to the underlying ``DeerFlowClient`` instance.

        Use this to call any upstream API not wrapped by this adapter:
        memory management, skill installation, file uploads, MCP config, etc.
        """
        return self._client

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
                msg = str(item) or f"{type(item).__name__} (no message)"
                yield ErrorEvent(message=msg)
                break
            for translated in _translate_event(item):
                yield translated

    # ------------------------------------------------------------------
    # Skills / memory / MCP delegation
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        """Return all available models from configuration."""
        return self._client.list_models().get("models", [])

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
            from deerflow.client import StreamEvent as _cls  # type: ignore[import]
        except ImportError:
            try:
                from src.client import StreamEvent as _cls  # type: ignore[import]
            except ImportError:
                return []

        _DFStreamEvent = _cls

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
            tool_name = data.get("name", "")
            content = str(data.get("content", ""))
            # ask_clarification is intercepted by ClarificationMiddleware and halts
            # the graph — surface it as a dedicated event so callers can implement HITL.
            if tool_name == "ask_clarification":
                return [ClarificationEvent(question=content)]
            return [
                ToolResultEvent(
                    tool_name=tool_name,
                    content=content[:500],
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
