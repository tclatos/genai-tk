"""HTTP client for the Deer-flow backend.

Wraps the LangGraph Server (port 2024) and Gateway API (port 8001).
Both ports are also reachable via nginx at 2026, but for dev use direct ports.

SSE streaming uses ``stream_mode=["updates","messages"]``:
- ``event: updates``  → ``{node_name: state_diff}`` — one per graph node
- ``event: messages`` → ``[AIMessageChunk, ...]``    — streamed tokens
- ``event: metadata`` → run metadata
- ``event: end``      → run complete

Example usage:
```python
client = DeerFlowClient()
thread_id = await client.create_thread()
async for event in client.stream_run(thread_id, "Tell me a joke"):
    if event.kind == "token":
        print(event.data, end="", flush=True)
    elif event.kind == "node":
        print(f"\\n[{event.node}]")
```
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx
from loguru import logger

# ---------------------------------------------------------------------------
# Typed events yielded by stream_run()
# ---------------------------------------------------------------------------


@dataclass
class TokenEvent:
    """A streamed text token from the AI message."""

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


# Union type alias
StreamEvent = TokenEvent | NodeEvent | ToolCallEvent | ToolResultEvent | ErrorEvent

# Nodes that are internal plumbing — suppress from trace display
_INTERNAL_NODES = frozenset(
    {
        "__start__",
        "__end__",
        "background_investigation_team",
        "lead_agent",
        # ThreadDataMiddleware / lifecycle hooks
        "ThreadDataMiddleware.before_agent",
        "UploadsMiddleware.before_agent",
        "SandboxMiddleware.before_agent",
        "DanglingToolCallMiddleware.before_model",
        "MemoryMiddleware.after_agent",
        "TitleMiddleware.after_agent",
    }
)


class DeerFlowClient:
    """Async HTTP client for a running Deer-flow server.

    Communicates with:
    - LangGraph Server (port ``langgraph_port``, default 2024)
    - Gateway API (port ``gateway_port``, default 8001)

    Both servers must already be running (see DeerFlowServerManager).
    """

    def __init__(
        self,
        langgraph_url: str = "http://localhost:2024",
        gateway_url: str = "http://localhost:8001",
        timeout: float = 120.0,
        assistant_id: str = "lead_agent",
    ) -> None:
        self._lg_url = langgraph_url.rstrip("/")
        self._gw_url = gateway_url.rstrip("/")
        self._timeout = timeout
        self._assistant_id = assistant_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lg(self, path: str) -> str:
        return f"{self._lg_url}{path}"

    def _gw(self, path: str) -> str:
        return f"{self._gw_url}{path}"

    @staticmethod
    def _client(timeout: float) -> httpx.AsyncClient:
        """Create an httpx client that bypasses any system proxy for local servers."""
        return httpx.AsyncClient(timeout=timeout, transport=httpx.AsyncHTTPTransport())

    # ------------------------------------------------------------------
    # Health / threads
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Return True if the LangGraph server responds to /info.

        Returns:
            True if server is reachable, False otherwise.
        """
        try:
            async with self._client(5.0) as client:
                r = await client.get(self._lg("/info"))
                return r.status_code < 500
        except Exception:
            return False

    async def create_thread(self, metadata: dict | None = None) -> str:
        """Create a new conversation thread.

        Args:
            metadata: Optional metadata dict to attach to the thread.

        Returns:
            The new thread_id string.
        """
        body = {"metadata": metadata or {}}
        async with self._client(30.0) as client:
            r = await client.post(self._lg("/threads"), json=body)
            r.raise_for_status()
        return r.json()["thread_id"]

    # ------------------------------------------------------------------
    # Streaming run
    # ------------------------------------------------------------------

    async def stream_run(
        self,
        thread_id: str,
        user_input: str,
        model_name: str | None = None,
        thinking_enabled: bool = False,
        is_plan_mode: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a run on the given thread.

        Yields:
            ``NodeEvent`` for each active graph node (filtered by _INTERNAL_NODES),
            ``TokenEvent`` for each streamed text token from the AI response,
            ``ErrorEvent`` on server-side errors.

        Args:
            thread_id: ID returned by create_thread().
            user_input: User message text.
            model_name: Override model for this run.
            thinking_enabled: Enable extended thinking (for supported models).
            is_plan_mode: Enable TodoList planning mode.
        """
        # LangGraph 0.6+ uses a top-level ``context`` dict instead of
        # ``config.configurable``.  The thread_id must be in context so that
        # deer-flow's ThreadDataMiddleware can read it.
        context: dict[str, Any] = {
            "thread_id": thread_id,
            "thinking_enabled": thinking_enabled,
            "is_plan_mode": is_plan_mode,
        }
        if model_name:
            context["model_name"] = model_name

        body: dict[str, Any] = {
            "assistant_id": self._assistant_id,
            "input": {"messages": [{"role": "user", "content": user_input}]},
            "stream_mode": ["updates", "messages"],
            "context": context,
        }

        url = self._lg(f"/threads/{thread_id}/runs/stream")
        seen_nodes: set[str] = set()
        # Track accumulated content per message id for messages/partial events
        # (LangGraph streams full accumulated state, not just deltas)
        msg_content_seen: dict[str, int] = {}

        async with self._client(self._timeout) as client:
            async with client.stream("POST", url, json=body) as resp:
                if resp.status_code >= 400:
                    text = await resp.aread()
                    yield ErrorEvent(message=f"HTTP {resp.status_code}: {text.decode()}")
                    return

                event_type: str = ""
                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue

                    if line.startswith("event:"):
                        event_type = line[len("event:") :].strip()
                        continue

                    if line.startswith("data:"):
                        payload = line[len("data:") :].strip()
                        if not payload or payload == "{}":
                            if event_type == "end":
                                return
                            continue
                        try:
                            data = json.loads(payload)
                        except json.JSONDecodeError:
                            logger.debug(f"[DeerFlowClient] Non-JSON SSE data: {payload[:80]}")
                            continue

                        # --------------- updates event ---------------
                        # Format: {node_name: state_diff, ...}
                        if event_type == "updates" and isinstance(data, dict):
                            for node_name, state_diff in data.items():
                                if node_name in _INTERNAL_NODES:
                                    continue

                                # Always emit node event (callers filter by show_trace)
                                yield NodeEvent(node=node_name, state=state_diff or {})
                                seen_nodes.add(node_name)

                                # Extract tool calls (from model node AIMessage with tool_calls)
                                if isinstance(state_diff, dict):
                                    for msg in state_diff.get("messages", []):
                                        if not isinstance(msg, dict):
                                            continue
                                        msg_type = msg.get("type", "")
                                        # Tool calls from AI message
                                        if msg_type in ("ai", "AIMessage", "AIMessageChunk"):
                                            for tc in msg.get("tool_calls", []):
                                                yield ToolCallEvent(
                                                    tool_name=tc.get("name", ""),
                                                    args=tc.get("args", {}),
                                                    call_id=tc.get("id", ""),
                                                )
                                        # Tool results from ToolMessage
                                        elif msg_type in ("tool", "ToolMessage"):
                                            content = msg.get("content", "")
                                            if isinstance(content, list):
                                                content = " ".join(
                                                    b.get("text", "")
                                                    for b in content
                                                    if isinstance(b, dict) and b.get("type") == "text"
                                                )
                                            yield ToolResultEvent(
                                                tool_name=msg.get("name", ""),
                                                content=str(content)[:500],
                                                call_id=msg.get("tool_call_id", ""),
                                            )

                        # --------------- messages/partial event ----------
                        # LangGraph 0.6+: each event carries the full accumulated
                        # message — emit only the new characters since last event.
                        elif event_type == "messages/partial" and isinstance(data, list):
                            for chunk in data:
                                if not isinstance(chunk, dict):
                                    continue
                                content = chunk.get("content", "")
                                msg_id = chunk.get("id", "")

                                if isinstance(content, str) and content:
                                    already_sent = msg_content_seen.get(msg_id, 0)
                                    new_text = content[already_sent:]
                                    if new_text:
                                        msg_content_seen[msg_id] = len(content)
                                        yield TokenEvent(data=new_text)
                                elif isinstance(content, list):
                                    for block in content:
                                        if isinstance(block, dict) and block.get("type") == "text":
                                            text = block.get("text", "")
                                            block_id = f"{msg_id}:{block.get('index', 0)}"
                                            already_sent = msg_content_seen.get(block_id, 0)
                                            new_text = text[already_sent:]
                                            if new_text:
                                                msg_content_seen[block_id] = len(text)
                                                yield TokenEvent(data=new_text)

                        # --------------- messages event (legacy delta format) --
                        elif event_type == "messages" and isinstance(data, list):
                            for chunk in data:
                                if not isinstance(chunk, dict):
                                    continue
                                content = chunk.get("content", "")
                                if isinstance(content, str) and content:
                                    yield TokenEvent(data=content)
                                elif isinstance(content, list):
                                    for block in content:
                                        if isinstance(block, dict) and block.get("type") == "text":
                                            text = block.get("text", "")
                                            if text:
                                                yield TokenEvent(data=text)

                        elif event_type == "error":
                            err_msg = (
                                data.get("message") or data.get("error", str(data))
                                if isinstance(data, dict)
                                else str(data)
                            )
                            yield ErrorEvent(message=err_msg)
                            return

    # ------------------------------------------------------------------
    # File uploads
    # ------------------------------------------------------------------

    async def upload_file(self, thread_id: str, file_path: str) -> dict:
        """Upload a file to a thread.

        Args:
            thread_id: Target thread ID.
            file_path: Local path to the file to upload.

        Returns:
            Server response dict (contains filename, path, size, etc.).
        """
        import pathlib

        path = pathlib.Path(file_path)
        async with self._client(60.0) as client:
            with open(path, "rb") as fh:
                files = {"file": (path.name, fh, "application/octet-stream")}
                r = await client.post(self._gw(f"/api/threads/{thread_id}/uploads"), files=files)
                r.raise_for_status()
        return r.json()

    async def list_uploads(self, thread_id: str) -> list[dict]:
        """List uploaded files for a thread.

        Args:
            thread_id: Target thread ID.

        Returns:
            List of upload info dicts.
        """
        async with self._client(15.0) as client:
            r = await client.get(self._gw(f"/api/threads/{thread_id}/uploads/list"))
            r.raise_for_status()
        return r.json().get("files", [])

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    async def get_artifact(self, thread_id: str, artifact_path: str) -> bytes:
        """Download a generated artifact.

        Args:
            thread_id: Thread that produced the artifact.
            artifact_path: Relative path within the thread artifacts directory.

        Returns:
            Raw bytes of the artifact.
        """
        async with self._client(30.0) as client:
            r = await client.get(self._gw(f"/api/threads/{thread_id}/artifacts/{artifact_path}"))
            r.raise_for_status()
        return r.content

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    async def list_skills(self) -> list[dict]:
        """List all available skills with their enabled status.

        Returns:
            List of skill info dicts (``name``, ``enabled``, ``description``).
        """
        async with self._client(15.0) as client:
            r = await client.get(self._gw("/api/skills"))
            r.raise_for_status()
        return r.json().get("skills", [])

    async def set_skill(self, skill_name: str, *, enabled: bool) -> None:
        """Enable or disable a skill.

        Args:
            skill_name: Name of the skill (e.g. ``public/deep-research``).
            enabled: True to enable, False to disable.
        """
        async with self._client(15.0) as client:
            r = await client.put(
                self._gw(f"/api/skills/{skill_name}"),
                json={"enabled": enabled},
            )
            r.raise_for_status()

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    async def get_memory_status(self) -> dict:
        """Return combined memory config and data.

        Returns:
            Dict with ``config`` and ``data`` keys.
        """
        async with self._client(15.0) as client:
            r = await client.get(self._gw("/api/memory/status"))
            r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # MCP config push
    # ------------------------------------------------------------------

    async def push_mcp_config(self, mcp_config: dict) -> None:
        """Replace the server's MCP configuration at runtime.

        Args:
            mcp_config: Dict in deer-flow extensions_config.json ``mcpServers`` format.
        """
        async with self._client(15.0) as client:
            r = await client.put(self._gw("/api/mcp/config"), json=mcp_config)
            r.raise_for_status()
