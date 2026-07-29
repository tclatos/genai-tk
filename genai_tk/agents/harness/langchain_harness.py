"""LangChain harness adapter — wraps ``create_langchain_agent`` (react | deep | custom,
including DeepAgents SDK profiles) behind the shared :class:`BaseHarness` interface.

Uses LangGraph's standard ``astream_events`` API so react, deep (DeepAgents SDK),
and custom agent types are all supported uniformly — they are all compiled
LangGraph graphs under the hood.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

from genai_tk.agents.harness.base import BaseHarness
from genai_tk.agents.harness.events import (
    EndEvent,
    ErrorEvent,
    NodeEvent,
    StreamEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from genai_tk.agents.langchain.config import AgentProfileConfig
from genai_tk.utils.tracing import apply_harness_trace_metadata, get_monitoring_callbacks, setup_monitoring

# Chain-node names that are internal plumbing of the compiled graph rather than
# meaningful agent phases. These are emitted by ``astream_events`` as
# ``on_chain_start``/``on_chain_end`` but should NOT surface as ``NodeEvent`` —
# they either are the root graph invocation, the langchain react "agent"/"tools"
# step nodes (already covered by tool/token events), or generic wrappers.
_INTERNAL_NODE_NAMES: frozenset[str] = frozenset(
    {
        "LangGraph",
        "LangGraphAPI",
        "Agent",
        "agent",
        "Tools",
        "tools",
        "model",
        "should_continue",
    }
)


class LangChainHarness(BaseHarness):
    """Harness session backed by a LangChain/LangGraph compiled agent.

    Args:
        profile: Resolved agent profile (``type`` may be ``react``, ``deep``, or ``custom``).
        llm_override: LLM identifier that takes precedence over ``profile.llm``.
        force_memory_checkpointer: Use an in-process ``MemorySaver`` even if the
            profile specifies ``checkpointer.type: none`` (useful for interactive/chat use).
        extra_mcp: Additional MCP server names appended to the profile's
            ``mcp_servers`` (duplicates dropped) before the agent is built.
        extra_tools: Additional LangChain tools appended on top of the profile's
            own tools (e.g. tools a hosting MCP server already exposes).
    """

    name = "langchain"

    def __init__(
        self,
        profile: AgentProfileConfig,
        *,
        llm_override: str | None = None,
        force_memory_checkpointer: bool = False,
        extra_mcp: list[str] | None = None,
        extra_tools: list[Any] | None = None,
    ) -> None:
        self._profile = profile
        self._llm_override = llm_override
        self._force_memory_checkpointer = force_memory_checkpointer
        self._extra_mcp = list(extra_mcp or [])
        self._extra_tools = list(extra_tools or [])
        self._agent: Any = None

    async def _ensure_agent(self) -> Any:
        if self._agent is None:
            import os

            from genai_tk.agents.langchain.factory import create_langchain_agent
            from genai_tk.utils.tracing import HarnessTraceMetadata

            # Initialise monitoring backends (LangSmith env vars, LangFuse/OTEL
            # auto-instrumentation, local JSONL handler) before the agent runs.
            # Without this LANGSMITH_TRACING stays "false" and no backend
            # receives traces from the LangChain harness.
            setup_monitoring()

            # Append any extra MCP servers (deduped) the caller requested.
            if self._extra_mcp:
                existing = set(self._profile.mcp_servers)
                self._profile.mcp_servers = [
                    *self._profile.mcp_servers,
                    *(m for m in self._extra_mcp if m not in existing),
                ]

            llm_id = self._llm_override or self._profile.llm or "default"
            apply_harness_trace_metadata(
                HarnessTraceMetadata(
                    harness=self.name,
                    profile_name=self._profile.name,
                    model_name=llm_id,
                    environment=os.environ.get("GENAI_TK_ENV"),
                )
            )
            self._agent = await create_langchain_agent(
                self._profile,
                llm_override=self._llm_override,
                force_memory_checkpointer=self._force_memory_checkpointer,
                extra_tools=self._extra_tools or None,
            )
        return self._agent

    async def astream(self, message: str, *, thread_id: str | None = None) -> AsyncIterator[StreamEvent]:
        agent = await self._ensure_agent()
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id or self.default_thread_id},
            "recursion_limit": self._profile.recursion_limit,
        }
        # Attach monitoring callbacks (local JSONL log, LangFuse CallbackHandler)
        # so agent runs are traced alongside the env-var/OTEL backends.
        callbacks = get_monitoring_callbacks()
        if callbacks:
            config["callbacks"] = callbacks
        # Per-run accumulator: maps astream `run_id` → streamed-text buffer.
        # On `on_chat_model_end` we emit any un-streamed remainder of the final
        # AIMessage as a TokenEvent so non-streaming models still surface their
        # answer — and streamed models don't get their text duplicated.
        streamed_per_run: dict[str, str] = {}
        try:
            async for raw_event in agent.astream_events({"messages": message}, config=config, version="v2"):
                for translated in _translate_langchain_event(raw_event, streamed_per_run):
                    yield translated
        except Exception as exc:
            logger.opt(exception=True).warning(f"LangChainHarness stream error: {exc}")
            yield ErrorEvent(message=str(exc))
        yield EndEvent()

    async def aclose(self) -> None:
        backend = getattr(self._agent, "_backend", None)
        stop = getattr(backend, "stop", None)
        if callable(stop):
            await stop()

    async def get_graph(self) -> Any:
        """Return the compiled LangGraph graph backing this harness (built lazily)."""
        return await self._ensure_agent()

    async def get_checkpointer(self) -> Any:
        """Return the graph's checkpointer, if one was configured for the profile."""
        agent = await self._ensure_agent()
        return getattr(agent, "checkpointer", None)


def _translate_langchain_event(
    ev: dict[str, Any],
    streamed_per_run: dict[str, str] | None = None,
) -> list[StreamEvent]:
    """Translate one LangGraph ``astream_events`` (v2) event into harness events.

    A single raw event may yield zero, one, or several :class:`StreamEvent`s:

    - ``on_chat_model_stream`` → one :class:`TokenEvent` per non-empty chunk.
    - ``on_chat_model_end``     → a :class:`TokenEvent` for any un-streamed tail
      of the final AIMessage (so non-streaming models still surface their
      answer) followed by a :class:`UsageEvent` when usage metadata is present.
    - ``on_tool_start`` / ``on_tool_end`` → the matching tool events.
    - ``on_chain_start``                   → a :class:`NodeEvent` for non-root,
      meaningful graph phases.

    When ``streamed_per_run`` is provided, the per-``run_id`` streamed-text
    buffer is updated on ``on_chat_model_stream`` and consumed on
    ``on_chat_model_end``, so streamed chunks are not re-emitted as a tail.

    Args:
        ev: Raw event dict from ``Runnable.astream_events``.
        streamed_per_run: Mutable accumulator mapping ``run_id`` → text already
            streamed out via ``on_chat_model_stream`` chunks.  When ``None``,
            tail-flush tracking is skipped (every ``on_chat_model_end`` emits
            its full text as a single TokenEvent).

    Returns:
        Zero or more typed :class:`StreamEvent`s to yield upstream.
    """
    event_type = ev.get("event", "")
    data = ev.get("data", {}) or {}
    run_id = ev.get("run_id", "")

    if event_type == "on_chat_model_stream":
        chunk = data.get("chunk")
        content = getattr(chunk, "content", "") if chunk is not None else ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        text = str(content) if content else ""
        if not text:
            return []
        if streamed_per_run is not None:
            streamed_per_run[run_id] = streamed_per_run.get(run_id, "") + text
        return [TokenEvent(text=text)]

    if event_type == "on_tool_start":
        return [
            ToolCallEvent(
                tool_name=ev.get("name", ""),
                args=data.get("input", {}) or {},
                call_id=run_id,
            )
        ]

    if event_type == "on_tool_end":
        output = data.get("output")
        content = getattr(output, "content", None)
        if content is None:
            content = str(output) if output is not None else ""
        return [
            ToolResultEvent(
                tool_name=ev.get("name", ""),
                content=str(content),
                call_id=run_id,
            )
        ]

    if event_type == "on_chat_model_end":
        events: list[StreamEvent] = []
        output = data.get("output")
        full = _ai_message_text(output) if output is not None else ""
        if full:
            streamed = streamed_per_run.pop(run_id, "") if streamed_per_run is not None else ""
            if streamed and full.startswith(streamed):
                tail = full[len(streamed) :]
            elif streamed:
                # Chunks were emitted but don't align with the final message;
                # assume streaming already delivered the content and avoid dupes.
                tail = ""
            else:
                tail = full
            if tail:
                events.append(TokenEvent(text=tail))
        usage = getattr(output, "usage_metadata", None) if output is not None else None
        if usage:
            events.append(
                UsageEvent(
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                )
            )
        return events

    if event_type == "on_chain_start":
        name = ev.get("name", "")
        parent_ids = ev.get("parent_ids") or []
        if name and parent_ids and name not in _INTERNAL_NODE_NAMES:
            return [NodeEvent(node=str(name))]
        return []

    return []


def _ai_message_text(output: Any) -> str:
    """Flatten an ``on_chat_model_end`` output into its visible text content."""
    msg = output
    if hasattr(msg, "model_response"):
        msg = msg.model_response
    if hasattr(msg, "result"):
        msgs = msg.result
        msg = msgs[0] if msgs else msg
    content = getattr(msg, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)
