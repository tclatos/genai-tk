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
    """

    name = "langchain"

    def __init__(
        self,
        profile: AgentProfileConfig,
        *,
        llm_override: str | None = None,
        force_memory_checkpointer: bool = False,
    ) -> None:
        self._profile = profile
        self._llm_override = llm_override
        self._force_memory_checkpointer = force_memory_checkpointer
        self._agent: Any = None

    async def _ensure_agent(self) -> Any:
        if self._agent is None:
            import os

            from genai_tk.agents.langchain.factory import create_langchain_agent
            from genai_tk.utils.tracing import HarnessTraceMetadata, apply_harness_trace_metadata

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
            )
        return self._agent

    async def astream(self, message: str, *, thread_id: str | None = None) -> AsyncIterator[StreamEvent]:
        agent = await self._ensure_agent()
        config = {"configurable": {"thread_id": thread_id or self.default_thread_id}}
        try:
            async for raw_event in agent.astream_events({"messages": message}, config=config, version="v2"):
                translated = _translate_langchain_event(raw_event)
                if translated is not None:
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


def _translate_langchain_event(ev: dict[str, Any]) -> StreamEvent | None:
    """Translate one LangGraph ``astream_events`` (v2) event into a harness event.

    Args:
        ev: Raw event dict from ``Runnable.astream_events``.

    Returns:
        A typed :class:`StreamEvent`, or ``None`` for events with no
        displayable incremental info.
    """
    event_type = ev.get("event", "")
    data = ev.get("data", {}) or {}

    if event_type == "on_chat_model_stream":
        chunk = data.get("chunk")
        content = getattr(chunk, "content", "") if chunk is not None else ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        if content:
            return TokenEvent(text=str(content))
        return None

    if event_type == "on_tool_start":
        return ToolCallEvent(
            tool_name=ev.get("name", ""),
            args=data.get("input", {}) or {},
            call_id=ev.get("run_id", ""),
        )

    if event_type == "on_tool_end":
        output = data.get("output")
        content = getattr(output, "content", None)
        if content is None:
            content = str(output) if output is not None else ""
        return ToolResultEvent(
            tool_name=ev.get("name", ""),
            content=str(content),
            call_id=ev.get("run_id", ""),
        )

    if event_type == "on_chat_model_end":
        output = data.get("output")
        usage = getattr(output, "usage_metadata", None) if output is not None else None
        if usage:
            return UsageEvent(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )
        return None

    if event_type == "on_chain_start":
        name = ev.get("name", "")
        parent_ids = ev.get("parent_ids") or []
        # Emit a NodeEvent only for non-root, meaningful graph phases.
        # The root graph invocation has no parent; internal react plumbing
        # node names are filtered by _INTERNAL_NODE_NAMES.
        if name and parent_ids and name not in _INTERNAL_NODE_NAMES:
            return NodeEvent(node=str(name))
        return None

    return None
