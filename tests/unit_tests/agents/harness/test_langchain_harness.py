"""Unit tests for LangChain harness event translation.

Covers the ``NodeEvent`` parity between DeepAgents (``type: deep``) LangChain
profiles and DeerFlow: graph phase nodes emitted via ``on_chain_start`` are
translated into :class:`~genai_tk.agents.harness.events.NodeEvent`, while
internal plumbing nodes and the root invocation are filtered out.

These tests call :func:`_translate_langchain_event` directly with synthetic
``astream_events`` (v2) event dicts — no live LLM or deepagents install needed.
"""

import pytest

from genai_tk.agents.harness.events import (
    NodeEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from genai_tk.agents.harness.langchain_harness import _translate_langchain_event


def _chain_start(name: str, *, parent_ids: list[str] | None = None) -> dict:
    return {
        "event": "on_chain_start",
        "name": name,
        "parent_ids": parent_ids if parent_ids is not None else ["root-run-id"],
        "data": {},
        "run_id": "run-" + name,
    }


def test_chain_start_meaningful_phase_emits_node_event() -> None:
    """A DeepAgents planner/researcher/coder/reporter node surfaces as NodeEvent."""
    for phase in ("planner", "researcher", "coder", "reporter"):
        ev = _translate_langchain_event(_chain_start(phase))
        assert isinstance(ev, NodeEvent)
        assert ev.node == phase


def test_chain_start_root_invocation_not_emitted() -> None:
    """The root graph invocation (no parent_ids) is filtered out."""
    ev = _translate_langchain_event(_chain_start("LangGraph", parent_ids=[]))
    assert ev is None


def test_chain_start_internal_plumbing_names_filtered() -> None:
    """Internal react plumbing nodes ('agent', 'tools', 'model', ...) do not surface."""
    for internal in ("LangGraph", "Agent", "agent", "Tools", "tools", "model", "should_continue"):
        assert _translate_langchain_event(_chain_start(internal)) is None


def test_chat_model_stream_emits_token_event() -> None:
    chunk = type("Chunk", (), {"content": "hello"})()
    ev = _translate_langchain_event({"event": "on_chat_model_stream", "data": {"chunk": chunk}})
    assert isinstance(ev, TokenEvent)
    assert ev.text == "hello"


def test_tool_start_and_end_events() -> None:
    start = _translate_langchain_event(
        {"event": "on_tool_start", "name": "search", "run_id": "r1", "data": {"input": {"q": "x"}}}
    )
    assert isinstance(start, ToolCallEvent)
    assert start.tool_name == "search"
    assert start.args == {"q": "x"}

    output = type("Out", (), {"content": "result text"})()
    end = _translate_langchain_event(
        {"event": "on_tool_end", "name": "search", "run_id": "r1", "data": {"output": output}}
    )
    assert isinstance(end, ToolResultEvent)
    assert end.content == "result text"
    assert end.call_id == "r1"


def test_tool_result_content_not_truncated() -> None:
    """Long tool outputs are passed through untruncated; rendering truncation is the UI's job."""
    output = type("Out", (), {"content": "x" * 5000})()
    end = _translate_langchain_event(
        {"event": "on_tool_end", "name": "search", "run_id": "r1", "data": {"output": output}}
    )
    assert isinstance(end, ToolResultEvent)
    assert len(end.content) == 5000


def test_chat_model_end_emits_usage() -> None:
    output = type("Out", (), {"usage_metadata": {"input_tokens": 10, "output_tokens": 5}})()
    ev = _translate_langchain_event({"event": "on_chat_model_end", "data": {"output": output}})
    assert isinstance(ev, UsageEvent)
    assert ev.input_tokens == 10
    assert ev.output_tokens == 5


def test_unknown_event_returns_none() -> None:
    assert _translate_langchain_event({"event": "on_retriever_stream", "data": {}}) is None


@pytest.mark.unit
class TestBaseHarnessDefaultThreadId:
    """The default thread id policy lives on BaseHarness, not as a magic literal in adapters."""

    def test_default_thread_id_is_default_string(self) -> None:
        from genai_tk.agents.harness.base import BaseHarness

        assert BaseHarness.default_thread_id == "default"
