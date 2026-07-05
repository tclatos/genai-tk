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
        assert len(ev) == 1
        assert isinstance(ev[0], NodeEvent)
        assert ev[0].node == phase


def test_chain_start_root_invocation_not_emitted() -> None:
    """The root graph invocation (no parent_ids) is filtered out."""
    ev = _translate_langchain_event(_chain_start("LangGraph", parent_ids=[]))
    assert ev == []


def test_chain_start_internal_plumbing_names_filtered() -> None:
    """Internal react plumbing nodes ('agent', 'tools', 'model', ...) do not surface."""
    for internal in ("LangGraph", "Agent", "agent", "Tools", "tools", "model", "should_continue"):
        assert _translate_langchain_event(_chain_start(internal)) == []


def test_chat_model_stream_emits_token_event() -> None:
    chunk = type("Chunk", (), {"content": "hello"})()
    ev = _translate_langchain_event({"event": "on_chat_model_stream", "data": {"chunk": chunk}})
    assert len(ev) == 1
    assert isinstance(ev[0], TokenEvent)
    assert ev[0].text == "hello"


def test_tool_start_and_end_events() -> None:
    start = _translate_langchain_event(
        {"event": "on_tool_start", "name": "search", "run_id": "r1", "data": {"input": {"q": "x"}}}
    )
    assert len(start) == 1
    assert isinstance(start[0], ToolCallEvent)
    assert start[0].tool_name == "search"
    assert start[0].args == {"q": "x"}

    output = type("Out", (), {"content": "result text"})()
    end = _translate_langchain_event(
        {"event": "on_tool_end", "name": "search", "run_id": "r1", "data": {"output": output}}
    )
    assert len(end) == 1
    assert isinstance(end[0], ToolResultEvent)
    assert end[0].content == "result text"
    assert end[0].call_id == "r1"


def test_tool_result_content_not_truncated() -> None:
    """Long tool outputs are passed through untruncated; rendering truncation is the UI's job."""
    output = type("Out", (), {"content": "x" * 5000})()
    end = _translate_langchain_event(
        {"event": "on_tool_end", "name": "search", "run_id": "r1", "data": {"output": output}}
    )
    assert len(end) == 1
    assert isinstance(end[0], ToolResultEvent)
    assert len(end[0].content) == 5000


def test_chat_model_end_emits_usage() -> None:
    output = type("Out", (), {"usage_metadata": {"input_tokens": 10, "output_tokens": 5}})()
    ev = _translate_langchain_event({"event": "on_chat_model_end", "data": {"output": output}})
    usage = [e for e in ev if isinstance(e, UsageEvent)]
    assert len(usage) == 1
    assert usage[0].input_tokens == 10
    assert usage[0].output_tokens == 5


def test_chat_model_end_flushes_unstreamed_text() -> None:
    """Without streaming chunks, on_chat_model_end emits the full text as a TokenEvent."""
    output = type("Out", (), {"content": "final answer"})()
    ev = _translate_langchain_event({"event": "on_chat_model_end", "run_id": "r1", "data": {"output": output}})
    tokens = [e for e in ev if isinstance(e, TokenEvent)]
    assert len(tokens) == 1
    assert tokens[0].text == "final answer"


def test_chat_model_end_no_duplicate_text_after_streaming() -> None:
    """When chunks were streamed for a run_id, the end event must not re-emit them."""
    streamed_per_run: dict[str, str] = {}
    chunk = type("Chunk", (), {"content": "Hello"})()
    res = _translate_langchain_event(
        {"event": "on_chat_model_stream", "run_id": "r1", "data": {"chunk": chunk}},
        streamed_per_run,
    )
    assert res and isinstance(res[-1], TokenEvent)
    output = type("Out", (), {"content": "Hello world"})()
    end = _translate_langchain_event(
        {"event": "on_chat_model_end", "run_id": "r1", "data": {"output": output}},
        streamed_per_run,
    )
    tokens = [e for e in end if isinstance(e, TokenEvent)]
    assert len(tokens) == 1
    assert tokens[0].text == " world"


def test_unknown_event_returns_empty_list() -> None:
    assert _translate_langchain_event({"event": "on_retriever_stream", "data": {}}) == []


@pytest.mark.unit
class TestBaseHarnessDefaultThreadId:
    """The default thread id policy lives on BaseHarness, not as a magic literal in adapters."""

    def test_default_thread_id_is_default_string(self) -> None:
        from genai_tk.agents.harness.base import BaseHarness

        assert BaseHarness.default_thread_id == "default"
