"""Unit tests for harness event translation (LangChain + DeerFlow adapters)."""

from genai_tk.agents.harness.events import (
    ClarificationEvent,
    ErrorEvent,
    NodeEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from genai_tk.agents.harness.langchain_harness import _translate_langchain_event


class _FakeChunk:
    def __init__(self, content):
        self.content = content


class _FakeUsageOutput:
    def __init__(self, usage):
        self.usage_metadata = usage


def test_translate_chat_model_stream_yields_token_event() -> None:
    ev = {"event": "on_chat_model_stream", "data": {"chunk": _FakeChunk("Hello")}}
    result = _translate_langchain_event(ev)
    assert len(result) == 1
    assert isinstance(result[0], TokenEvent)
    assert result[0].text == "Hello"


def test_translate_chat_model_stream_empty_content_returns_empty_list() -> None:
    ev = {"event": "on_chat_model_stream", "data": {"chunk": _FakeChunk("")}}
    assert _translate_langchain_event(ev) == []


def test_translate_tool_start_yields_tool_call_event() -> None:
    ev = {"event": "on_tool_start", "name": "web_search", "run_id": "abc123", "data": {"input": {"query": "AI"}}}
    result = _translate_langchain_event(ev)
    assert len(result) == 1
    assert isinstance(result[0], ToolCallEvent)
    assert result[0].tool_name == "web_search"
    assert result[0].args == {"query": "AI"}
    assert result[0].call_id == "abc123"


def test_translate_tool_end_yields_tool_result_event() -> None:
    ev = {"event": "on_tool_end", "name": "web_search", "run_id": "abc123", "data": {"output": "some result"}}
    result = _translate_langchain_event(ev)
    assert len(result) == 1
    assert isinstance(result[0], ToolResultEvent)
    assert result[0].content == "some result"


def test_translate_chat_model_end_yields_usage_event() -> None:
    ev = {
        "event": "on_chat_model_end",
        "data": {"output": _FakeUsageOutput({"input_tokens": 10, "output_tokens": 5})},
    }
    result = _translate_langchain_event(ev)
    usage = [e for e in result if isinstance(e, UsageEvent)]
    assert len(usage) == 1
    assert usage[0].input_tokens == 10
    assert usage[0].output_tokens == 5


def test_translate_unknown_event_returns_empty_list() -> None:
    assert _translate_langchain_event({"event": "on_chain_start", "data": {}}) == []


def test_deerflow_translation_maps_all_event_kinds() -> None:
    from genai_tk.agents.deer_flow import embedded_client as dfc
    from genai_tk.agents.harness.deerflow_harness import _translate_deerflow_event

    assert _translate_deerflow_event(dfc.TokenEvent(data="hi")) == TokenEvent(text="hi")
    assert _translate_deerflow_event(dfc.NodeEvent(node="planner", state={"a": 1})) == NodeEvent(
        node="planner", state={"a": 1}
    )
    assert _translate_deerflow_event(
        dfc.ToolCallEvent(tool_name="search", args={"q": "x"}, call_id="1")
    ) == ToolCallEvent(tool_name="search", args={"q": "x"}, call_id="1")
    assert _translate_deerflow_event(
        dfc.ToolResultEvent(tool_name="search", content="result", call_id="1")
    ) == ToolResultEvent(tool_name="search", content="result", call_id="1")
    assert _translate_deerflow_event(dfc.ClarificationEvent(question="which one?")) == ClarificationEvent(
        question="which one?"
    )
    assert _translate_deerflow_event(dfc.ErrorEvent(message="boom")) == ErrorEvent(message="boom")
