"""Unit tests for converters between HarnessEvent and AUIP models."""

from __future__ import annotations

from genai_tk.agents.harness.events import (
    ClarificationEvent,
    EndEvent,
    ErrorEvent,
    NodeEvent,
    ThinkingEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from genai_tk.protocols.auip.converters import auip_to_harness_event, harness_event_to_auip
from genai_tk.protocols.auip.models import (
    AuipEndEvent,
    AuipErrorEvent,
    AuipHitlRequestEvent,
    AuipPhaseEvent,
    AuipThinkingEvent,
    AuipTokenEvent,
    AuipToolCallEvent,
    AuipToolResultEvent,
    AuipUsageTelemetryEvent,
)


def test_token_conversion() -> None:
    h_ev = TokenEvent(text="chunk")
    a_ev = harness_event_to_auip(h_ev, session_id="s1")
    assert isinstance(a_ev, AuipTokenEvent)
    assert a_ev.text == "chunk"
    assert a_ev.session_id == "s1"

    back = auip_to_harness_event(a_ev)
    assert isinstance(back, TokenEvent)
    assert back.text == "chunk"


def test_thinking_conversion() -> None:
    h_ev = ThinkingEvent(text="think")
    a_ev = h_ev.to_auip(session_id="s1")
    assert isinstance(a_ev, AuipThinkingEvent)
    assert a_ev.text == "think"


def test_node_conversion() -> None:
    h_ev = NodeEvent(node="researcher", state={"k": "v"})
    a_ev = harness_event_to_auip(h_ev)
    assert isinstance(a_ev, AuipPhaseEvent)
    assert a_ev.phase == "researcher"
    assert a_ev.state == {"k": "v"}

    back = auip_to_harness_event(a_ev)
    assert isinstance(back, NodeEvent)
    assert back.node == "researcher"


def test_tool_call_conversion() -> None:
    h_ev = ToolCallEvent(tool_name="search", call_id="123", args={"q": "test"})
    a_ev = harness_event_to_auip(h_ev)
    assert isinstance(a_ev, AuipToolCallEvent)
    assert a_ev.tool_name == "search"
    assert a_ev.call_id == "123"
    assert a_ev.args == {"q": "test"}

    back = auip_to_harness_event(a_ev)
    assert isinstance(back, ToolCallEvent)
    assert back.tool_name == "search"


def test_tool_result_conversion() -> None:
    h_ev = ToolResultEvent(tool_name="search", call_id="123", content="results")
    a_ev = harness_event_to_auip(h_ev)
    assert isinstance(a_ev, AuipToolResultEvent)
    assert a_ev.content == "results"


def test_clarification_conversion() -> None:
    h_ev = ClarificationEvent(question="Which database?", clarification_type="choice")
    a_ev = harness_event_to_auip(h_ev)
    assert isinstance(a_ev, AuipHitlRequestEvent)
    assert a_ev.question == "Which database?"
    assert a_ev.clarification_type == "choice"


def test_usage_and_end_conversion() -> None:
    h_ev = UsageEvent(input_tokens=10, output_tokens=20)
    a_ev = harness_event_to_auip(h_ev)
    assert isinstance(a_ev, AuipUsageTelemetryEvent)
    assert a_ev.input_tokens == 10
    assert a_ev.output_tokens == 20

    end_ev = EndEvent()
    a_end = harness_event_to_auip(end_ev)
    assert isinstance(a_end, AuipEndEvent)

    err_ev = ErrorEvent(message="boom")
    a_err = harness_event_to_auip(err_ev)
    assert isinstance(a_err, AuipErrorEvent)
    assert a_err.message == "boom"
