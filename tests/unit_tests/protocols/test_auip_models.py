"""Unit tests for AUIP models (outbound events and inbound messages)."""

from __future__ import annotations

import json

from genai_tk.protocols.auip.models import (
    AuipArtifactEvent,
    AuipHitlRequestEvent,
    AuipHitlResponse,
    AuipInterruptControl,
    AuipPhaseEvent,
    AuipThinkingEvent,
    AuipTokenEvent,
    AuipToolCallEvent,
    AuipToolDecision,
    AuipToolResultEvent,
    AuipUsageTelemetryEvent,
    AuipUserMessage,
)


def test_token_event_serialization() -> None:
    event = AuipTokenEvent(text="Hello world", session_id="s1")
    assert event.type == "stream.token"
    assert event.protocol == "auip/1.0"
    assert event.text == "Hello world"
    assert event.session_id == "s1"
    raw = event.model_dump_json()
    data = json.loads(raw)
    assert data["type"] == "stream.token"
    assert data["text"] == "Hello world"


def test_thinking_event_serialization() -> None:
    event = AuipThinkingEvent(text="Reasoning step...")
    assert event.type == "stream.thinking"
    assert event.text == "Reasoning step..."


def test_phase_event_serialization() -> None:
    event = AuipPhaseEvent(phase="planner", state={"todos": ["task 1"]})
    assert event.type == "phase.change"
    assert event.phase == "planner"
    assert event.state["todos"] == ["task 1"]


def test_tool_call_and_result_events() -> None:
    call = AuipToolCallEvent(tool_name="sql_db", call_id="c1", args={"query": "SELECT 1"})
    assert call.type == "tool.call"
    assert call.tool_name == "sql_db"
    assert call.args == {"query": "SELECT 1"}

    result = AuipToolResultEvent(tool_name="sql_db", call_id="c1", content="1 row", duration_ms=120.5)
    assert result.type == "tool.result"
    assert result.content == "1 row"
    assert result.duration_ms == 120.5
    assert result.status == "success"


def test_artifact_event() -> None:
    art = AuipArtifactEvent(
        artifact_type="mermaid",
        title="Architecture",
        content="graph TD\nA-->B",
        language="mermaid",
    )
    assert art.type == "artifact.emit"
    assert art.artifact_type == "mermaid"
    assert art.title == "Architecture"


def test_hitl_request_and_response() -> None:
    req = AuipHitlRequestEvent(question="Do you approve dropping the table?", clarification_type="confirmation")
    assert req.type == "hitl.request"
    assert req.question == "Do you approve dropping the table?"
    assert req.clarification_type == "confirmation"

    resp = AuipHitlResponse(request_id=req.request_id, response="yes", approved=True)
    assert resp.type == "hitl.response"
    assert resp.request_id == req.request_id
    assert resp.approved is True


def test_inbound_messages() -> None:
    user_msg = AuipUserMessage(text="How are you?", thread_id="t1")
    assert user_msg.type == "user.message"
    assert user_msg.text == "How are you?"

    tool_dec = AuipToolDecision(call_id="c1", decision="approve")
    assert tool_dec.type == "tool.decision"
    assert tool_dec.decision == "approve"

    ctrl = AuipInterruptControl(signal="cancel", reason="user requested")
    assert ctrl.type == "control.interrupt"
    assert ctrl.signal == "cancel"


def test_telemetry_usage_event() -> None:
    usage = AuipUsageTelemetryEvent(
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.002,
        model_name="gpt-4o",
        duration_ms=450.0,
    )
    assert usage.type == "telemetry.usage"
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.cost_usd == 0.002
    assert usage.model_name == "gpt-4o"
