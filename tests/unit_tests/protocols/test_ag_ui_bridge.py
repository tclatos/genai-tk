"""Unit tests for AG-UI protocol bridge (harness_events_to_ag_ui)."""

from __future__ import annotations

import pytest
from ag_ui.core import (
    CustomEvent,
    EventType,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from ag_ui.encoder import EventEncoder

from genai_tk.agents.harness.events import (
    ArtifactEvent,
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
from genai_tk.protocols.auip.ag_ui_bridge import harness_events_to_ag_ui


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tokens_streaming_lifecycle() -> None:
    async def stream():
        yield TokenEvent(text="Hello ")
        yield TokenEvent(text="world!")
        yield EndEvent()

    events = [
        ev
        async for ev in harness_events_to_ag_ui(
            stream(),
            thread_id="th_1",
            run_id="run_1",
        )
    ]

    types = [ev.type for ev in events]
    assert types == [
        EventType.RUN_STARTED,
        EventType.TEXT_MESSAGE_START,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_CONTENT,
        EventType.TEXT_MESSAGE_END,
        EventType.RUN_FINISHED,
    ]

    assert isinstance(events[0], RunStartedEvent)
    assert events[0].thread_id == "th_1"

    assert isinstance(events[1], TextMessageStartEvent)
    assert events[1].role == "assistant"

    assert isinstance(events[2], TextMessageContentEvent)
    assert events[2].delta == "Hello "

    assert isinstance(events[3], TextMessageContentEvent)
    assert events[3].delta == "world!"

    assert isinstance(events[4], TextMessageEndEvent)

    assert isinstance(events[5], RunFinishedEvent)
    assert events[5].outcome.type == "success"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reasoning_and_tool_calls_lifecycle() -> None:
    async def stream():
        yield ThinkingEvent(text="Pondering...")
        yield NodeEvent(node="researcher")
        yield ToolCallEvent(tool_name="web_search", call_id="c123", args={"query": "AI"})
        yield ToolResultEvent(tool_name="web_search", call_id="c123", content="Results")
        yield TokenEvent(text="Found it.")
        yield EndEvent()

    events = [
        ev
        async for ev in harness_events_to_ag_ui(
            stream(),
            thread_id="th_test",
            run_id="run_test",
        )
    ]

    types = [ev.type for ev in events]
    assert EventType.REASONING_START in types
    assert EventType.REASONING_MESSAGE_CONTENT in types
    assert EventType.REASONING_END in types
    assert EventType.STEP_STARTED in types
    assert EventType.TOOL_CALL_START in types
    assert EventType.TOOL_CALL_ARGS in types
    assert EventType.TOOL_CALL_END in types
    assert EventType.TOOL_CALL_RESULT in types
    assert EventType.TEXT_MESSAGE_START in types
    assert EventType.RUN_FINISHED in types

    encoder = EventEncoder()
    for ev in events:
        encoded = encoder.encode(ev)
        assert encoded.startswith("data: {")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_artifacts_and_hitl_events() -> None:
    async def stream():
        yield ArtifactEvent(type="mermaid", title="Diagram", content="graph TD; A-->B;")
        yield ClarificationEvent(question="Proceed with delete?", clarification_type="confirmation")
        yield UsageEvent(input_tokens=100, output_tokens=50)
        yield EndEvent()

    events = [
        ev
        async for ev in harness_events_to_ag_ui(
            stream(),
            thread_id="th_hitl",
            run_id="run_hitl",
        )
    ]

    custom_events = [ev for ev in events if isinstance(ev, CustomEvent)]
    assert len(custom_events) == 3
    names = [ev.name for ev in custom_events]
    assert "artifact" in names
    assert "clarification" in names
    assert "usage" in names


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_handling() -> None:
    async def stream():
        yield TokenEvent(text="Starting...")
        yield ErrorEvent(message="Something exploded")
        yield EndEvent()

    events = [
        ev
        async for ev in harness_events_to_ag_ui(
            stream(),
            thread_id="th_err",
            run_id="run_err",
        )
    ]

    error_events = [ev for ev in events if isinstance(ev, RunErrorEvent)]
    assert len(error_events) == 1
    assert error_events[0].message == "Something exploded"

    last_event = events[-1]
    assert isinstance(last_event, RunFinishedEvent)
