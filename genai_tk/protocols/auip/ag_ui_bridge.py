"""Bridge converting BaseHarness streaming events to AG-UI standard protocol events.

Enables full interoperability with ``@assistant-ui/react-ag-ui``, ``@ag-ui/client``,
and other standard AG-UI frontends.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ag_ui.core import (
    BaseEvent,
    CustomEvent,
    EventType,
    ReasoningEndEvent,
    ReasoningMessageContentEvent,
    ReasoningStartEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunFinishedSuccessOutcome,
    RunStartedEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from loguru import logger

if TYPE_CHECKING:
    from genai_tk.agents.harness.events import HarnessEvent


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _error_events(message: str, open_msg_id: str | None) -> list[BaseEvent]:
    """Build events reporting an error, also as a visible assistant message (UIs often hide RUN_ERROR)."""
    logger.error("AG-UI run error: {}", message)
    msg_id = _gen_id("err")
    events: list[BaseEvent] = []
    if open_msg_id is not None:
        events.append(TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=open_msg_id))
    events += [
        TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id=msg_id, role="assistant"),
        TextMessageContentEvent(type=EventType.TEXT_MESSAGE_CONTENT, message_id=msg_id, delta=f"⚠️ Error: {message}"),
        TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=msg_id),
        RunErrorEvent(type=EventType.RUN_ERROR, message=message),
    ]
    return events


async def harness_events_to_ag_ui(
    event_stream: AsyncIterator[HarnessEvent],
    *,
    thread_id: str,
    run_id: str,
) -> AsyncIterator[BaseEvent]:
    """Translate a canonical HarnessEvent stream into official AG-UI protocol events.

    Args:
        event_stream: Async stream of HarnessEvent instances from a BaseHarness.
        thread_id: Conversation thread identifier.
        run_id: Specific execution run identifier.

    Yields:
        Official AG-UI protocol event instances conforming to ag-ui-protocol spec.
    """
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

    yield RunStartedEvent(
        type=EventType.RUN_STARTED,
        thread_id=thread_id,
        run_id=run_id,
    )

    current_text_msg_id: str | None = None
    current_reasoning_id: str | None = None
    active_step: str | None = None
    had_error = False

    try:
        async for event in event_stream:
            # 1. Assistant Text Tokens
            if isinstance(event, TokenEvent):
                if current_reasoning_id is not None:
                    yield ReasoningEndEvent(
                        type=EventType.REASONING_END,
                        message_id=current_reasoning_id,
                    )
                    current_reasoning_id = None

                if current_text_msg_id is None:
                    current_text_msg_id = _gen_id("msg")
                    yield TextMessageStartEvent(
                        type=EventType.TEXT_MESSAGE_START,
                        message_id=current_text_msg_id,
                        role="assistant",
                    )
                yield TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=current_text_msg_id,
                    delta=event.text,
                )

            # 2. Thinking / Reasoning Chunks
            elif isinstance(event, ThinkingEvent):
                if current_text_msg_id is not None:
                    yield TextMessageEndEvent(
                        type=EventType.TEXT_MESSAGE_END,
                        message_id=current_text_msg_id,
                    )
                    current_text_msg_id = None

                if current_reasoning_id is None:
                    current_reasoning_id = _gen_id("reason")
                    yield ReasoningStartEvent(
                        type=EventType.REASONING_START,
                        message_id=current_reasoning_id,
                    )
                yield ReasoningMessageContentEvent(
                    type=EventType.REASONING_MESSAGE_CONTENT,
                    message_id=current_reasoning_id,
                    delta=event.text,
                )

            # 3. Tool Invocations
            elif isinstance(event, ToolCallEvent):
                if current_text_msg_id is not None:
                    yield TextMessageEndEvent(
                        type=EventType.TEXT_MESSAGE_END,
                        message_id=current_text_msg_id,
                    )
                    current_text_msg_id = None
                if current_reasoning_id is not None:
                    yield ReasoningEndEvent(
                        type=EventType.REASONING_END,
                        message_id=current_reasoning_id,
                    )
                    current_reasoning_id = None

                call_id = event.call_id or _gen_id("call")
                yield ToolCallStartEvent(
                    type=EventType.TOOL_CALL_START,
                    tool_call_id=call_id,
                    tool_call_name=event.tool_name,
                )
                args_delta = json.dumps(event.args) if isinstance(event.args, dict) else str(event.args)
                yield ToolCallArgsEvent(
                    type=EventType.TOOL_CALL_ARGS,
                    tool_call_id=call_id,
                    delta=args_delta,
                )
                yield ToolCallEndEvent(
                    type=EventType.TOOL_CALL_END,
                    tool_call_id=call_id,
                )

            # 4. Tool Execution Results
            elif isinstance(event, ToolResultEvent):
                call_id = event.call_id or _gen_id("call")
                res_id = _gen_id("res")
                yield ToolCallResultEvent(
                    type=EventType.TOOL_CALL_RESULT,
                    tool_call_id=call_id,
                    message_id=res_id,
                    content=event.content,
                    role="tool",
                )

            # 5. Cognitive Steps / Node Transitions
            elif isinstance(event, NodeEvent):
                if active_step is not None:
                    yield StepFinishedEvent(
                        type=EventType.STEP_FINISHED,
                        step_name=active_step,
                    )
                active_step = event.node
                yield StepStartedEvent(
                    type=EventType.STEP_STARTED,
                    step_name=active_step,
                )

            # 6. Artifacts (Mermaid, code, tables, charts)
            elif isinstance(event, ArtifactEvent):
                yield CustomEvent(
                    type=EventType.CUSTOM,
                    name="artifact",
                    value=event.model_dump(),
                )

            # 7. Clarifications / Human-In-The-Loop
            elif isinstance(event, ClarificationEvent):
                yield CustomEvent(
                    type=EventType.CUSTOM,
                    name="clarification",
                    value=event.model_dump(),
                )

            # 8. Token Usage & Telemetry
            elif isinstance(event, UsageEvent):
                yield CustomEvent(
                    type=EventType.CUSTOM,
                    name="usage",
                    value=event.model_dump(),
                )

            # 9. Errors
            elif isinstance(event, ErrorEvent):
                had_error = True
                for err_event in _error_events(event.message, current_text_msg_id):
                    yield err_event
                current_text_msg_id = None

            # 10. Turn End
            elif isinstance(event, EndEvent):
                break

    except Exception as exc:
        had_error = True
        for err_event in _error_events(str(exc), current_text_msg_id):
            yield err_event
        current_text_msg_id = None

    finally:
        # Close any lingering open stream messages
        if current_reasoning_id is not None:
            yield ReasoningEndEvent(
                type=EventType.REASONING_END,
                message_id=current_reasoning_id,
            )
        if current_text_msg_id is not None:
            yield TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END,
                message_id=current_text_msg_id,
            )
        if active_step is not None:
            yield StepFinishedEvent(
                type=EventType.STEP_FINISHED,
                step_name=active_step,
            )

        if had_error:
            yield RunFinishedEvent(
                type=EventType.RUN_FINISHED,
                thread_id=thread_id,
                run_id=run_id,
            )
        else:
            yield RunFinishedEvent(
                type=EventType.RUN_FINISHED,
                thread_id=thread_id,
                run_id=run_id,
                outcome=RunFinishedSuccessOutcome(type="success"),
            )
