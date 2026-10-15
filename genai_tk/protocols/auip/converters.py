"""Bidirectional converters between HarnessEvent and AUIP models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from genai_tk.protocols.auip.models import (
    AuipArtifactEvent,
    AuipEndEvent,
    AuipErrorEvent,
    AuipEvent,
    AuipHitlRequestEvent,
    AuipPhaseEvent,
    AuipThinkingEvent,
    AuipTokenEvent,
    AuipToolCallEvent,
    AuipToolResultEvent,
    AuipUsageTelemetryEvent,
)

if TYPE_CHECKING:
    from genai_tk.agents.harness.events import HarnessEvent


def harness_event_to_auip(
    event: HarnessEvent,
    *,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> AuipEvent:
    """Convert a canonical HarnessEvent into its corresponding AUIP model."""
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

    common: dict[str, Any] = {
        "session_id": session_id,
        "turn_id": turn_id,
    }

    if isinstance(event, TokenEvent):
        return AuipTokenEvent(text=event.text, **common)
    if isinstance(event, ThinkingEvent):
        return AuipThinkingEvent(text=event.text, **common)
    if isinstance(event, NodeEvent):
        return AuipPhaseEvent(phase=event.node, state=event.state, **common)
    if isinstance(event, ToolCallEvent):
        return AuipToolCallEvent(
            tool_name=event.tool_name,
            call_id=event.call_id,
            args=event.args,
            **common,
        )
    if isinstance(event, ToolResultEvent):
        return AuipToolResultEvent(
            tool_name=event.tool_name,
            call_id=event.call_id,
            content=event.content,
            **common,
        )
    if isinstance(event, ArtifactEvent):
        return AuipArtifactEvent(
            artifact_type=event.type,
            title=event.title,
            content=event.content,
            language=event.language,
            **common,
        )
    if isinstance(event, ClarificationEvent):
        return AuipHitlRequestEvent(
            question=event.question,
            clarification_type=event.clarification_type,
            context=event.context,
            **common,
        )
    if isinstance(event, UsageEvent):
        return AuipUsageTelemetryEvent(
            input_tokens=event.input_tokens,
            output_tokens=event.output_tokens,
            **common,
        )
    if isinstance(event, ErrorEvent):
        return AuipErrorEvent(message=event.message, **common)
    if isinstance(event, EndEvent):
        return AuipEndEvent(**common)

    # Fallback generic event
    return AuipEvent(type=f"harness.{event.kind}", **common)


def auip_to_harness_event(event: AuipEvent) -> HarnessEvent:
    """Convert an AUIP event back into the legacy HarnessEvent format."""
    from genai_tk.agents.harness.events import (
        ArtifactEvent,
        ClarificationEvent,
        EndEvent,
        ErrorEvent,
        HarnessEvent,
        NodeEvent,
        ThinkingEvent,
        TokenEvent,
        ToolCallEvent,
        ToolResultEvent,
        UsageEvent,
    )

    if isinstance(event, AuipTokenEvent):
        return TokenEvent(text=event.text)
    if isinstance(event, AuipThinkingEvent):
        return ThinkingEvent(text=event.text)
    if isinstance(event, AuipPhaseEvent):
        return NodeEvent(node=event.phase, state=event.state)
    if isinstance(event, AuipToolCallEvent):
        return ToolCallEvent(tool_name=event.tool_name, call_id=event.call_id, args=event.args)
    if isinstance(event, AuipToolResultEvent):
        return ToolResultEvent(tool_name=event.tool_name, call_id=event.call_id, content=event.content)
    if isinstance(event, AuipArtifactEvent):
        return ArtifactEvent(
            type=event.artifact_type, title=event.title, content=event.content, language=event.language
        )
    if isinstance(event, AuipHitlRequestEvent):
        return ClarificationEvent(
            question=event.question,
            clarification_type=event.clarification_type,
            context=event.context,
        )
    if isinstance(event, AuipUsageTelemetryEvent):
        return UsageEvent(input_tokens=event.input_tokens, output_tokens=event.output_tokens)
    if isinstance(event, AuipErrorEvent):
        return ErrorEvent(message=event.message)
    if isinstance(event, AuipEndEvent):
        return EndEvent()

    return HarnessEvent(kind=event.type)
