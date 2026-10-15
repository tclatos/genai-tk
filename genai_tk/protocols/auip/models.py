"""Agent-User Interaction Protocol (AUIP) Core Event & Message Models.

Provides strongly-typed Pydantic v2 schemas for all outbound streaming events
(agent -> user) and inbound control messages (user -> agent).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _gen_event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:12]}"


# ============================================================================
# Outbound Stream Events (Agent -> User Interface)
# ============================================================================


class AuipEvent(BaseModel):
    """Base class for all outbound AUIP events emitted by an agent runtime."""

    protocol: Literal["auip/1.0"] = "auip/1.0"
    id: str = Field(default_factory=_gen_event_id)
    timestamp: str = Field(default_factory=_now_utc_iso)
    session_id: str | None = None
    turn_id: str | None = None
    type: str

    model_config = ConfigDict(frozen=True)


class AuipTokenEvent(AuipEvent):
    """Incremental or complete assistant response text."""

    type: Literal["stream.token"] = "stream.token"
    text: str = ""


class AuipThinkingEvent(AuipEvent):
    """Incremental or complete model thinking / reasoning trace."""

    type: Literal["stream.thinking"] = "stream.thinking"
    text: str = ""


class AuipPhaseEvent(AuipEvent):
    """A cognitive graph node or subagent phase transition."""

    type: Literal["phase.change"] = "phase.change"
    phase: str = ""
    state: dict[str, Any] = Field(default_factory=dict)


class AuipToolCallEvent(AuipEvent):
    """The agent is invoking an external tool or code executor."""

    type: Literal["tool.call"] = "tool.call"
    tool_name: str = ""
    call_id: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    authorization: Literal["auto", "pending_approval", "approved", "rejected"] = "auto"


class AuipToolResultEvent(AuipEvent):
    """Execution outcome from an external tool or code executor."""

    type: Literal["tool.result"] = "tool.result"
    tool_name: str = ""
    call_id: str = ""
    content: str = ""
    duration_ms: float = 0.0
    status: Literal["success", "error"] = "success"


class AuipArtifactEvent(AuipEvent):
    """A rich renderable artifact produced during execution (code, table, SVG, file)."""

    type: Literal["artifact.emit"] = "artifact.emit"
    artifact_type: str = "text"  # code | file | table | mermaid | chart | search | web
    title: str = ""
    content: str = ""
    language: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuipHitlRequestEvent(AuipEvent):
    """Agent paused to request human input, choice, or permission (Human-in-the-Loop)."""

    type: Literal["hitl.request"] = "hitl.request"
    request_id: str = Field(default_factory=lambda: f"hitl_{uuid.uuid4().hex[:8]}")
    question: str = ""
    clarification_type: str = "missing_info"  # missing_info | confirmation | permission
    options: list[str] = Field(default_factory=list)
    context: str = ""


class AuipUsageTelemetryEvent(AuipEvent):
    """Turn-level token usage, duration, and financial accounting."""

    type: Literal["telemetry.usage"] = "telemetry.usage"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    model_name: str = ""
    duration_ms: float = 0.0


class AuipErrorEvent(AuipEvent):
    """The agent encountered an unhandled error or safety boundary violation."""

    type: Literal["turn.error"] = "turn.error"
    message: str = ""
    code: str | None = None


class AuipEndEvent(AuipEvent):
    """The turn has concluded; no further events will follow for this turn."""

    type: Literal["turn.end"] = "turn.end"
    status: Literal["success", "error", "interrupted"] = "success"
    summary: str = ""


# Discriminated union of all outbound AUIP events
AuipOutboundEvent = Annotated[
    AuipTokenEvent
    | AuipThinkingEvent
    | AuipPhaseEvent
    | AuipToolCallEvent
    | AuipToolResultEvent
    | AuipArtifactEvent
    | AuipHitlRequestEvent
    | AuipUsageTelemetryEvent
    | AuipErrorEvent
    | AuipEndEvent,
    Field(discriminator="type"),
]


# ============================================================================
# Inbound Control Messages (User Interface -> Agent Runtime)
# ============================================================================


class AuipUserMessage(BaseModel):
    """User prompt initiating a turn or continuing a conversation."""

    protocol: Literal["auip/1.0"] = "auip/1.0"
    type: Literal["user.message"] = "user.message"
    text: str = ""
    thread_id: str | None = None
    attachments: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class AuipHitlResponse(BaseModel):
    """User response to a pending hitl.request event."""

    protocol: Literal["auip/1.0"] = "auip/1.0"
    type: Literal["hitl.response"] = "hitl.response"
    request_id: str
    response: str = ""
    approved: bool = True

    model_config = ConfigDict(frozen=True)


class AuipToolDecision(BaseModel):
    """User decision granting, denying, or amending a pending tool call."""

    protocol: Literal["auip/1.0"] = "auip/1.0"
    type: Literal["tool.decision"] = "tool.decision"
    call_id: str
    decision: Literal["approve", "reject", "modify"]
    modified_args: dict[str, Any] | None = None

    model_config = ConfigDict(frozen=True)


class AuipInterruptControl(BaseModel):
    """User control signal interrupting or steering an in-flight agent turn."""

    protocol: Literal["auip/1.0"] = "auip/1.0"
    type: Literal["control.interrupt"] = "control.interrupt"
    signal: Literal["cancel", "pause", "resume", "steer"]
    reason: str = ""

    model_config = ConfigDict(frozen=True)


# Discriminated union of all inbound messages
AuipInboundMessage = Annotated[
    AuipUserMessage | AuipHitlResponse | AuipToolDecision | AuipInterruptControl,
    Field(discriminator="type"),
]
