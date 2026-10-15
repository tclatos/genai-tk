"""Agent-User Interaction Protocol (AUIP) v1.0.

Defines the universal, bidirectional contract between autonomous agent runtimes
(LangChain, DeepAgents, DeerFlow) and multimodal frontends (Console CLI,
Terminal TUI, Streamlit Webapp, Remote Web APIs).
"""

from __future__ import annotations

from genai_tk.protocols.auip.models import (
    AuipArtifactEvent,
    AuipEndEvent,
    AuipErrorEvent,
    AuipEvent,
    AuipHitlRequestEvent,
    AuipHitlResponse,
    AuipInboundMessage,
    AuipInterruptControl,
    AuipOutboundEvent,
    AuipPhaseEvent,
    AuipThinkingEvent,
    AuipTokenEvent,
    AuipToolCallEvent,
    AuipToolDecision,
    AuipToolResultEvent,
    AuipUsageTelemetryEvent,
    AuipUserMessage,
)

__all__ = [
    "AuipArtifactEvent",
    "AuipEndEvent",
    "AuipErrorEvent",
    "AuipEvent",
    "AuipHitlRequestEvent",
    "AuipHitlResponse",
    "AuipInboundMessage",
    "AuipInterruptControl",
    "AuipOutboundEvent",
    "AuipPhaseEvent",
    "AuipThinkingEvent",
    "AuipTokenEvent",
    "AuipToolCallEvent",
    "AuipToolDecision",
    "AuipToolResultEvent",
    "AuipUsageTelemetryEvent",
    "AuipUserMessage",
]
