"""Agent-User Interaction Protocol (AUIP) v1.0.

Defines the universal, bidirectional contract between autonomous agent runtimes
(LangChain, DeepAgents, DeerFlow) and multimodal frontends (Console CLI,
Terminal TUI, Streamlit Webapp, Remote Web APIs).
"""

from __future__ import annotations

from genai_tk.protocols.auip.ag_ui_bridge import harness_events_to_ag_ui
from genai_tk.protocols.auip.config import AuipConfig, AuipServerSettings, auip_config
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
from genai_tk.protocols.auip.server import create_app, serve

__all__ = [
    "AuipArtifactEvent",
    "AuipConfig",
    "AuipEndEvent",
    "AuipErrorEvent",
    "AuipEvent",
    "AuipHitlRequestEvent",
    "AuipHitlResponse",
    "AuipInboundMessage",
    "AuipInterruptControl",
    "AuipOutboundEvent",
    "AuipPhaseEvent",
    "AuipServerSettings",
    "AuipThinkingEvent",
    "AuipTokenEvent",
    "AuipToolCallEvent",
    "AuipToolDecision",
    "AuipToolResultEvent",
    "AuipUsageTelemetryEvent",
    "AuipUserMessage",
    "auip_config",
    "create_app",
    "harness_events_to_ag_ui",
    "serve",
]
