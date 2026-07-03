"""Shared Pydantic event model for all agent harnesses (LangChain, DeerFlow).

Every harness adapter translates its own native events (LangGraph
``astream_events``, DeerFlow ``StreamEvent``) into these canonical types so the
CLI and Streamlit UI only need to understand one event vocabulary.

Event kinds:

- ``token``         — incremental or complete assistant text
- ``node``          — a graph node/phase became active (planner, researcher, …)
- ``tool_call``     — the model is calling a tool
- ``tool_result``   — a tool returned a result
- ``artifact``      — a renderable artifact (code, file, chart, mermaid, …)
- ``clarification`` — the agent paused to ask the user a question (HITL)
- ``usage``         — token usage / cost accounting for a turn
- ``error``         — the run produced an error
- ``end``           — the run has completed
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HarnessEvent(BaseModel):
    """Base class for all harness-normalized streaming events."""

    kind: str

    model_config = ConfigDict(frozen=True)


class TokenEvent(HarnessEvent):
    """Incremental or complete assistant response text."""

    kind: Literal["token"] = "token"
    text: str = ""


class NodeEvent(HarnessEvent):
    """A graph node/phase became active (e.g. planner, researcher, coder, reporter)."""

    kind: Literal["node"] = "node"
    node: str = ""
    state: dict[str, Any] = Field(default_factory=dict)


class ToolCallEvent(HarnessEvent):
    """The model is calling a tool."""

    kind: Literal["tool_call"] = "tool_call"
    tool_name: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    call_id: str = ""


class ToolResultEvent(HarnessEvent):
    """A tool returned a result."""

    kind: Literal["tool_result"] = "tool_result"
    tool_name: str = ""
    content: str = ""
    call_id: str = ""


class ArtifactEvent(HarnessEvent):
    """A renderable artifact produced by a tool or the agent (code, file, chart, …)."""

    kind: Literal["artifact"] = "artifact"
    type: str = "text"
    title: str = ""
    content: str = ""
    language: str = ""


class ClarificationEvent(HarnessEvent):
    """The agent paused and is asking the user a clarifying question (human-in-the-loop).

    The caller should display ``question``, collect a reply, and send it as
    the next message on the same ``thread_id``.
    """

    kind: Literal["clarification"] = "clarification"
    question: str = ""
    clarification_type: str = "missing_info"
    context: str = ""


class UsageEvent(HarnessEvent):
    """Token usage for the current turn, when the harness reports it."""

    kind: Literal["usage"] = "usage"
    input_tokens: int = 0
    output_tokens: int = 0


class ErrorEvent(HarnessEvent):
    """The run produced an error."""

    kind: Literal["error"] = "error"
    message: str = ""


class EndEvent(HarnessEvent):
    """The run has completed; no more events will follow for this turn."""

    kind: Literal["end"] = "end"


StreamEvent = (
    TokenEvent
    | NodeEvent
    | ToolCallEvent
    | ToolResultEvent
    | ArtifactEvent
    | ClarificationEvent
    | UsageEvent
    | ErrorEvent
    | EndEvent
)


class HarnessThread(BaseModel):
    """Metadata about a persisted conversation thread."""

    thread_id: str
    title: str = ""
    updated_at: str = ""


class HarnessModel(BaseModel):
    """A model available to the harness."""

    name: str
    provider: str = ""


class HarnessSkill(BaseModel):
    """A discoverable skill (SKILL.md) available to the harness."""

    name: str
    enabled: bool = True
