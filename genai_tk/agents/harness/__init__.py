"""Harness abstraction layer — one event model and one session interface for
LangChain (react | deep | custom, including DeepAgents SDK) and DeerFlow.

This package normalizes the differences between agent runtimes behind a thin
adapter boundary so the CLI and Streamlit UI can consume any harness through
the same :class:`~genai_tk.agents.harness.base.BaseHarness` interface and the
same :mod:`~genai_tk.agents.harness.events` event types.

It intentionally does **not** introduce a second middleware system: harnesses
built on LangChain/LangGraph (which includes DeerFlow) accept plain
``langchain.agents.middleware.AgentMiddleware`` instances directly — see
``docs/middleware-pii-and-routing.md``.

Example:
    ```python
    from genai_tk.agents.harness import create_harness, TokenEvent

    harness = create_harness("research")  # resolves across langchain + deerflow profiles
    async for event in harness.astream("What is RAG?"):
        if isinstance(event, TokenEvent):
            print(event.text, end="", flush=True)
    ```
"""

from genai_tk.agents.harness.base import BaseHarness
from genai_tk.agents.harness.events import (
    ArtifactEvent,
    ClarificationEvent,
    EndEvent,
    ErrorEvent,
    HarnessEvent,
    HarnessModel,
    HarnessSkill,
    HarnessThread,
    NodeEvent,
    StreamEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from genai_tk.agents.harness.profiles import AgentDefaultsConfig, AgentProfile, load_agent_profiles
from genai_tk.agents.harness.registry import HarnessProfileRef, create_harness, list_harness_profiles

__all__ = [
    "BaseHarness",
    "HarnessEvent",
    "StreamEvent",
    "TokenEvent",
    "NodeEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "ArtifactEvent",
    "ClarificationEvent",
    "UsageEvent",
    "ErrorEvent",
    "EndEvent",
    "HarnessThread",
    "HarnessModel",
    "HarnessSkill",
    "create_harness",
    "list_harness_profiles",
    "HarnessProfileRef",
    "AgentProfile",
    "AgentDefaultsConfig",
    "load_agent_profiles",
]
