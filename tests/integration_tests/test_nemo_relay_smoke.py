"""Phase-0 spike: NeMo Relay ATOF capture through the deep-agent flow.

Verifies that a local-backend Deep Agents run, built via the instrumented
``_create_deep_agent``, emits canonical ATOF scope/llm/tool events to the local
JSONL subscriber — with no API key (a scripted tool-calling fake model is used).

Skips automatically when ``nemo_relay`` is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("nemo_relay")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import Field, PrivateAttr

from genai_tk.agents.langchain.config import AgentProfileConfig
from genai_tk.agents.langchain.factory import _create_deep_agent
from genai_tk.utils.nemo_relay_setup import (
    flush_nemo_relay_async,
    get_relay_callback_handler,
    reset_nemo_relay,
    setup_nemo_relay,
)
from genai_tk.utils.tracing import reset_monitoring


class ScriptedToolChatModel(BaseChatModel):
    """A minimal BaseChatModel that replays a scripted message queue.

    Supports ``bind_tools`` (returns self) so it can drive a Deep Agents graph;
    the scripted AIMessage already carries the tool call to emit.
    """

    model_name: str = "scripted-fake"
    responses: list[BaseMessage] = Field(default_factory=list)
    _idx: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedToolChatModel":  # noqa: ARG002
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> ChatResult:
        if not self.responses:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])
        idx = min(self._idx, len(self.responses) - 1)
        msg = self.responses[idx]
        self._idx += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


@tool
def echo(message: str) -> str:
    """Echo the message back unchanged."""
    return f"echo:{message}"


@pytest.mark.asyncio
async def test_deep_agent_emits_atof_scope_llm_tool(tmp_path: Path) -> None:
    """A deep-agent run writes ATOF scope/llm/tool events to the local JSONL file."""
    reset_monitoring()
    reset_nemo_relay()
    atof_path = tmp_path / "events.jsonl"
    assert setup_nemo_relay(atof_path=atof_path), "nemo_relay subscriber did not activate"

    model = ScriptedToolChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "echo", "args": {"message": "hello"}, "id": "tc1", "type": "tool_call"}],
            ),
            AIMessage(content="The echo result is echo:hello"),
        ],
    )
    profile = AgentProfileConfig(
        name="relay-smoke",
        type="deep",
        llm="fake",
        tools=[],
        mcp_servers=[],
        skill_directories=[],
    )

    agent = await _create_deep_agent(
        model,
        [echo],
        MemorySaver(),
        profile,
        middlewares=[],
        backend=None,
    )

    handler = get_relay_callback_handler()
    assert handler is not None, "Relay callback handler unavailable"

    await agent.ainvoke(
        {"messages": "please echo 'hello' using the echo tool"},
        config={"configurable": {"thread_id": "1"}, "callbacks": [handler]},
    )
    await flush_nemo_relay_async()

    # Parse the ATOF JSONL stream written by the manual subscriber.
    lines = [ln for ln in atof_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines, f"no ATOF events written to {atof_path}"
    events = [json.loads(ln) for ln in lines]
    kinds = {ev.get("kind") for ev in events}
    assert "scope" in kinds, f"expected scope events, got kinds={kinds}"

    scope_cats = {
        ev.get("category") for ev in events if ev.get("kind") == "scope" and ev.get("scope_category") == "start"
    }
    assert "llm" in scope_cats, f"expected an LLM scope, got categories={scope_cats}"
    assert "tool" in scope_cats, f"expected a tool scope, got categories={scope_cats}"
    # Agent scope comes from the callback mapping the LangGraph run hierarchy.
    assert "agent" in scope_cats, f"expected an agent scope, got categories={scope_cats}"

    reset_nemo_relay()
    reset_monitoring()
