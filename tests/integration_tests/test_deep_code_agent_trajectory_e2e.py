"""End-to-end test for a deep code agent with programmatic trajectory and outcome check.

Verifies:
1. Deep agent executes a code-execution task (using a python execution tool).
2. NeMo Relay captures the execution into the local ATOF trajectory store.
3. Trajectory structure is programmatically validated (agent, llm, and tool scopes).
4. Tool invocation and arguments/results are accurately captured.
5. Deterministic tool sequence matching (match_tools) passes.
6. The agent's outcome is evaluated using a System One Decision Model.
"""

from __future__ import annotations

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
from genai_tk.core.decision_models.evaluators import evaluate_correctness
from genai_tk.core.decision_models.fake import FakeDecisionModel
from genai_tk.extra.monitoring.nemo_relay_setup import (
    _state,
    flush_nemo_relay_async,
    get_relay_callback_handler,
    reset_nemo_relay,
    setup_nemo_relay,
)
from genai_tk.extra.monitoring.tracing import reset_monitoring
from genai_tk.extra.monitoring.trajectory_store import TrajectoryStore, match_trajectory_tools


class ScriptedCodeChatModel(BaseChatModel):
    """Replays scripted messages simulating a code-act agent."""

    model_name: str = "scripted-code-model"
    responses: list[BaseMessage] = Field(default_factory=list)
    _idx: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted-code-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedCodeChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.responses:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])
        idx = min(self._idx, len(self.responses) - 1)
        msg = self.responses[idx]
        self._idx += 1
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=msg,
                    generation_info={"usage_metadata": {"input_tokens": 120, "output_tokens": 45}},
                )
            ]
        )


@tool
def python_interpreter(code: str) -> str:
    """Execute python code and return stdout."""
    # Safe evaluated arithmetic for testing
    if "fib" in code:
        return "55"
    return "result_ok"


@pytest.mark.asyncio
async def test_deep_code_agent_trajectory_and_outcome_e2e(tmp_path: Path) -> None:
    """End-to-end test: Deep Code Agent execution -> NeMo Relay Trajectory -> Programmatic Checks -> Decision Model Eval."""
    reset_monitoring()
    reset_nemo_relay()

    store_dir = tmp_path / "trajectories"
    assert setup_nemo_relay(store_dir=store_dir, enable_otlp=False), "NeMo Relay failed to setup"

    # Simulate CodeAgent: 1. Tool call to python_interpreter, 2. Final synthesis
    code_snippet = (
        "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\nprint(fib(10))"
    )
    scripted_model = ScriptedCodeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "python_interpreter",
                        "args": {"code": code_snippet},
                        "id": "call_fib_10",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The 10th Fibonacci number is 55."),
        ]
    )

    profile = AgentProfileConfig(
        name="code-agent-e2e",
        type="deep",
        llm="fake",
        tools=[],
        mcp_servers=[],
        skill_directories=[],
        enable_planning=False,
        enable_file_system=False,
    )

    agent = await _create_deep_agent(
        scripted_model,
        [python_interpreter],
        MemorySaver(),
        profile,
        middlewares=[],
        backend=None,
    )

    handler = get_relay_callback_handler()
    assert handler is not None, "NeMo Relay callback handler should be available"

    user_query = "Calculate the 10th Fibonacci number using Python."
    result = await agent.ainvoke(
        {"messages": user_query},
        config={"configurable": {"thread_id": "thread-fib-e2e"}, "callbacks": [handler]},
    )

    # Flush events to local store
    await flush_nemo_relay_async()
    if _state.store is not None:
        _state.store.close()

    # 1. Programmatically inspect the recorded Trajectory in store
    store = TrajectoryStore(root=store_dir)
    runs = store.list_runs()
    assert len(runs) >= 1, "Expected at least one recorded run in the trajectory store"
    run_id = runs[0].run_id

    traj = store.get(run_id)
    assert traj is not None, f"Failed to retrieve trajectory for run {run_id}"

    # 2. Check scopes and tools
    assert traj.tool_calls, "Trajectory should contain captured tool calls"
    called_tool = traj.tool_calls[0]
    assert called_tool.name == "python_interpreter"
    assert "fib" in called_tool.args.get("code", "")
    assert called_tool.result == "55"

    # 3. Deterministic tool sequence matching
    assert match_trajectory_tools([called_tool.name], ["python_interpreter"], mode="strict")
    assert traj.match_tools(["python_interpreter"], mode="superset")
    assert not traj.match_tools(["bash", "read_file"], mode="superset")

    # 4. Check message projection
    msgs = store.messages(run_id)
    assert len(msgs) >= 3, f"Expected at least 3 messages (user, tool, assistant), got {len(msgs)}"
    roles = [m["role"] for m in msgs]
    assert "user" in roles
    assert "tool" in roles
    assert "assistant" in roles

    # 5. Programmatically check agent final outcome
    final_messages = result.get("messages", [])
    assert final_messages, "Agent returned empty messages list"
    final_answer = final_messages[-1].content
    assert "55" in final_answer

    # 6. Evaluate outcome with Decision Model (System One model)
    decision_judge = FakeDecisionModel(fixed_noul=0.98)
    verdict = evaluate_correctness(
        decision_judge,
        question=user_query,
        gold_answer="55",
        agent_answer=final_answer,
    )
    assert verdict.noul >= 0.9, f"Decision model scored correctness below threshold: {verdict.noul}"

    reset_nemo_relay()
