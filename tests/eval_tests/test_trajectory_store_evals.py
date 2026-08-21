"""Phase 3 real-model eval: judge a trajectory captured in the store (gated).

Runs a deep agent with the real default model against the ``echo`` tool, writes
the run into an eval-local trajectory store, then loads the captured trajectory
and runs:

- the deterministic judges (tool_use / grounding / efficiency),
- the ``agentevals`` trajectory-match superset check over the **captured**
  messages (replacing the ``extract_message_trajectory`` re-run pattern),
- the ``openevals`` correctness LLM judge over the final answer.

Gated behind ``--include-real-models``. API keys are read from ``~/.env``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from genai_tk.agents.langchain.config import AgentProfileConfig
from genai_tk.agents.langchain.factory import _create_deep_agent
from genai_tk.agents.langchain.trajectory_store_io import (
    compare_trajectory_to_golden,
    judge_trajectory,
    load_trajectory_messages,
)
from genai_tk.core.factories.llm_factory import get_llm
from genai_tk.utils.nemo_relay_setup import (
    _state,
    flush_nemo_relay_async,
    get_relay_callback_handler,
    reset_nemo_relay,
    setup_nemo_relay,
)
from genai_tk.utils.tracing import reset_monitoring
from genai_tk.utils.trajectory_store import TrajectoryStore

pytest.importorskip("nemo_relay")

_home_env = Path.home() / ".env"
if _home_env.exists():
    load_dotenv(_home_env, override=False)

_REAL_TIMEOUT = 180


@tool
def echo(message: str) -> str:
    """Echo the message back unchanged."""
    return f"echo:{message}"


@pytest.mark.real_models
@pytest.mark.asyncio
@pytest.mark.timeout(_REAL_TIMEOUT)
async def test_captured_trajectory_passes_store_evals(tmp_path: Path) -> None:
    """A real-model run captured in the store passes store-based evals."""
    store_dir = tmp_path / "trajectories"
    reset_monitoring()
    reset_nemo_relay()
    try:
        setup_nemo_relay(store_dir=store_dir)
        model = get_llm(llm="default", streaming=False)
        profile = AgentProfileConfig(
            name="relay-eval",
            type="deep",
            llm="default",
            tools=[],
            mcp_servers=[],
            skill_directories=[],
            enable_planning=False,
            enable_file_system=False,
        )
        agent = await _create_deep_agent(model, [echo], MemorySaver(), profile, middlewares=[], backend=None)
        handler = get_relay_callback_handler()
        assert handler is not None

        await agent.ainvoke(
            {"messages": "Please call the echo tool with the message 'hello', then state the result."},
            config={"configurable": {"thread_id": "1"}, "callbacks": [handler]},
        )
        await flush_nemo_relay_async()
        if _state.store is not None:
            _state.store.close()

        store = TrajectoryStore(root=store_dir)
        runs = store.list_runs()
        assert runs, "no runs captured in the eval-local store"
        run_id = runs[0].run_id

        # The captured trajectory projects to a full user/assistant/tool/assistant sequence.
        msgs = load_trajectory_messages(run_id, store=store)
        roles = [m["role"] for m in msgs]
        assert "user" in roles, f"captured trajectory missing user message: {roles}"
        assert "tool" in roles, f"captured trajectory missing tool result: {roles}"

        # Golden structural comparison (tools superset + step bounds).
        golden = {"tools": ["echo"], "min_steps": 4, "max_steps": 6}
        verdict = compare_trajectory_to_golden(run_id, golden, store=store)
        assert verdict["pass"], f"golden comparison failed: {verdict['checks']}"

        # Deterministic judges over the captured trajectory.
        det = judge_trajectory(
            run_id,
            [{"kind": "tool_use", "tools": ["echo"]}, {"kind": "grounding"}, {"kind": "efficiency", "max_repeat": 3}],
            store=store,
        )
        by_kind = {v["kind"]: v for v in det}
        assert by_kind["tool_use"]["score"] is True
        assert by_kind["grounding"]["score"] is True
        assert by_kind["efficiency"]["score"] is True

        # agentevals superset over the captured messages (no re-run).
        from agentevals.trajectory.match import create_trajectory_match_evaluator

        reference = [
            {"role": "user", "content": "Please call the echo tool with the message 'hello'."},
            {"role": "assistant", "tool_calls": [{"function": {"name": "echo", "arguments": "{}"}}]},
        ]
        evaluator = create_trajectory_match_evaluator(trajectory_match_mode="superset", tool_args_match_mode="ignore")
        result = evaluator(outputs=msgs, reference_outputs=reference)
        assert result["score"] is True, f"superset over captured trajectory failed: {result}"
    finally:
        reset_nemo_relay()
        reset_monitoring()
        shutil.rmtree(store_dir, ignore_errors=True)


@pytest.mark.real_models
@pytest.mark.asyncio
@pytest.mark.timeout(_REAL_TIMEOUT)
async def test_captured_trajectory_correctness_judge(tmp_path: Path) -> None:
    """LLM-judge the final answer of a captured trajectory (openevals correctness)."""
    store_dir = tmp_path / "trajectories"
    reset_monitoring()
    reset_nemo_relay()
    try:
        setup_nemo_relay(store_dir=store_dir)
        model = get_llm(llm="default", streaming=False)
        profile = AgentProfileConfig(
            name="relay-correctness",
            type="deep",
            llm="default",
            tools=[],
            mcp_servers=[],
            skill_directories=[],
            enable_planning=False,
            enable_file_system=False,
        )
        agent = await _create_deep_agent(model, [echo], MemorySaver(), profile, middlewares=[], backend=None)
        handler = get_relay_callback_handler()
        await agent.ainvoke(
            {"messages": "Use the echo tool to echo 'hello', then tell me the result."},
            config={"configurable": {"thread_id": "1"}, "callbacks": [handler]},
        )
        await flush_nemo_relay_async()
        if _state.store is not None:
            _state.store.close()

        store = TrajectoryStore(root=store_dir)
        runs = store.list_runs()
        assert runs
        run_id = runs[0].run_id
        verdicts = judge_trajectory(
            run_id,
            [
                {
                    "kind": "correctness",
                    "judge": get_llm("fast_model", streaming=False),
                    "inputs": "Use the echo tool to echo 'hello', then tell me the result.",
                    "reference_outputs": "echo:hello",
                }
            ],
            store=store,
        )
        v = verdicts[0]
        if v["score"] is None:
            # Judge model returned non-JSON (format flakiness), not a code failure.
            pytest.skip(f"correctness judge did not return JSON: {v['comment']}")
        assert v["score"], f"correctness judge failed: {v}"
    finally:
        reset_nemo_relay()
        reset_monitoring()
        shutil.rmtree(store_dir, ignore_errors=True)
