"""Multi-turn simulation evaluation tests.

Tests in this module use ``openevals.simulators.run_multiturn_simulation`` to
simulate conversations between a callable "app" (wrapping a LangchainAgent) and
a simulated user (either ``fixed_responses`` or LLM-powered).

Test tiers:
- ``TestDeterministicSimulation`` — Uses ``fixed_responses`` for the simulated
  user and a mocked agent graph, so no LLM is needed at all.  Runs in CI.
- ``TestRealModelSimulation`` — Uses a real LLM for the agent, gated by
  ``--include-real-models``.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Deterministic simulation (no real LLM needed)
# ---------------------------------------------------------------------------


class TestDeterministicSimulation:
    """Multi-turn simulation with fixed user turns and a scripted agent."""

    def _build_scripted_app(self, responses: list[str]):
        """Return a sync app callable that returns scripted agent responses.

        The simulator calls ``app(message, thread_id=..., **kwargs)`` where
        ``message`` is the latest user ``ChatCompletionMessage`` dict.  The app
        must return a single message dict (role + content).

        Args:
            responses: List of agent responses to return in order.  If the
                conversation has more turns than responses, the last response
                is repeated.
        """
        turn_counter = [0]

        def app(message: dict[str, Any], *, thread_id: str | None = None, **kwargs) -> dict[str, Any]:
            idx = min(turn_counter[0], len(responses) - 1)
            turn_counter[0] += 1
            return {"role": "assistant", "content": responses[idx]}

        return app

    @pytest.mark.evals
    def test_single_turn_simulation_produces_trajectory(self) -> None:
        """Single-turn simulation with fixed response produces a valid trajectory."""
        from openevals.simulators import run_multiturn_simulation

        app = self._build_scripted_app(["The answer is 42."])
        result = run_multiturn_simulation(
            app=app,
            user=["What is the answer to life, the universe, and everything?"],
            max_turns=1,
        )

        assert result is not None
        trajectory = result.get("trajectory") or result.get("messages") or []
        # Should have at least the user turn and the assistant response
        roles = [m.get("role") if isinstance(m, dict) else getattr(m, "role", None) for m in trajectory]
        assert "user" in roles, f"Expected user message in trajectory. Got roles: {roles}"
        assert "assistant" in roles, f"Expected assistant message in trajectory. Got roles: {roles}"

    @pytest.mark.evals
    def test_multi_turn_simulation_produces_correct_turn_count(self) -> None:
        """Three-turn simulation produces at least 3 exchange pairs."""
        from openevals.simulators import run_multiturn_simulation

        app = self._build_scripted_app(["Response 1", "Response 2", "Response 3"])
        result = run_multiturn_simulation(
            app=app,
            user=["Turn 1 question", "Turn 2 followup", "Turn 3 question"],
            max_turns=3,
        )

        trajectory = result.get("trajectory") or result.get("messages") or []
        # 3 user turns + 3 assistant turns = at least 6 messages
        assert len(trajectory) >= 6, f"Expected >= 6 messages for 3 turns, got {len(trajectory)}: {trajectory}"

    @pytest.mark.evals
    def test_simulation_with_trajectory_evaluator(self) -> None:
        """Trajectory evaluator runs against finished multi-turn trajectory."""
        from openevals.simulators import run_multiturn_simulation

        # The evaluator checks that the final trajectory contains the expected assistant content
        def check_response_present(*, outputs: Any, reference_outputs: Any = None, **kwargs) -> dict:
            # trajectory_evaluators receive outputs as a list of message dicts
            trajectory = outputs if isinstance(outputs, list) else outputs.get("messages", [])
            assistant_contents = [
                m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
                for m in trajectory
                if (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) == "assistant"
            ]
            found = any("42" in str(c) for c in assistant_contents)
            return {"key": "has_42_in_response", "score": found, "comment": None, "metadata": None}

        app = self._build_scripted_app(["The answer is 42."])
        result = run_multiturn_simulation(
            app=app,
            user=["What is the answer?"],
            max_turns=1,
            trajectory_evaluators=[check_response_present],
        )

        eval_results = result.get("evaluator_results", [])
        assert len(eval_results) >= 1, "Expected at least one evaluator result"
        scores = [r.get("score") if isinstance(r, dict) else getattr(r, "score", None) for r in eval_results]
        assert any(scores), f"Expected at least one True score in eval results: {eval_results}"

    @pytest.mark.evals
    def test_stopping_condition_ends_simulation_early(self) -> None:
        """Simulation stops early when stopping_condition returns True."""
        from openevals.simulators import run_multiturn_simulation

        app = self._build_scripted_app(["I need more info.", "Done!", "Extra turn"])
        call_count = [0]

        def stop_after_two(trajectory: Any, *, turn_counter: int = 0, **kwargs) -> bool:
            call_count[0] = turn_counter
            return turn_counter >= 2

        result = run_multiturn_simulation(
            app=app,
            user=["Turn 1", "Turn 2", "Turn 3", "Turn 4"],
            max_turns=4,
            stopping_condition=stop_after_two,
        )

        trajectory = result.get("trajectory") or result.get("messages") or []
        # Should have stopped after 2 turns (4 messages max)
        assert len(trajectory) <= 4 + 2, f"Expected simulation to stop early but got {len(trajectory)} messages"


# ---------------------------------------------------------------------------
# Agent integration — mocked graph (deterministic, no real LLM)
# ---------------------------------------------------------------------------


@pytest.mark.evals
@pytest.mark.asyncio
async def test_agent_app_wrapper_with_mocked_graph() -> None:
    """LangchainAgent wrapped as a simulation 'app' callable integrates correctly.

    Mocks the compiled graph to return a scripted response, then runs a
    single-turn simulation to verify the integration between the openevals
    simulator interface and LangchainAgent.

    The agent is wrapped synchronously using ``asyncio.run_coroutine_threadsafe``
    so the synchronous simulator can call through to the async agent.
    """
    import concurrent.futures

    from langchain_core.messages import AIMessage, HumanMessage
    from openevals.simulators import run_multiturn_simulation

    from genai_tk.agents.langchain.langchain_agent import LangchainAgent

    # Build a mock graph that returns a scripted response
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "messages": [
            HumanMessage(content="What is 2 + 2?"),
            AIMessage(content="2 + 2 equals 4."),
        ]
    }

    agent = LangchainAgent("eval-test", llm="parrot_local@fake")
    agent._agent = mock_graph

    # Wrap the async agent.arun() as a synchronous callable for the sync simulator.
    # We run the coroutine in a separate thread's event loop to avoid nested loop issues.
    def agent_app(message: dict[str, Any], *, thread_id: str | None = None, **kwargs) -> dict[str, Any]:
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")

        def _run():
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(agent.arun(content))
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            response = pool.submit(_run).result(timeout=30)

        return {"role": "assistant", "content": response}

    result = run_multiturn_simulation(
        app=agent_app,
        user=["What is 2 + 2?"],
        max_turns=1,
    )

    trajectory = result.get("trajectory") or result.get("messages") or []
    assert any((m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) == "assistant" for m in trajectory), (
        f"Expected assistant message in trajectory: {trajectory}"
    )


# ---------------------------------------------------------------------------
# Real-model simulation (opt-in, --include-real-models)
# ---------------------------------------------------------------------------


@pytest.mark.evals
@pytest.mark.real_models
def test_real_agent_multiturn_calculator_conversation(judge_llm) -> None:
    """Multi-turn simulation verifying the agent produces correct arithmetic answers.

    Each turn is self-contained so the test works with stateless agents.
    The evaluator checks that the final trajectory contains at least one
    assistant message with a correct numeric answer.

    Requires ``--include-real-models``.
    """
    import concurrent.futures

    from openevals.simulators import run_multiturn_simulation

    from genai_tk.agents.langchain.langchain_agent import LangchainAgent
    from tests.eval_tests.conftest import calculator, echo

    agent = LangchainAgent("eval-test", llm="fast_model", agent_type="react", tools=[calculator, echo])

    def agent_app(message: dict[str, Any], *, thread_id: str | None = None, **kwargs) -> dict[str, Any]:
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")

        def _run():
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(agent.arun(content))
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            response = pool.submit(_run).result(timeout=60)

        return {"role": "assistant", "content": response}

    def check_correct_answers(*, outputs: Any, reference_outputs: Any = None, **kwargs) -> dict:
        """Check that the trajectory contains correct numeric answers (96 and 192)."""
        trajectory = outputs if isinstance(outputs, list) else outputs.get("messages", [])
        assistant_texts = " ".join(
            str(m.get("content", "") if isinstance(m, dict) else getattr(m, "content", ""))
            for m in trajectory
            if (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) == "assistant"
        )
        found_96 = "96" in assistant_texts
        found_192 = "192" in assistant_texts
        score = found_96 and found_192
        return {
            "key": "correct_arithmetic_answers",
            "score": score,
            "comment": f"96 found={found_96}, 192 found={found_192}",
            "metadata": None,
        }

    result = run_multiturn_simulation(
        app=agent_app,
        user=[
            "What is 12 * 8?",
            "What is 96 * 2?",
        ],
        max_turns=2,
        trajectory_evaluators=[check_correct_answers],
    )

    eval_results = result.get("evaluator_results", [])
    for er in eval_results:
        score = er.get("score") if isinstance(er, dict) else getattr(er, "score", None)
        key = er.get("key") if isinstance(er, dict) else getattr(er, "key", "")
        comment = er.get("comment") if isinstance(er, dict) else getattr(er, "comment", "")
        assert score is True or score == 1, (
            f"Evaluator '{key}' failed ({comment}). Full result: {er}\n"
            f"Trajectory: {result.get('trajectory', result.get('messages', []))}"
        )
