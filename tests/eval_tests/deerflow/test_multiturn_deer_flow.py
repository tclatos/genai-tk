"""Multi-turn simulation evaluation tests for Deer-flow agents.

Uses ``openevals.simulators.run_multiturn_simulation`` with a
``deerflow_agent_app`` wrapper that delegates to ``DeerFlowEvalClient.run_sync``.
A stable ``thread_id`` (set once per test via the fixture) lets deer-flow's
checkpointer maintain conversation state across turns.

All tests are real-model-only (gated by ``--include-real-models`` and
``DEER_FLOW_PATH``).

Design notes
------------
* ``flash`` mode is used for speed — no planning or subagent overhead.
* Conversations are kept to 2 turns to bound test duration.
* Custom trajectory evaluators check for expected content in the accumulated
  assistant responses rather than asserting on exact phrasing.
* Per-test timeout = agent_per_turn × 2 turns + judge overhead.
"""

from __future__ import annotations

from typing import Any

import pytest

TURN_TIMEOUT = 180  # seconds per agent turn
TEST_TIMEOUT = TURN_TIMEOUT * 2 + 60  # 2 turns + judge headroom


# ---------------------------------------------------------------------------
# Helper: extract all assistant text from a simulation result
# ---------------------------------------------------------------------------


def _assistant_texts(result: dict) -> list[str]:
    """Return all non-empty assistant message contents from a simulation result."""
    trajectory = result.get("trajectory") or result.get("messages") or []
    return [
        str(m.get("content", "") if isinstance(m, dict) else getattr(m, "content", ""))
        for m in trajectory
        if (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) == "assistant"
        if (m.get("content", "") if isinstance(m, dict) else getattr(m, "content", ""))
    ]


# ---------------------------------------------------------------------------
# Conversation structure tests
# ---------------------------------------------------------------------------


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.deerflow
@pytest.mark.timeout(TEST_TIMEOUT)
def test_single_turn_simulation_produces_valid_trajectory(deerflow_agent_app) -> None:
    """A one-turn simulation produces a trajectory with user and assistant roles."""
    from openevals.simulators import run_multiturn_simulation

    result = run_multiturn_simulation(
        app=deerflow_agent_app,
        user=["What is the boiling point of water in Celsius?"],
        max_turns=1,
    )

    assert result is not None
    trajectory = result.get("trajectory") or result.get("messages") or []
    roles = [(m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) for m in trajectory]
    assert "user" in roles, f"Expected user message in trajectory. Got roles: {roles}"
    assert "assistant" in roles, f"Expected assistant message in trajectory. Got roles: {roles}"


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.deerflow
@pytest.mark.timeout(TEST_TIMEOUT)
def test_two_turn_simulation_produces_two_agent_responses(deerflow_agent_app) -> None:
    """A two-turn simulation produces two distinct assistant responses."""
    from openevals.simulators import run_multiturn_simulation

    result = run_multiturn_simulation(
        app=deerflow_agent_app,
        user=[
            "What does 'LLM' stand for in the context of AI?",
            "Give me one sentence example of how an LLM is used.",
        ],
        max_turns=2,
    )

    texts = _assistant_texts(result)
    assert len(texts) >= 2, (
        f"Expected at least 2 assistant responses for a 2-turn simulation, got {len(texts)}: {texts}"
    )
    # Both responses should be non-empty
    for i, text in enumerate(texts[:2]):
        assert text.strip(), f"Assistant response #{i + 1} is empty"


# ---------------------------------------------------------------------------
# Context continuity tests
# ---------------------------------------------------------------------------


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.deerflow
@pytest.mark.timeout(TEST_TIMEOUT)
def test_agent_maintains_context_across_turns(deerflow_agent_app) -> None:
    """The agent references information from an earlier turn in a follow-up.

    Turn 1: Define a term.
    Turn 2: Ask a follow-up that requires remembering turn 1.
    Expected: the second response connects to the first (checked by a custom
    evaluator that looks for the term introduced in turn 1).
    """
    from openevals.simulators import run_multiturn_simulation

    introduced_term = "RAG"  # short enough to appear verbatim in a follow-up

    def check_term_referenced(*, outputs: Any, reference_outputs: Any = None, **kwargs) -> dict:
        """Check that the second assistant turn mentions the introduced term."""
        trajectory = outputs if isinstance(outputs, list) else outputs.get("messages", [])
        assistant_msgs = [
            m for m in trajectory if (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) == "assistant"
        ]
        # We are interested in the second assistant message (the follow-up answer)
        if len(assistant_msgs) < 2:
            return {
                "key": "context_continuity",
                "score": False,
                "comment": f"Only {len(assistant_msgs)} assistant message(s) found; expected >= 2",
                "metadata": None,
            }
        second_answer = str(
            assistant_msgs[1].get("content", "")
            if isinstance(assistant_msgs[1], dict)
            else getattr(assistant_msgs[1], "content", "")
        )
        found = introduced_term.upper() in second_answer.upper()
        return {
            "key": "context_continuity",
            "score": found,
            "comment": (f"Second answer mentions '{introduced_term}': {found}. Answer snippet: {second_answer[:200]}"),
            "metadata": None,
        }

    result = run_multiturn_simulation(
        app=deerflow_agent_app,
        user=[
            f"In one sentence, define {introduced_term} (Retrieval-Augmented Generation).",
            f"How does {introduced_term} improve LLM accuracy?",
        ],
        max_turns=2,
        trajectory_evaluators=[check_term_referenced],
    )

    eval_results = result.get("evaluator_results", [])
    scores = [r.get("score") if isinstance(r, dict) else getattr(r, "score", None) for r in eval_results]
    assert any(scores), (
        f"Context continuity evaluator failed.\n"
        f"Evaluator results: {eval_results}\n"
        f"Assistant texts: {_assistant_texts(result)}"
    )


# ---------------------------------------------------------------------------
# Answer verification across turns
# ---------------------------------------------------------------------------


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.deerflow
@pytest.mark.timeout(TEST_TIMEOUT)
def test_sequential_math_questions_answered_correctly(deerflow_agent_app) -> None:
    """Agent answers two independent arithmetic questions correctly across turns.

    Each turn is self-contained — no cross-turn context is required — so this
    tests raw correctness under the simulation harness rather than memory.
    """
    from openevals.simulators import run_multiturn_simulation

    def check_math_answers(*, outputs: Any, reference_outputs: Any = None, **kwargs) -> dict:
        trajectory = outputs if isinstance(outputs, list) else outputs.get("messages", [])
        all_text = " ".join(
            str(m.get("content", "") if isinstance(m, dict) else getattr(m, "content", ""))
            for m in trajectory
            if (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) == "assistant"
        )
        has_35 = "35" in all_text  # 5 * 7 = 35
        has_144 = "144" in all_text  # 12 * 12 = 144
        return {
            "key": "math_answers_correct",
            "score": has_35 and has_144,
            "comment": f"Found '35'={has_35}, Found '144'={has_144}. Text snippet: {all_text[:300]}",
            "metadata": None,
        }

    result = run_multiturn_simulation(
        app=deerflow_agent_app,
        user=["What is 5 times 7?", "What is 12 times 12?"],
        max_turns=2,
        trajectory_evaluators=[check_math_answers],
    )

    eval_results = result.get("evaluator_results", [])
    for er in eval_results:
        score = er.get("score") if isinstance(er, dict) else getattr(er, "score", None)
        comment = er.get("comment") if isinstance(er, dict) else getattr(er, "comment", "")
        assert score, f"Math answer check failed: {comment}"
