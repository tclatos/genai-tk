"""LLM-as-judge evaluation tests.

All tests in this module call a real language model as a judge and therefore
require ``--include-real-models`` to run.  They are automatically skipped
otherwise via the ``judge_llm_id`` fixture.

Design notes
------------
* All evaluators from ``openevals`` and ``agentevals`` are **synchronous** — no
  ``await`` is needed.  Tests that only call evaluators are plain ``def`` so
  they don't block an asyncio event loop.  Only the end-to-end test that runs
  the real agent remains ``async def``.
* Assertions use ``_assert_pass`` / ``_assert_fail`` helpers that print the
  judge's full reasoning on failure — essential for diagnosing flaky scores.
* Every test is marked ``@pytest.mark.timeout(JUDGE_TIMEOUT)`` (requires
  ``pytest-timeout`` from the ``evals`` dependency group) to prevent a slow
  or hung LLM API call from blocking the whole suite.
* Binary evaluators can return ``True``/``False`` (bool) or ``1``/``0`` (int)
  depending on the prompt and model.  Assertions use ``bool()`` rather than
  ``== 1`` to handle both.
"""

from __future__ import annotations

import pytest

# Wall-clock limit per test.  LLM API calls should complete well within this.
JUDGE_TIMEOUT = 120  # seconds


# ---------------------------------------------------------------------------
# Assertion helpers — print full judge reasoning on failure
# ---------------------------------------------------------------------------


def _assert_pass(result: dict, label: str) -> None:
    """Assert an evaluator result is a pass (truthy score).

    Args:
        result: EvaluatorResult dict returned by the evaluator.
        label: Short description of the scenario for the failure message.
    """
    score = result.get("score")
    reasoning = result.get("comment") or "(no reasoning returned)"
    assert score, (
        f"[{label}] Expected a PASS (score=True/1) but got score={score!r}.\n"
        f"Judge reasoning: {reasoning}\n"
        f"Full result: {result}"
    )


def _assert_fail(result: dict, label: str) -> None:
    """Assert an evaluator result is a fail (falsy score).

    Args:
        result: EvaluatorResult dict returned by the evaluator.
        label: Short description of the scenario for the failure message.
    """
    score = result.get("score")
    reasoning = result.get("comment") or "(no reasoning returned)"
    assert not score, (
        f"[{label}] Expected a FAIL (score=False/0) but got score={score!r}.\n"
        f"Judge reasoning: {reasoning}\n"
        f"Full result: {result}"
    )


# ---------------------------------------------------------------------------
# Output quality: correctness (openevals LLM-as-judge)
# ---------------------------------------------------------------------------


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.timeout(JUDGE_TIMEOUT)
def test_correct_arithmetic_answer_scores_high(judge_llm) -> None:
    """A correct arithmetic answer receives a passing score from the correctness evaluator."""
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import CORRECTNESS_PROMPT

    judge = judge_llm
    evaluator = create_llm_as_judge(prompt=CORRECTNESS_PROMPT, judge=judge)

    result = evaluator(
        inputs="What is 17 * 3?",
        outputs="17 multiplied by 3 equals 51.",
        reference_outputs="51",
    )
    _assert_pass(result, "correctness — 17*3=51 (correct)")


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.timeout(JUDGE_TIMEOUT)
def test_wrong_arithmetic_answer_scores_low(judge_llm) -> None:
    """A factually wrong answer receives a failing score."""
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import CORRECTNESS_PROMPT

    judge = judge_llm
    evaluator = create_llm_as_judge(prompt=CORRECTNESS_PROMPT, judge=judge)

    result = evaluator(
        inputs="What is 17 * 3?",
        outputs="17 multiplied by 3 equals 60.",
        reference_outputs="51",
    )
    _assert_fail(result, "correctness — 17*3=60 (wrong)")


# ---------------------------------------------------------------------------
# Output quality: conciseness
# ---------------------------------------------------------------------------


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.timeout(JUDGE_TIMEOUT)
def test_concise_answer_scores_high(judge_llm) -> None:
    """A concise, direct answer receives a passing conciseness score."""
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import CONCISENESS_PROMPT

    judge = judge_llm
    evaluator = create_llm_as_judge(prompt=CONCISENESS_PROMPT, judge=judge)

    result = evaluator(
        inputs="What is the capital of France?",
        outputs="The capital of France is Paris.",
    )
    _assert_pass(result, "conciseness — direct answer")


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.timeout(JUDGE_TIMEOUT)
def test_verbose_answer_scores_low_for_conciseness(judge_llm) -> None:
    """An excessively padded answer receives a failing conciseness score."""
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import CONCISENESS_PROMPT

    judge = judge_llm
    evaluator = create_llm_as_judge(prompt=CONCISENESS_PROMPT, judge=judge)

    verbose = (
        "That is a wonderful question! Let me provide you with a very comprehensive "
        "and detailed answer that covers all relevant history and geography. "
        "Paris, which is absolutely the capital of France, is a major European city "
        "and a global center for art, fashion, gastronomy, and culture. It has been "
        "the capital since ancient Gallo-Roman times. Furthermore, Paris is situated "
        "on the River Seine in northern France and has a population of over 2 million "
        "people within the city limits. So, to directly and simply answer your very "
        "simple question: the capital of France is Paris. I sincerely hope that helps!"
    )
    result = evaluator(
        inputs="What is the capital of France?",
        outputs=verbose,
    )
    _assert_fail(result, "conciseness — verbose/padded answer")


# ---------------------------------------------------------------------------
# Trajectory accuracy: LLM judge rates tool-call trajectory quality
# ---------------------------------------------------------------------------


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.timeout(JUDGE_TIMEOUT)
def test_trajectory_accuracy_correct_tool_call(judge_llm) -> None:
    """A trajectory that correctly uses the calculator tool scores >= 0.5 accuracy."""
    import json

    from agentevals.trajectory.llm import TRAJECTORY_ACCURACY_PROMPT, create_trajectory_llm_as_judge

    judge = judge_llm
    evaluator = create_trajectory_llm_as_judge(
        prompt=TRAJECTORY_ACCURACY_PROMPT,
        judge=judge,
        continuous=True,  # scores in [0.0, 1.0]
    )

    good_trajectory = [
        {"role": "user", "content": "What is 15 * 4?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "calculator", "arguments": json.dumps({"expression": "15 * 4"})}}],
        },
        {"role": "tool", "content": "60.0"},
        {"role": "assistant", "content": "15 * 4 equals 60."},
    ]
    reference_trajectory = [
        {"role": "user", "content": "What is 15 * 4?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "calculator", "arguments": json.dumps({"expression": "15 * 4"})}}],
        },
        {"role": "tool", "content": "60.0"},
        {"role": "assistant", "content": "60"},
    ]

    result = evaluator(outputs=good_trajectory, reference_outputs=reference_trajectory)
    score = result.get("score")
    reasoning = result.get("comment") or "(no reasoning returned)"
    assert score is not None and score >= 0.5, (
        f"[trajectory accuracy — correct tool call] Expected score >= 0.5 but got {score!r}.\n"
        f"Judge reasoning: {reasoning}\nFull result: {result}"
    )


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.timeout(JUDGE_TIMEOUT)
def test_trajectory_accuracy_wrong_tool_scores_low(judge_llm) -> None:
    """A trajectory using the wrong tool for arithmetic scores < 0.8 accuracy."""
    import json

    from agentevals.trajectory.llm import TRAJECTORY_ACCURACY_PROMPT, create_trajectory_llm_as_judge

    judge = judge_llm
    evaluator = create_trajectory_llm_as_judge(
        prompt=TRAJECTORY_ACCURACY_PROMPT,
        judge=judge,
        continuous=True,
    )

    bad_trajectory = [
        {"role": "user", "content": "What is 15 * 4?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "echo", "arguments": json.dumps({"message": "15 * 4"})}}],
        },
        {"role": "tool", "content": "15 * 4"},
        {"role": "assistant", "content": "The result of 15 * 4 is 15 * 4."},
    ]
    reference_trajectory = [
        {"role": "user", "content": "What is 15 * 4?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "calculator", "arguments": json.dumps({"expression": "15 * 4"})}}],
        },
        {"role": "tool", "content": "60.0"},
        {"role": "assistant", "content": "60"},
    ]

    result = evaluator(outputs=bad_trajectory, reference_outputs=reference_trajectory)
    score = result.get("score")
    reasoning = result.get("comment") or "(no reasoning returned)"
    assert score is not None and score < 0.8, (
        f"[trajectory accuracy — wrong tool] Expected score < 0.8 but got {score!r}.\n"
        f"Judge reasoning: {reasoning}\nFull result: {result}"
    )


# ---------------------------------------------------------------------------
# End-to-end: run a real agent and judge its output
# ---------------------------------------------------------------------------


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.timeout(JUDGE_TIMEOUT)
@pytest.mark.asyncio
async def test_real_agent_calculator_output_correctness(judge_llm) -> None:
    """Run a real React agent with the calculator tool, then judge its final answer.

    Steps:
    1. Run ``LangchainAgent`` (fast_model) with the calculator and echo tools.
    2. Extract the full message trajectory via ``extract_message_trajectory``.
    3. Rate the final text response for correctness using the LLM judge.

    This is the only ``async`` test in this module because it ``await``s the
    agent run.  The judge evaluator itself is still called synchronously after
    the agent finishes.

    Requires ``--include-real-models`` and a valid API key.
    """
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import CORRECTNESS_PROMPT

    from genai_tk.agents.langchain.langchain_agent import LangchainAgent
    from genai_tk.agents.langchain.trajectory import extract_message_trajectory
    from tests.eval_tests.conftest import calculator, echo
    from tests.eval_tests.helpers import get_final_response

    agent = LangchainAgent("eval-test", llm="fast_model", agent_type="react", tools=[calculator, echo])
    trajectory = await extract_message_trajectory(agent, "What is 23 * 7? Use the calculator tool.")
    final_answer = get_final_response(trajectory)

    judge = judge_llm
    evaluator = create_llm_as_judge(prompt=CORRECTNESS_PROMPT, judge=judge)

    result = evaluator(
        inputs="What is 23 * 7?",
        outputs=final_answer,
        reference_outputs="161",
    )
    _assert_pass(
        result,
        f"correctness — agent answer for 23*7=161 (agent said: {final_answer!r})",
    )
