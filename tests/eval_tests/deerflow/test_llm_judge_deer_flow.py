"""LLM-as-judge evaluation tests for Deer-flow agents.

All tests call two real LLMs:
  - The **deer-flow agent** (to generate the answer under evaluation)
  - The **judge** (``fast_model`` via openevals, to score the answer)

Both require ``--include-real-models`` and ``DEER_FLOW_PATH`` (enforced by the
session fixtures in ``conftest.py``).  Tests are also gated behind
``@pytest.mark.real_models`` and ``@pytest.mark.deerflow`` for fine-grained
filtering.

Design notes
------------
* Deer-flow uses ``flash`` mode for speed; no thinking / planning overhead.
* Queries are chosen so that a capable LLM can answer correctly without web
  search, avoiding flakiness from external network dependencies.
* All assertions use ``bool(score)`` rather than ``score == 1`` because
  evaluators can return ``True``/``1`` depending on the prompt and model.
"""

from __future__ import annotations

import asyncio

import pytest

AGENT_TIMEOUT = 180  # seconds
JUDGE_TIMEOUT = 120  # seconds
TOTAL_TIMEOUT = AGENT_TIMEOUT + JUDGE_TIMEOUT  # per test ceiling


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _assert_pass(result: dict, label: str) -> None:
    score = result.get("score")
    reasoning = result.get("comment") or "(no reasoning returned)"
    assert score, (
        f"[{label}] Expected PASS (score=True/1) but got score={score!r}.\n"
        f"Judge reasoning: {reasoning}\n"
        f"Full result: {result}"
    )


def _assert_fail(result: dict, label: str) -> None:
    score = result.get("score")
    reasoning = result.get("comment") or "(no reasoning returned)"
    assert not score, (
        f"[{label}] Expected FAIL (score=False/0) but got score={score!r}.\n"
        f"Judge reasoning: {reasoning}\n"
        f"Full result: {result}"
    )


# ---------------------------------------------------------------------------
# Correctness — does the agent give factually correct answers?
# ---------------------------------------------------------------------------


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.deerflow
@pytest.mark.timeout(TOTAL_TIMEOUT)
def test_factual_geography_answer_is_correct(deerflow_client, judge_llm) -> None:
    """Deer-flow answers a simple geography question correctly."""
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import CORRECTNESS_PROMPT

    answer = asyncio.run(deerflow_client.arun("What is the capital of France?"))
    assert answer, "Agent returned an empty response"

    evaluator = create_llm_as_judge(prompt=CORRECTNESS_PROMPT, judge=judge_llm)
    result = evaluator(
        inputs="What is the capital of France?",
        outputs=answer,
        reference_outputs="Paris",
    )
    _assert_pass(result, "capital of France")


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.deerflow
@pytest.mark.timeout(TOTAL_TIMEOUT)
def test_math_answer_is_correct(deerflow_client, judge_llm) -> None:
    """Deer-flow computes a simple arithmetic result correctly."""
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import CORRECTNESS_PROMPT

    answer = asyncio.run(deerflow_client.arun("What is 17 multiplied by 3?"))
    assert answer, "Agent returned an empty response"

    evaluator = create_llm_as_judge(prompt=CORRECTNESS_PROMPT, judge=judge_llm)
    result = evaluator(
        inputs="What is 17 multiplied by 3?",
        outputs=answer,
        reference_outputs="51",
    )
    _assert_pass(result, "17 * 3 = 51")


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.deerflow
@pytest.mark.timeout(TOTAL_TIMEOUT)
def test_wrong_answer_scores_low(deerflow_client, judge_llm) -> None:
    """A deliberately wrong answer is scored as a fail by the correctness judge.

    This validates that the judge is actually checking content, not rubber-
    stamping every response.  The agent is not involved — we feed a static
    wrong answer directly to the evaluator.
    """
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import CORRECTNESS_PROMPT

    evaluator = create_llm_as_judge(prompt=CORRECTNESS_PROMPT, judge=judge_llm)
    result = evaluator(
        inputs="What is the capital of France?",
        outputs="The capital of France is Berlin.",
        reference_outputs="Paris",
    )
    _assert_fail(result, "wrong capital — should fail")


# ---------------------------------------------------------------------------
# Conciseness — is the answer brief and on-point?
# ---------------------------------------------------------------------------


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.deerflow
@pytest.mark.timeout(TOTAL_TIMEOUT)
def test_single_sentence_answer_is_concise(deerflow_client, judge_llm) -> None:
    """A single-sentence factual answer scores well for conciseness."""
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import CONCISENESS_PROMPT

    evaluator = create_llm_as_judge(prompt=CONCISENESS_PROMPT, judge=judge_llm)
    result = evaluator(
        inputs="What is the capital of France?",
        outputs="The capital of France is Paris.",
    )
    _assert_pass(result, "concise factual answer")


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.deerflow
@pytest.mark.timeout(TOTAL_TIMEOUT)
def test_verbose_padding_scores_low_for_conciseness(judge_llm) -> None:
    """A padded, repetitive answer scores poorly for conciseness.

    Uses a static response — no agent involved — to validate the evaluator.
    """
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import CONCISENESS_PROMPT

    bloated = (
        "That is a great question! The capital of France is a fascinating topic. "
        "Paris, which has been the capital for many centuries, is indeed the capital "
        "of France. So to answer your question, the capital is Paris. "
        "I hope that answers your question about the capital of France."
    )
    evaluator = create_llm_as_judge(prompt=CONCISENESS_PROMPT, judge=judge_llm)
    result = evaluator(
        inputs="What is the capital of France?",
        outputs=bloated,
    )
    _assert_fail(result, "verbose padded answer — should fail conciseness")


# ---------------------------------------------------------------------------
# End-to-end: deer-flow answers + judge scores in one test
# ---------------------------------------------------------------------------


@pytest.mark.evals
@pytest.mark.real_models
@pytest.mark.deerflow
@pytest.mark.timeout(TOTAL_TIMEOUT)
def test_deer_flow_answer_is_relevant_to_question(deerflow_client, judge_llm) -> None:
    """The agent's response is relevant to the question asked.

    Uses ``RELEVANCE_PROMPT`` — a lighter-weight check than correctness that
    only asks whether the response addresses the question.
    """
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import ANSWER_RELEVANCE_PROMPT

    question = "What are the main differences between Python lists and tuples?"
    answer = asyncio.run(deerflow_client.arun(question))
    assert answer, "Agent returned an empty response"

    evaluator = create_llm_as_judge(prompt=ANSWER_RELEVANCE_PROMPT, judge=judge_llm)
    result = evaluator(inputs=question, outputs=answer)
    _assert_pass(result, "lists vs tuples — relevance check")
