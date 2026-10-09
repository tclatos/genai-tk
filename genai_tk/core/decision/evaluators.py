"""Standard evaluator functions built on System One Decision Models.

Provides typed evaluators replacing legacy prompt-based LLM judges (such as openevals)
with calibrated binary judgments (Noul), categorical choices (Choice), and ordinal
rubrics (Score).
"""

from __future__ import annotations

from typing import Any

from genai_tk.core.decision.base import BaseDecisionModel
from genai_tk.core.decision.types import (
    Choice,
    ChoiceAnswer,
    ClassifierRequest,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)


def evaluate_correctness(
    decision_model: BaseDecisionModel,
    *,
    question: str,
    gold_answer: str,
    agent_answer: str,
    context: str | None = None,
) -> NoulAnswer:
    """Evaluate factual correctness of an agent's answer against a reference answer.

    Returns a :class:`NoulAnswer` with a calibrated probability in [0.0, 1.0].
    """
    state: dict[str, Any] = {
        "question": question,
        "gold_reference_answer": gold_answer,
        "agent_answer": agent_answer,
    }
    if context:
        state["context"] = context

    request = ClassifierRequest(
        state=state,
        questions={
            "correctness": Noul(
                instructions="Based on the question and reference gold answer, is the agent answer factually correct?"
            )
        },
    )
    response = decision_model.invoke(request)
    ans = response.answers["correctness"]
    if isinstance(ans, NoulAnswer):
        return ans
    if isinstance(ans, dict) and "noul" in ans:
        return NoulAnswer.model_validate(ans)
    raise TypeError(f"Expected NoulAnswer, got {type(ans).__name__}")


def evaluate_conciseness(
    decision_model: BaseDecisionModel,
    *,
    question: str,
    agent_answer: str,
) -> ScoreAnswer:
    """Rate answer conciseness on a 3-point ordinal scale (0=padded, 1=acceptable, 2=concise).

    Returns a :class:`ScoreAnswer`.
    """
    request = ClassifierRequest(
        state={"question": question, "agent_answer": agent_answer},
        questions={
            "conciseness": Score(
                instructions="Rate the conciseness and lack of unnecessary padding in the agent answer.",
                criteria=[
                    "Verbose, conversational padding, or repetitive fluff",
                    "Acceptable length with slight wordiness",
                    "Direct, concise, and to the point without filler",
                ],
            )
        },
    )
    response = decision_model.invoke(request)
    ans = response.answers["conciseness"]
    if isinstance(ans, ScoreAnswer):
        return ans
    if isinstance(ans, dict) and "score" in ans:
        return ScoreAnswer.model_validate(ans)
    raise TypeError(f"Expected ScoreAnswer, got {type(ans).__name__}")


def evaluate_groundedness(
    decision_model: BaseDecisionModel,
    *,
    evidence: str | list[str],
    agent_answer: str,
) -> NoulAnswer:
    """Evaluate whether claims in the agent answer are grounded in the provided evidence.

    Returns a :class:`NoulAnswer`.
    """
    evidence_text = "\n".join(evidence) if isinstance(evidence, list) else evidence
    request = ClassifierRequest(
        state={"evidence": evidence_text, "agent_answer": agent_answer},
        questions={
            "groundedness": Noul(
                instructions="Are all factual claims made in the agent answer strictly supported by the evidence?"
            )
        },
    )
    response = decision_model.invoke(request)
    ans = response.answers["groundedness"]
    if isinstance(ans, NoulAnswer):
        return ans
    if isinstance(ans, dict) and "noul" in ans:
        return NoulAnswer.model_validate(ans)
    raise TypeError(f"Expected NoulAnswer, got {type(ans).__name__}")


def evaluate_tool_selection(
    decision_model: BaseDecisionModel,
    *,
    task: str,
    available_tools: list[str],
    selected_tools: list[str],
) -> ChoiceAnswer:
    """Evaluate whether the agent's tool selection was optimal, suboptimal, or incorrect.

    Returns a :class:`ChoiceAnswer`.
    """
    request = ClassifierRequest(
        state={
            "task": task,
            "available_tools": available_tools,
            "selected_tools": selected_tools,
        },
        questions={
            "tool_selection": Choice(
                instructions="Classify the quality and efficiency of the tools selected for this task.",
                criteria={
                    "optimal": "The ideal tool(s) were selected efficiently without redundancy.",
                    "suboptimal": "Valid tools were selected, but redundant calls or inefficient choices occurred.",
                    "incorrect": "The wrong tools were selected, or necessary tools were omitted.",
                },
            )
        },
    )
    response = decision_model.invoke(request)
    ans = response.answers["tool_selection"]
    if isinstance(ans, ChoiceAnswer):
        return ans
    if isinstance(ans, dict) and "choice" in ans:
        return ChoiceAnswer.model_validate(ans)
    raise TypeError(f"Expected ChoiceAnswer, got {type(ans).__name__}")
