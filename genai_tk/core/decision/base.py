"""Base class for decision models (System One models)."""

from __future__ import annotations

import abc
from typing import Any

from langchain_core.runnables import RunnableConfig, RunnableSerializable
from pydantic import ConfigDict

from genai_tk.core.decision.types import (
    Choice,
    ChoiceAnswer,
    ClassifierRequest,
    ClassifierResponse,
    DecisionState,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Score,
    ScoreAnswer,
)


class BaseDecisionModel(RunnableSerializable[ClassifierRequest, ClassifierResponse], abc.ABC):
    """Abstract base class for decision models.

    Inherits from LangChain RunnableSerializable to integrate cleanly with chains,
    agents, and runnables while operating on typed ClassifierRequest / ClassifierResponse.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abc.abstractmethod
    def invoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        """Execute a decision request synchronously."""
        ...

    @abc.abstractmethod
    async def ainvoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        """Execute a decision request asynchronously."""
        ...

    def decide_noul(
        self,
        state: DecisionState,
        instructions: str,
        criteria: NoulCriteria | dict[str, Any] | None = None,
        question_id: str = "decision",
    ) -> float:
        """Convenience method to evaluate a single boolean question returning probability [0.0, 1.0]."""
        request = ClassifierRequest(
            state=state,
            questions={question_id: Noul(instructions=instructions, criteria=criteria)},
        )
        response = self.invoke(request)
        answer = response.answers[question_id]
        if isinstance(answer, NoulAnswer):
            return answer.noul
        raise TypeError(f"Expected NoulAnswer, got {type(answer).__name__}")

    def decide_choice(
        self,
        state: DecisionState,
        instructions: str,
        criteria: dict[str, Any],
        question_id: str = "decision",
    ) -> ChoiceAnswer:
        """Convenience method to classify state into one of several categorical options."""
        request = ClassifierRequest(
            state=state,
            questions={question_id: Choice(instructions=instructions, criteria=criteria)},
        )
        response = self.invoke(request)
        answer = response.answers[question_id]
        if isinstance(answer, ChoiceAnswer):
            return answer
        raise TypeError(f"Expected ChoiceAnswer, got {type(answer).__name__}")

    def decide_score(
        self,
        state: DecisionState,
        instructions: str,
        criteria: list[Any],
        question_id: str = "decision",
    ) -> ScoreAnswer:
        """Convenience method to score state against an ordered rubric."""
        request = ClassifierRequest(
            state=state,
            questions={question_id: Score(instructions=instructions, criteria=criteria)},
        )
        response = self.invoke(request)
        answer = response.answers[question_id]
        if isinstance(answer, ScoreAnswer):
            return answer
        raise TypeError(f"Expected ScoreAnswer, got {type(answer).__name__}")
