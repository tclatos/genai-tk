"""Mock / Fake decision model for testing."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from genai_tk.core.decision_models.base import BaseDecisionModel
from genai_tk.core.decision_models.types import (
    Answer,
    Choice,
    ChoiceAnswer,
    ClassifierRequest,
    ClassifierResponse,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    Usage,
)


class FakeDecisionModel(BaseDecisionModel):
    """Deterministic fake decision model for offline unit tests."""

    model: str = "fake_decision@fake"
    fixed_noul: float = 0.85
    fixed_choice: str | None = None
    fixed_score: float | None = None

    def invoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        request = input if isinstance(input, ClassifierRequest) else ClassifierRequest.model_validate(input)
        self.validate_question_count(request)
        with self._trace_scope(request) as handle:
            answers: dict[str, Answer] = {}

            for q_id, q in request.questions.items():
                if isinstance(q, Noul):
                    answers[q_id] = NoulAnswer(noul=self.fixed_noul)
                elif isinstance(q, Choice):
                    keys = list(q.criteria.keys())
                    choice_val = self.fixed_choice if self.fixed_choice in keys else (keys[0] if keys else "default")
                    probs = {k: (1.0 if k == choice_val else 0.0) for k in keys}
                    answers[q_id] = ChoiceAnswer(choice=choice_val, confidence=0.95, probabilities=probs)
                elif isinstance(q, Score):
                    n_levels = len(q.criteria)
                    score_val = self.fixed_score if self.fixed_score is not None else 1.0
                    probs = {i: (1.0 if i == int(score_val) else 0.0) for i in range(n_levels)}
                    legend = {i: q.criteria[i] for i in range(n_levels)}
                    answers[q_id] = ScoreAnswer(
                        score=score_val,
                        confidence=0.9,
                        probabilities=probs,
                        legend=legend,
                    )

            response = ClassifierResponse(
                model=self.model,
                answers=answers,
                usage=Usage(input_tokens=10, output_tokens=0, cost=0.0),
                provider="fake",
                request_id="fake-test-id",
            )
            self._emit_trace_event(handle, response)
            return response

    async def ainvoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        return self.invoke(input, config=config, **kwargs)
