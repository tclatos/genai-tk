"""Sensitivity scorer backed by a System One decision model.

Adapts :class:`~genai_tk.core.decision_models.base.BaseDecisionModel` to the
``assess`` protocol consumed by
:class:`~genai_tk.agents.langchain.middleware.sensitivity_router_middleware.SensitivityRouterMiddleware`:
the message text is used as decision state and a single ``Noul`` question returns
the calibrated probability that the message is sensitive.

Example:

    ```python
    from genai_tk.agents.langchain.middleware.decision_model_scorer import DecisionModelSensitivityScorer

    scorer = DecisionModelSensitivityScorer()  # defaults to Laya (local, in-process router)
    result = scorer.assess("My password is hunter2")
    print(result.is_sensitive, result.score)
    ```
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from genai_tk.core.decision_models import BaseDecisionModel, ClassifierRequest, Noul, NoulAnswer
from genai_tk.extra.nlp.classifiers.sensitivity import SensitivityAssessment

_DEFAULT_INSTRUCTIONS = (
    "Does this message contain sensitive information that must not be sent to an "
    "untrusted model? Consider passwords, API keys, tokens, credit card or bank "
    "details, personal data (email, phone, address, national ID), health data, "
    "salaries, or anything explicitly marked private or confidential."
)


def _score_level(score: float) -> Literal["low", "medium", "high", "critical"]:
    """Map a probability to a discrete level (same cut-offs as DefaultSensitivityScorer)."""
    if score < 0.25:
        return "low"
    if score < 0.50:
        return "medium"
    if score < 0.75:
        return "high"
    return "critical"


class DecisionModelSensitivityScorer(BaseModel):
    """Assess message sensitivity with a decision model instead of regex/Presidio.

    The scorer sends the whole message as decision state and asks one binary
    (``Noul``) question, so classification cost does not grow with the number of
    detection rules and benefits from the decision model's calibrated probability.

    Attributes:
        model_id: Decision model id (``model@provider``) or tag, resolved lazily
            via the decision model factory.  Defaults to Laya running locally
            in-process.
        model_kwargs: Options forwarded to the factory (e.g. ``device='cpu'``,
            ``preload=True``).
        instructions: Instructions of the ``Noul`` sensitivity question.
        threshold: Probability above which the message is considered sensitive.
        decision_model: Pre-built decision model instance; ``model_id`` is
            ignored when set.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: str = "laya@laya"
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    instructions: str = _DEFAULT_INSTRUCTIONS
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    decision_model: BaseDecisionModel | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.decision_model is None:
            from genai_tk.core.factories.decision_model_factory import get_decision_model

            self.decision_model = get_decision_model(self.model_id, **self.model_kwargs)

    def classify(self, text: str) -> SensitivityAssessment:
        """Classify *text* for sensitivity (TextClassifier protocol)."""
        return self.assess(text)

    def assess(self, text: str) -> SensitivityAssessment:
        """Return a sensitivity assessment for *text* using the decision model."""
        if not text or not text.strip():
            return SensitivityAssessment(
                is_sensitive=False,
                score=0.0,
                level="low",
                summary="No content to analyze.",
            )

        assert self.decision_model is not None  # set in model_post_init
        request = ClassifierRequest(state=text, questions={"sensitivity": Noul(instructions=self.instructions)})
        response = self.decision_model.invoke(request)

        answer = response.answers.get("sensitivity")
        if not isinstance(answer, NoulAnswer):
            raise TypeError(f"Decision model '{response.model}' returned {type(answer).__name__}, expected NoulAnswer")

        probability = answer.noul
        return SensitivityAssessment(
            is_sensitive=probability >= self.threshold,
            score=probability,
            level=_score_level(probability),
            summary=f"p(sensitive)={probability:.2f} (threshold={self.threshold:.2f}, model={response.model})",
            labels=["decision_model"],
            details={"model": response.model, "provider": response.provider, "threshold": self.threshold},
        )
