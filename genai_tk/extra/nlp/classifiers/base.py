"""Base protocol and result model for text classifiers."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    """Base result model returned by text classifiers."""

    score: float = Field(ge=0.0, le=1.0, description="Overall classification score (0-1)")
    level: Literal["low", "medium", "high", "critical"] = Field(description="Discrete level")
    labels: list[str] = Field(default_factory=list, description="Applicable classification labels")
    details: dict[str, Any] = Field(default_factory=dict, description="Classifier-specific detail breakdown")


class TextClassifier(Protocol):
    """Protocol for text classifiers.

    Implementations must expose a single ``classify`` method. The returned
    :class:`ClassificationResult` (or a subclass) provides structured output
    suitable for routing, filtering, or logging decisions.
    """

    def classify(self, text: str) -> ClassificationResult:
        """Classify *text* and return a structured result.

        Args:
            text: Input text to classify.

        Returns:
            A :class:`ClassificationResult` (or subclass) with score, level, and details.
        """
        ...
