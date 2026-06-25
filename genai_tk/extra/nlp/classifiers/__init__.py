"""Text classifiers — protocol and implementations.

Provides a :class:`TextClassifier` protocol for reusable text classification,
with a default implementation for sensitivity scoring.
"""

from genai_tk.extra.nlp.classifiers.base import ClassificationResult, TextClassifier
from genai_tk.extra.nlp.classifiers.sensitivity import (
    DefaultScorerConfig,
    DefaultSensitivityScorer,
    SensitivityAssessment,
)

__all__ = [
    "ClassificationResult",
    "DefaultScorerConfig",
    "DefaultSensitivityScorer",
    "SensitivityAssessment",
    "TextClassifier",
]
