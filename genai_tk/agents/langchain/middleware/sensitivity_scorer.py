"""Sensitivity scorer protocol and default implementation.

This module re-exports from :mod:`genai_tk.extra.nlp.classifiers.sensitivity` for
backward compatibility. New code should import directly from that module.

Example:
    ```python
    from genai_tk.extra.nlp.classifiers import DefaultSensitivityScorer, DefaultScorerConfig

    scorer = DefaultSensitivityScorer(DefaultScorerConfig())
    result = scorer.classify("My email is john@example.com")
    print(result.is_sensitive, result.score, result.level)
    ```
"""

from __future__ import annotations

from genai_tk.extra.nlp.classifiers.sensitivity import (
    DefaultScorerConfig,
    DefaultSensitivityScorer,
    KeywordGroupConfig,
    RegexPatternConfig,
    SensitivityAssessment,
)
from genai_tk.extra.nlp.presidio import DetectedEntity

# ---------------------------------------------------------------------------
# Legacy protocol — kept for type compatibility with existing middleware
# ---------------------------------------------------------------------------


class SensitivityScorer:
    """Protocol for sensitivity scorers.

    Implementations must expose a single ``assess`` method.
    """

    def assess(self, text: str) -> SensitivityAssessment:
        """Assess the sensitivity of *text*."""
        raise NotImplementedError


__all__ = [
    "DefaultScorerConfig",
    "DefaultSensitivityScorer",
    "DetectedEntity",
    "KeywordGroupConfig",
    "RegexPatternConfig",
    "SensitivityAssessment",
    "SensitivityScorer",
]
