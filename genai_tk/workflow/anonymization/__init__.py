"""PII anonymization core — DEPRECATED: use ``genai_tk.extra.nlp`` instead.

This package is kept for backward compatibility. All symbols are re-exported
from :mod:`genai_tk.extra.nlp`. Import from there in new code.
"""

import warnings

warnings.warn(
    "genai_tk.workflow.anonymization is deprecated. "
    "Use genai_tk.extra.nlp instead (e.g. from genai_tk.extra.nlp import PresidioDetector, anonymize_text).",
    DeprecationWarning,
    stacklevel=2,
)

from genai_tk.extra.nlp.anonymization import (  # noqa: E402
    AnonymizationConfig,
    _deduplicate_entities,
    anonymize_text,
    make_fake_value,
)
from genai_tk.extra.nlp.presidio import (  # noqa: E402
    CustomRecognizerConfig,
    DetectedEntity,
    PresidioDetector,
    PresidioDetectorConfig,
)

__all__ = [
    "AnonymizationConfig",
    "PresidioDetectorConfig",
    "PresidioDetector",
    "DetectedEntity",
    "CustomRecognizerConfig",
    "anonymize_text",
    "make_fake_value",
    "_deduplicate_entities",
]
