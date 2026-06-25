"""DEPRECATED: use genai_tk.extra.nlp.presidio instead.

This module is kept for backward compatibility only.
"""

import warnings

warnings.warn(
    "genai_tk.workflow.anonymization.presidio_detector is deprecated. "
    "Use genai_tk.extra.nlp.presidio (or genai_tk.extra.nlp) instead.",
    DeprecationWarning,
    stacklevel=2,
)

from genai_tk.extra.nlp.presidio import (  # noqa: F401, E402
    CustomRecognizerConfig,
    DetectedEntity,
    PresidioDetector,
    PresidioDetectorConfig,
)

__all__ = ["CustomRecognizerConfig", "DetectedEntity", "PresidioDetector", "PresidioDetectorConfig"]
