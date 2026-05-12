"""Backward-compatibility re-export — the implementation has moved to
:mod:`genai_tk.workflow.anonymization.presidio_detector`.

All names are re-exported unchanged so existing
``from genai_tk.agents.langchain.middleware.presidio_detector import ...``
calls continue to work without modification.
"""

from genai_tk.workflow.anonymization.presidio_detector import (
    CustomRecognizerConfig,
    DetectedEntity,
    PresidioDetector,
    PresidioDetectorConfig,
)

__all__ = [
    "CustomRecognizerConfig",
    "DetectedEntity",
    "PresidioDetector",
    "PresidioDetectorConfig",
]
