"""PII anonymization core — shared between the Prefect ETL flow and LangChain middleware.

Public API
----------
Configuration & detector (Pydantic models)
    :class:`~genai_tk.workflow.anonymization.core.AnonymizationConfig`
    :class:`~genai_tk.workflow.anonymization.presidio_detector.PresidioDetectorConfig`
    :class:`~genai_tk.workflow.anonymization.presidio_detector.PresidioDetector`
    :class:`~genai_tk.workflow.anonymization.presidio_detector.DetectedEntity`
    :class:`~genai_tk.workflow.anonymization.presidio_detector.CustomRecognizerConfig`

Stateless helpers
    :func:`~genai_tk.workflow.anonymization.core.anonymize_text`
    :func:`~genai_tk.workflow.anonymization.core.make_fake_value`
"""

from genai_tk.workflow.anonymization.core import (
    AnonymizationConfig,
    _deduplicate_entities,
    anonymize_text,
    make_fake_value,
)
from genai_tk.workflow.anonymization.presidio_detector import (
    CustomRecognizerConfig,
    DetectedEntity,
    PresidioDetector,
    PresidioDetectorConfig,
)

__all__ = [
    # Config
    "AnonymizationConfig",
    # Detector
    "PresidioDetectorConfig",
    "PresidioDetector",
    "DetectedEntity",
    "CustomRecognizerConfig",
    # Helpers
    "anonymize_text",
    "make_fake_value",
    "_deduplicate_entities",
]
