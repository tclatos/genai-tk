"""DEPRECATED: use genai_tk.extra.nlp.anonymization instead.

This module is kept for backward compatibility only.
"""

import warnings

warnings.warn(
    "genai_tk.workflow.anonymization.core is deprecated. "
    "Use genai_tk.extra.nlp.anonymization (or genai_tk.extra.nlp) instead.",
    DeprecationWarning,
    stacklevel=2,
)

from genai_tk.extra.nlp.anonymization import (  # noqa: F401, E402
    AnonymizationConfig,
    _deduplicate_entities,
    anonymize_text,
    make_fake_value,
)

__all__ = ["AnonymizationConfig", "_deduplicate_entities", "anonymize_text", "make_fake_value"]
