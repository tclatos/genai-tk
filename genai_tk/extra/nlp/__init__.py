"""Centralized NLP support — spaCy model management, preprocessing, PII detection, and text classification.

This package consolidates all spaCy and NLP-related functionality that was
previously scattered across ``workflow/anonymization/``, ``utils/``, and
``agents/langchain/middleware/``.

All public symbols are importable directly from this package:

    >>> from genai_tk.extra.nlp import NlpConfig, nlp_config, get_nlp, get_spacy_preprocess_fn
    >>> from genai_tk.extra.nlp import PresidioDetector, PresidioDetectorConfig, DetectedEntity
    >>> from genai_tk.extra.nlp import anonymize_text, AnonymizationConfig
"""

from genai_tk.extra.nlp.anonymization import AnonymizationConfig, anonymize_text, make_fake_value
from genai_tk.extra.nlp.config import NlpConfig, nlp_config
from genai_tk.extra.nlp.engine import get_nlp
from genai_tk.extra.nlp.language import (
    BUSINESS_LANGUAGES,
    detect_language,
    get_language_detector,
)
from genai_tk.extra.nlp.model_manager import SpaCyModelManager
from genai_tk.extra.nlp.preprocessing import get_spacy_preprocess_fn
from genai_tk.extra.nlp.presidio import (
    CustomRecognizerConfig,
    DetectedEntity,
    PresidioDetector,
    PresidioDetectorConfig,
)
from genai_tk.extra.nlp.stopwords import (
    get_dominant_language,
    get_ladybug_stemmer,
    get_stopwords,
    get_stopwords_union,
)

__all__ = [
    "BUSINESS_LANGUAGES",
    "AnonymizationConfig",
    "CustomRecognizerConfig",
    "DetectedEntity",
    "NlpConfig",
    "PresidioDetector",
    "PresidioDetectorConfig",
    "SpaCyModelManager",
    "anonymize_text",
    "detect_language",
    "get_dominant_language",
    "get_ladybug_stemmer",
    "get_language_detector",
    "get_nlp",
    "get_spacy_preprocess_fn",
    "get_stopwords",
    "get_stopwords_union",
    "make_fake_value",
    "nlp_config",
]
