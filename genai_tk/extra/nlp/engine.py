"""Centralized spaCy NLP engine — single entry point for loading spaCy pipelines.

All code that needs a spaCy ``Language`` object should go through :func:`get_nlp`
rather than calling ``spacy.load()`` directly. This ensures:

1. The ``nlp`` feature gate is checked once.
2. Models are resolved via :class:`~genai_tk.extra.nlp.config.NlpConfig`.
3. Loaded pipelines are cached per (language, model) pair.
4. Clear error messages when a model is not installed (especially for non-English).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Never

from loguru import logger

from genai_tk.config_mgmt.features import require_feature
from genai_tk.extra.nlp.config import nlp_config
from genai_tk.extra.nlp.model_manager import SpaCyModelManager

if TYPE_CHECKING:
    from spacy.language import Language

# Cache: (language, model_name) → spacy.Language
_nlp_cache: dict[tuple[str, str], Language] = {}


def get_nlp(language: str | None = None, model: str | None = None) -> Language:
    """Return a cached spaCy ``Language`` pipeline for the given language/model.

    Args:
        language: Language code (e.g. ``"en"``, ``"fr"``). Defaults to ``NlpConfig.default_language``.
        model: spaCy model name. Defaults to the model configured for *language* in ``NlpConfig``.

    Returns:
        A loaded spaCy ``Language`` object.

    Raises:
        ImportError: If the ``nlp`` feature is not installed or the specific model is missing.
    """
    require_feature("nlp", context="genai_tk.extra.nlp.engine")

    cfg = nlp_config()
    lang = language or cfg.default_language
    model_name = model or cfg.get_model_for_language(lang)

    cache_key = (lang, model_name)
    if cache_key in _nlp_cache:
        return _nlp_cache[cache_key]

    import spacy

    # Try loading directly
    try:
        nlp: Language = spacy.load(model_name)
    except OSError:
        # Try custom path
        model_path = SpaCyModelManager.get_model_path(model_name)
        if model_path.exists():
            try:
                nlp = spacy.load(model_path)
            except Exception:
                _raise_model_not_found(model_name, lang)
        else:
            _raise_model_not_found(model_name, lang)

    logger.debug("Loaded spaCy model '{}' for language '{}'", model_name, lang)
    _nlp_cache[cache_key] = nlp
    return nlp


def _raise_model_not_found(model_name: str, language: str) -> Never:
    """Raise a descriptive ImportError for a missing spaCy model."""
    raise ImportError(
        f"spaCy model '{model_name}' for language '{language}' is not installed.\n"
        f"Install it with: python -m spacy download {model_name}\n"
        f"Or configure a different model in your config YAML under 'nlp.models.{language}'."
    )


def clear_cache() -> None:
    """Clear the NLP engine cache (useful in tests)."""
    _nlp_cache.clear()
