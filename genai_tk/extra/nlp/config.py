"""NLP configuration model and accessor.

Provides a centralized ``nlp:`` YAML config section that defines language/model
defaults.  Individual consumers (Presidio, BM25, etc.) can override these
defaults on a per-use basis.

Example YAML::

    nlp:
      default_language: en
      default_model: en_core_web_sm
      models:
        en: en_core_web_sm
        fr: fr_core_news_sm
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class NlpConfig(BaseModel):
    """Top-level NLP configuration section."""

    default_language: str = Field(default="en", description="Default language code for NLP operations")
    default_model: str = Field(default="en_core_web_sm", description="Default spaCy model to use")
    models: dict[str, str] = Field(
        default_factory=lambda: {"en": "en_core_web_sm"},
        description="Mapping of language codes to spaCy model names",
    )

    def get_model_for_language(self, language: str | None = None) -> str:
        """Return the configured spaCy model for *language*, falling back to default.

        Args:
            language: Language code (e.g. ``"en"``, ``"fr"``). Uses ``default_language`` when None.

        Returns:
            spaCy model name.

        Raises:
            ValueError: If the language is not configured and no default model exists.
        """
        lang = language or self.default_language
        if lang in self.models:
            return self.models[lang]
        return self.default_model


def nlp_config() -> NlpConfig:
    """Return the global NLP configuration (from ``nlp:`` YAML section).

    Returns an ``NlpConfig`` with defaults when the section is absent.
    """
    from genai_tk.config_mgmt.config_mngr import global_config

    return global_config().section("nlp", NlpConfig)
