"""Presidio + spaCy PII detector.

Wraps Microsoft Presidio's ``AnalyzerEngine`` with spaCy NER support and
custom pattern-based recognizers. Uses the centralized :func:`~genai_tk.extra.nlp.engine.get_nlp`
for spaCy model management.

Example:
    ```python
    from genai_tk.extra.nlp.presidio import PresidioDetector, PresidioDetectorConfig

    config = PresidioDetectorConfig(
        analyzed_fields=["PERSON", "EMAIL_ADDRESS"],
        language="fr",  # Will use model from NlpConfig.models["fr"]
    )
    detector = PresidioDetector(config=config)
    entities = detector.detect("Appelez Jean au jean@example.com")
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from genai_tk.config_mgmt.features import require_feature
from genai_tk.utils.singleton import once

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineBase


class CustomRecognizerConfig(BaseModel):
    """Configuration for a single custom pattern-based Presidio recognizer."""

    entity_name: str
    patterns: list[str] = Field(default_factory=list, description="Regex patterns to match the entity")
    context: list[str] = Field(default_factory=list, description="Context words that boost confidence")
    score: float = Field(default=0.9, ge=0.0, le=1.0)


class PresidioDetectorConfig(BaseModel):
    """Configuration for the Presidio-based PII detector."""

    analyzed_fields: list[str] = Field(
        default_factory=lambda: [
            "PERSON",
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "CREDIT_CARD",
            "LOCATION",
            "IBAN_CODE",
            "US_SSN",
            "IP_ADDRESS",
        ],
        description="Presidio entity types to detect",
    )
    custom_recognizers: list[CustomRecognizerConfig] = Field(
        default_factory=list,
        description="Custom pattern-based recognizers to add to the analyzer",
    )
    spacy_model: str | None = Field(
        default=None,
        description="spaCy model name. Falls back to NlpConfig when None.",
    )
    language: str | None = Field(
        default=None,
        description="Language code. Falls back to NlpConfig.default_language when None.",
    )
    enable_spacy: bool = Field(default=True, description="Enable spaCy-backed recognizer in Presidio")
    score_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum Presidio confidence score to report a detection",
    )

    def resolve_language(self) -> str:
        """Return the effective language, falling back to NlpConfig."""
        if self.language:
            return self.language
        from genai_tk.extra.nlp.config import nlp_config

        return nlp_config().default_language

    def resolve_model(self) -> str:
        """Return the effective spaCy model, falling back to NlpConfig."""
        if self.spacy_model:
            return self.spacy_model
        from genai_tk.extra.nlp.config import nlp_config

        cfg = nlp_config()
        return cfg.get_model_for_language(self.resolve_language())


class DetectedEntity(BaseModel):
    """A single PII entity detected in a text."""

    entity_type: str
    start: int
    end: int
    text: str
    score: float


class PresidioDetector(BaseModel):
    """PII detector backed by Microsoft Presidio and optionally spaCy.

    The underlying ``AnalyzerEngine`` is built lazily on the first ``detect()``
    call and cached per-config via ``@once`` so the expensive initialization
    only happens once per process for each unique configuration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: PresidioDetectorConfig = Field(default_factory=PresidioDetectorConfig)
    _analyzer: AnalyzerEngine | None = PrivateAttr(default=None)

    def detect(self, text: str) -> list[DetectedEntity]:
        """Detect PII entities in *text*.

        Args:
            text: Input text to analyze.

        Returns:
            List of detected entities sorted by start position.
        """
        if self._analyzer is None:
            self._analyzer = _get_analyzer(self.config.model_dump_json())
        if not text or not text.strip():
            return []

        entities = self.config.analyzed_fields + [r.entity_name for r in self.config.custom_recognizers]
        language = self.config.resolve_language()
        try:
            results = self._analyzer.analyze(
                text=text,
                entities=entities,
                language=language,
                score_threshold=self.config.score_threshold,
            )
        except Exception:
            return []

        return sorted(
            [
                DetectedEntity(
                    entity_type=r.entity_type,
                    start=r.start,
                    end=r.end,
                    text=text[r.start : r.end],
                    score=r.score,
                )
                for r in results
            ],
            key=lambda e: e.start,
        )


# ---------------------------------------------------------------------------
# Module-level @once singletons
# ---------------------------------------------------------------------------


@once
def _ensure_spacy_once(model_name: str) -> None:
    """Download / verify *model_name* exactly once per process."""
    from genai_tk.extra.nlp.model_manager import SpaCyModelManager

    try:
        SpaCyModelManager.setup_spacy_model(model_name)
    except Exception:
        pass  # Best-effort: if setup fails, Presidio will try to load anyway


@once
def _get_nlp_engine(language: str, model_name: str) -> NlpEngineBase | None:
    """Return a spaCy-backed Presidio NLP engine, built once per (language, model)."""
    _ensure_spacy_once(model_name)
    try:
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": language, "model_name": model_name}],
            }
        )
        return provider.create_engine()
    except Exception:
        return None


@once
def _get_analyzer(config_json: str) -> AnalyzerEngine:
    """Return a Presidio ``AnalyzerEngine`` built from *config_json*, cached per unique config."""
    import logging

    require_feature("nlp", context="genai_tk.extra.nlp.presidio")

    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

    cfg = PresidioDetectorConfig.model_validate_json(config_json)
    language = cfg.resolve_language()
    model_name = cfg.resolve_model()

    # Suppress noisy Presidio warnings about language-specific recognizers
    presidio_logger = logging.getLogger("presidio-analyzer")
    old_level = presidio_logger.level
    presidio_logger.setLevel(logging.ERROR)

    try:
        if cfg.enable_spacy:
            nlp_engine = _get_nlp_engine(language, model_name)
            if nlp_engine is not None:
                from presidio_analyzer import RecognizerRegistry

                registry = RecognizerRegistry()
                registry.load_predefined_recognizers(languages=[language])
                analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
            else:
                analyzer = AnalyzerEngine()
        else:
            from presidio_analyzer import RecognizerRegistry
            from presidio_analyzer.nlp_engine import SpacyNlpEngine

            nlp_engine = SpacyNlpEngine(models=[])
            registry = RecognizerRegistry()
            registry.load_predefined_recognizers(languages=[language])
            analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)

        for rec_cfg in cfg.custom_recognizers:
            patterns = [
                Pattern(name=f"{rec_cfg.entity_name}_pattern_{i}", regex=p, score=rec_cfg.score)
                for i, p in enumerate(rec_cfg.patterns)
            ]
            recognizer = PatternRecognizer(
                supported_entity=rec_cfg.entity_name,
                patterns=patterns,
                context=rec_cfg.context or [],
            )
            analyzer.registry.add_recognizer(recognizer)

        return analyzer
    finally:
        presidio_logger.setLevel(old_level)
