"""Presidio + spaCy PII detector — canonical implementation.

Lives in :mod:`genai_tk.workflow.anonymization` so it can be imported by the
anonymization core without touching the LangChain middleware package (which
would create a circular import).  The middleware package re-exports everything
from here for backward compatibility.

Wraps Microsoft Presidio's ``AnalyzerEngine`` with optional spaCy NER support and
registers custom pattern-based recognizers.  Designed to be composed into config
models rather than instantiated standalone.

Example:
    ```python
    from genai_tk.workflow.anonymization.presidio_detector import (
        CustomRecognizerConfig,
        DetectedEntity,
        PresidioDetector,
        PresidioDetectorConfig,
    )

    config = PresidioDetectorConfig(
        analyzed_fields=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
        custom_recognizers=[
            CustomRecognizerConfig(
                entity_name="COMPANY",
                patterns=[r"(?i)\\b(Acme|Globex)\\b"],
                context=["company", "firm"],
            )
        ],
    )
    detector = PresidioDetector(config=config)
    entities = detector.detect("Call John at john@example.com")
    ```
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from genai_tk.config_mgmt.features import require_feature
from genai_tk.utils.singleton import once


class CustomRecognizerConfig(BaseModel):
    """Configuration for a single custom pattern-based Presidio recognizer."""

    entity_name: str
    patterns: list[str] = Field(default_factory=list, description="Regex patterns to match the entity")
    context: list[str] = Field(default_factory=list, description="Context words that boost confidence")
    score: float = Field(default=0.9, ge=0.0, le=1.0)


class PresidioDetectorConfig(BaseModel):
    """Configuration for the shared Presidio-based PII detector."""

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
    spacy_model: str = Field(
        default="en_core_web_sm",
        description="spaCy model name to use for NER",
    )
    language: str = Field(default="en", description="Language code for Presidio analysis")
    enable_spacy: bool = Field(default=True, description="Enable spaCy-backed recognizer in Presidio")
    score_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum Presidio confidence score to report a detection",
    )


class DetectedEntity(BaseModel):
    """A single PII entity detected in a text."""

    entity_type: str
    start: int
    end: int
    text: str
    score: float


class PresidioDetector(BaseModel):
    """PII detector backed by Microsoft Presidio and optionally spaCy.

    The underlying ``AnalyzerEngine`` (and the spaCy model it uses) are built
    lazily on the first ``detect()`` call and cached per-config via
    :func:`genai_tk.utils.singleton.once` so the expensive load only happens
    once per process regardless of how many ``PresidioDetector`` instances share
    the same configuration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: PresidioDetectorConfig = Field(default_factory=PresidioDetectorConfig)
    _analyzer: Any = PrivateAttr(default=None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        try:
            results = self._analyzer.analyze(
                text=text,
                entities=entities,
                language=self.config.language,
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
# Module-level @once singletons — each has its own lock (no shared lock,
# no possibility of self-deadlock)
# ---------------------------------------------------------------------------


@once
def _ensure_spacy_once(model_name: str) -> None:
    """Download / verify *model_name* exactly once per process."""
    try:
        from genai_tk.utils.spacy_model_mngr import SpaCyModelManager

        SpaCyModelManager.setup_spacy_model(model_name)
    except Exception:
        pass  # Best-effort: if setup fails, Presidio will try to load anyway


@once
def _get_nlp_engine(language: str, model_name: str) -> Any | None:
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
def _get_analyzer(config_json: str) -> Any:
    """Return a Presidio ``AnalyzerEngine`` built from *config_json*, cached per unique config."""
    import logging

    require_feature("nlp", context="genai_tk.workflow.anonymization.presidio_detector")

    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

    cfg = PresidioDetectorConfig.model_validate_json(config_json)

    # Suppress "Recognizer not added to registry" warnings from Presidio when
    # language-specific recognizers don't match the requested language.
    # (E.g., Spanish/Italian/Polish recognizers being skipped for English.)
    presidio_logger = logging.getLogger("presidio-analyzer")
    old_level = presidio_logger.level
    presidio_logger.setLevel(logging.ERROR)

    try:
        if cfg.enable_spacy:
            nlp_engine = _get_nlp_engine(cfg.language, cfg.spacy_model)
            if nlp_engine is not None:
                from presidio_analyzer import RecognizerRegistry

                registry = RecognizerRegistry()
                registry.load_predefined_recognizers(languages=[cfg.language])
                analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
            else:
                analyzer = AnalyzerEngine()
        else:
            # Explicit empty NLP engine so Presidio does not trigger its default
            # NlpEngineProvider (which loads en_core_web_sm unconditionally).
            from presidio_analyzer import RecognizerRegistry
            from presidio_analyzer.nlp_engine import SpacyNlpEngine

            nlp_engine = SpacyNlpEngine(models=[])
            registry = RecognizerRegistry()
            registry.load_predefined_recognizers(languages=[cfg.language])
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
