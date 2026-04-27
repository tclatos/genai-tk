"""Shared Presidio + spaCy PII detector used by anonymization and routing middlewares.

Wraps Microsoft Presidio's ``AnalyzerEngine`` with optional spaCy NER support and
registers custom pattern-based recognizers.  Designed to be composed into middleware
config models rather than instantiated standalone.

Example:
    ```python
    from genai_tk.agents.langchain.middleware.presidio_detector import (
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
    detector = PresidioDetector(config)
    entities = detector.detect("Call John at john@example.com")
    ```
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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


class PresidioDetector:
    """PII detector backed by Microsoft Presidio and optionally spaCy.

    Initialize once per middleware instance and reuse.  Not thread-safe if
    ``enable_spacy=True`` because the spaCy pipeline is not thread-safe.

    Args:
        config: Detector configuration.
    """

    def __init__(self, config: PresidioDetectorConfig | None = None) -> None:
        self._config = config or PresidioDetectorConfig()
        self._analyzer = self._build_analyzer()

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
        if not text or not text.strip():
            return []

        entities = self._config.analyzed_fields + [r.entity_name for r in self._config.custom_recognizers]
        try:
            results = self._analyzer.analyze(
                text=text,
                entities=entities,
                language=self._config.language,
                score_threshold=self._config.score_threshold,
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_analyzer(self) -> Any:
        """Build and configure the Presidio AnalyzerEngine."""
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

        if self._config.enable_spacy:
            self._ensure_spacy()
            nlp_engine = self._build_spacy_nlp_engine()
            if nlp_engine is not None:
                from presidio_analyzer import RecognizerRegistry

                registry = RecognizerRegistry()
                registry.load_predefined_recognizers(languages=[self._config.language])
                analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
            else:
                analyzer = AnalyzerEngine()
        else:
            analyzer = AnalyzerEngine(nlp_engine=None)

        # Register custom recognizers
        for rec_cfg in self._config.custom_recognizers:
            patterns = [
                Pattern(name=f"{rec_cfg.entity_name}_pattern_{i}", regex=p, score=rec_cfg.score)
                for i, p in enumerate(rec_cfg.patterns)
            ]
            recognizer = PatternRecognizer(
                supported_entity=rec_cfg.entity_name,
                patterns=patterns,
                context=rec_cfg.context or None,
            )
            analyzer.registry.add_recognizer(recognizer)

        return analyzer

    def _ensure_spacy(self) -> None:
        """Ensure the configured spaCy model is available."""
        try:
            from genai_tk.utils.spacy_model_mngr import SpaCyModelManager

            SpaCyModelManager.setup_spacy_model(self._config.spacy_model)
        except Exception:
            pass  # Best-effort: if setup fails, Presidio will try to load anyway

    def _build_spacy_nlp_engine(self) -> Any | None:
        """Build a spaCy-backed NLP engine for Presidio."""
        try:
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            provider = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": self._config.language, "model_name": self._config.spacy_model}],
                }
            )
            return provider.create_engine()
        except Exception:
            return None
