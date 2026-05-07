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
    detector = PresidioDetector(config=config)
    entities = detector.detect("Call John at john@example.com")
    ```
"""

from __future__ import annotations

import threading
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

# Module-level cache: config JSON → AnalyzerEngine instance.
# Avoids reloading the spaCy model (or presidio's default NLP engine) on every
# PresidioDetector instantiation.  Thread-safe via _CACHE_LOCK.
_ANALYZER_CACHE: dict[str, Any] = {}

# Separate cache for NlpEngine objects, keyed by (language, model_name).
# Two detector configs that share the same spaCy model but differ in other
# fields (analyzed_fields, custom_recognizers, …) reuse the same loaded
# NlpEngine rather than paying the spaCy load cost again.
_NLP_ENGINE_CACHE: dict[tuple[str, str], Any] = {}

_CACHE_LOCK = threading.Lock()


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

    Initialize once per middleware instance and reuse.  Not thread-safe if
    ``enable_spacy=True`` because the spaCy pipeline is not thread-safe.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: PresidioDetectorConfig = Field(default_factory=PresidioDetectorConfig)
    _analyzer: Any = PrivateAttr(default=None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str) -> list[DetectedEntity]:
        """Detect PII entities in *text*.

        The Presidio ``AnalyzerEngine`` is built lazily on the first call so
        that constructing a ``PresidioDetector`` does not trigger spaCy model
        loading.

        Args:
            text: Input text to analyze.

        Returns:
            List of detected entities sorted by start position.
        """
        if self._analyzer is None:
            self._analyzer = self._build_analyzer()
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_analyzer(self) -> Any:
        """Return a cached (or newly built) Presidio AnalyzerEngine.

        The engine is cached per unique config so the expensive spaCy model
        load (or presidio's default NLP engine initialisation) only happens
        once per process per distinct configuration.
        """
        cache_key = self.config.model_dump_json()
        if cache_key in _ANALYZER_CACHE:
            return _ANALYZER_CACHE[cache_key]
        with _CACHE_LOCK:
            if cache_key in _ANALYZER_CACHE:
                return _ANALYZER_CACHE[cache_key]
            analyzer = self._build_analyzer_uncached()
            _ANALYZER_CACHE[cache_key] = analyzer
        return analyzer

    def _build_analyzer_uncached(self) -> Any:
        """Build and configure the Presidio AnalyzerEngine (no caching)."""
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

        if self.config.enable_spacy:
            self._ensure_spacy()
            nlp_engine = self._build_spacy_nlp_engine()
            if nlp_engine is not None:
                from presidio_analyzer import RecognizerRegistry

                registry = RecognizerRegistry()
                registry.load_predefined_recognizers(languages=[self.config.language])
                analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
            else:
                analyzer = AnalyzerEngine()
        else:
            # Pass an explicit empty NLP engine so presidio does not trigger its
            # default NlpEngineProvider which loads en_core_web_sm even when spaCy
            # support is not needed.  Pattern-based recognizers (email, phone,
            # credit card, …) work without any NLP engine.
            from presidio_analyzer import RecognizerRegistry
            from presidio_analyzer.nlp_engine import SpacyNlpEngine

            nlp_engine = SpacyNlpEngine(models=[])
            registry = RecognizerRegistry()
            registry.load_predefined_recognizers(languages=[self.config.language])
            analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)

        # Register custom recognizers
        for rec_cfg in self.config.custom_recognizers:
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

    def _ensure_spacy(self) -> None:
        """Ensure the configured spaCy model is available."""
        try:
            from genai_tk.utils.spacy_model_mngr import SpaCyModelManager

            SpaCyModelManager.setup_spacy_model(self.config.spacy_model)
        except Exception:
            pass  # Best-effort: if setup fails, Presidio will try to load anyway

    def _build_spacy_nlp_engine(self) -> Any | None:
        """Build (or return a cached) spaCy-backed NLP engine for Presidio.

        The loaded spaCy model is cached in ``_NLP_ENGINE_CACHE`` keyed by
        ``(language, model_name)``.  Configs that share the same model name
        (but differ in other fields) reuse the already-loaded engine instead
        of paying the spaCy load cost again.
        """
        cache_key = (self.config.language, self.config.spacy_model)
        if cache_key in _NLP_ENGINE_CACHE:
            return _NLP_ENGINE_CACHE[cache_key]
        with _CACHE_LOCK:
            if cache_key in _NLP_ENGINE_CACHE:
                return _NLP_ENGINE_CACHE[cache_key]
            engine = self._build_spacy_nlp_engine_uncached()
            _NLP_ENGINE_CACHE[cache_key] = engine
        return engine

    def _build_spacy_nlp_engine_uncached(self) -> Any | None:
        """Build a fresh spaCy-backed NLP engine for Presidio (no caching)."""
        try:
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            provider = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": self.config.language, "model_name": self.config.spacy_model}],
                }
            )
            return provider.create_engine()
        except Exception:
            return None
