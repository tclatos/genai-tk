"""Sensitivity scorer protocol and default implementation.

The :class:`SensitivityScorer` protocol defines the interface consumed by
:class:`~genai_tk.agents.langchain.middleware.sensitivity_router_middleware.SensitivityRouterMiddleware`.
The default implementation combines multiple detection strategies:

- **Regex** — structured patterns (email, phone, credit card, IBAN, JWT, API keys, …)
- **Keywords** — contextual vocabulary groups (credentials, identity, financial, health)
- **Banwords** — critical phrases that always indicate high sensitivity
- **Presidio** — Microsoft Presidio entity recognition
- **Heuristics** — disclosure phrases, high digit density, long mixed tokens

Each strategy contributes a capped weighted sub-score.  The final score is a
sum with a small diversity boost.

Example:
    ```python
    from genai_tk.agents.langchain.middleware.sensitivity_scorer import (
        DefaultSensitivityScorer,
        DefaultScorerConfig,
        SensitivityAssessment,
    )

    scorer = DefaultSensitivityScorer(DefaultScorerConfig())
    result: SensitivityAssessment = scorer.assess("My email is john@example.com")
    print(result.is_sensitive, result.score, result.level)
    ```
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from genai_tk.agents.langchain.middleware.presidio_detector import (
    DetectedEntity,
    PresidioDetector,
    PresidioDetectorConfig,
)

# ---------------------------------------------------------------------------
# Hard-coded defaults
# ---------------------------------------------------------------------------

_DEFAULT_REGEX_PATTERNS: list[dict[str, Any]] = [
    {"name": "email", "pattern": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "weight": 0.28, "label": "EMAIL"},
    {
        "name": "phone",
        "pattern": r"(?:(?<=\s)|^)(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{2,4}\)?[ .-]?)?\d(?:[ .-]?\d){6,12}(?=(?:\s|$|[.,;:]))",
        "weight": 0.22,
        "label": "PHONE",
    },
    {"name": "credit_card", "pattern": r"\b(?:\d[ -]?){13,19}\b", "weight": 0.36, "label": "CREDIT_CARD"},
    {"name": "iban", "pattern": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", "weight": 0.32, "label": "IBAN"},
    {
        "name": "ipv4",
        "pattern": r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b",
        "weight": 0.10,
        "label": "IP_ADDRESS",
    },
    {
        "name": "jwt",
        "pattern": r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9._-]{10,}\.[A-Za-z0-9._-]{10,}\b",
        "weight": 0.34,
        "label": "JWT",
    },
    {"name": "bearer_token", "pattern": r"\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b", "weight": 0.34, "label": "ACCESS_TOKEN"},
    {
        "name": "api_key",
        "pattern": r"\b(?:sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z\-_]{20,}|ghp_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})\b",
        "weight": 0.34,
        "label": "API_KEY",
    },
    {
        "name": "password",
        "pattern": r"\b(?:password|passwd|pwd|mot de passe|secret)\b\s*[:=]\s*[^\s,;]+",
        "weight": 0.33,
        "label": "SECRET",
    },
    {
        "name": "private_key",
        "pattern": r"-----BEGIN[ A-Z]+PRIVATE KEY-----[\s\S]+?-----END[ A-Z]+PRIVATE KEY-----",
        "weight": 0.45,
        "label": "PRIVATE_KEY",
    },
]

_DEFAULT_KEYWORD_GROUPS: dict[str, dict[str, Any]] = {
    "credentials": {
        "weight": 0.08,
        "terms": ["password", "mot de passe", "api key", "token", "secret", "private key", "bearer", "credential"],
    },
    "identity": {
        "weight": 0.07,
        "terms": ["ssn", "social security", "passport", "date of birth", "birth date", "phone", "email"],
    },
    "financial": {
        "weight": 0.08,
        "terms": ["iban", "bank account", "credit card", "card number", "cvv", "salary", "payroll"],
    },
    "health": {"weight": 0.08, "terms": ["medical", "patient", "diagnosis", "health record", "dossier medical"]},
    "confidentiality": {
        "weight": 0.06,
        "terms": ["confidential", "strictly private", "internal only", "do not share", "sensitive"],
    },
}

_DEFAULT_BANWORDS: list[str] = [
    "root password",
    "private key",
    "prod database dump",
    "admin password",
    "ssh key",
    "database dump",
]

_DEFAULT_ENTITY_WEIGHTS: dict[str, float] = {
    "EMAIL_ADDRESS": 0.28,
    "PHONE_NUMBER": 0.22,
    "PERSON": 0.12,
    "LOCATION": 0.08,
    "CREDIT_CARD": 0.36,
    "IBAN_CODE": 0.32,
    "IP_ADDRESS": 0.10,
    "URL": 0.05,
    "US_SSN": 0.34,
    "DATE_TIME": 0.05,
}

_DISCLOSURE_PATTERNS = (
    re.compile(r"\bmy\s+(?:name|email|phone|address|salary|ssn|passport)\s+is\b", re.IGNORECASE),
    re.compile(r"\bhere\s+is\s+my\b", re.IGNORECASE),
    re.compile(r"\bcontact\s+me\s+at\b", re.IGNORECASE),
    re.compile(r"\byou\s+can\s+reach\s+me\s+at\b", re.IGNORECASE),
    re.compile(r"\bje\s+m[' ]appelle\b", re.IGNORECASE),
    re.compile(r"\bmon\s+(?:email|mail|telephone|numero|adresse|salaire)\s+est\b", re.IGNORECASE),
)

_SECRET_TOKEN_PATTERN = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*\d)[A-Za-z0-9_\-]{20,}\b")

_COMPONENT_CAPS: dict[str, float] = {
    "regex": 0.55,
    "keyword": 0.18,
    "presidio": 0.25,
    "heuristic": 0.17,
    "banword": 0.50,
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RegexPatternConfig(BaseModel):
    """A single regex-based detection rule."""

    name: str
    pattern: str
    weight: float = Field(ge=0.0, le=1.0)
    label: str


class KeywordGroupConfig(BaseModel):
    """A group of keywords that contribute a shared weight when any is present."""

    weight: float = Field(ge=0.0, le=1.0)
    terms: list[str]


class SensitivityAssessment(BaseModel):
    """Structured result from a sensitivity scorer."""

    model_config = ConfigDict(str_strip_whitespace=False)

    is_sensitive: bool
    score: float = Field(ge=0.0, le=1.0)
    level: Literal["low", "medium", "high", "critical"]
    detected_entities: list[DetectedEntity] = Field(default_factory=list)
    summary: str


class DefaultScorerConfig(BaseModel):
    """Configuration for :class:`DefaultSensitivityScorer`.

    All fields have sensible defaults so the scorer works out-of-the-box
    without any YAML configuration.
    """

    detector: PresidioDetectorConfig = Field(default_factory=PresidioDetectorConfig)
    sensitivity_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Score above which a message is considered sensitive",
    )
    entity_weights: dict[str, float] = Field(
        default_factory=lambda: dict(_DEFAULT_ENTITY_WEIGHTS),
        description="Per-entity-type weight for Presidio detections",
    )
    keyword_groups: dict[str, KeywordGroupConfig] = Field(
        default_factory=lambda: {k: KeywordGroupConfig(**v) for k, v in _DEFAULT_KEYWORD_GROUPS.items()},
        description="Keyword detection groups",
    )
    regex_patterns: list[RegexPatternConfig] = Field(
        default_factory=lambda: [RegexPatternConfig(**p) for p in _DEFAULT_REGEX_PATTERNS],
        description="Regex-based detection patterns",
    )
    banwords: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_BANWORDS),
        description="Critical phrases that always indicate high sensitivity",
    )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class SensitivityScorer:
    """Protocol for sensitivity scorers.

    Implementations must expose a single ``assess`` method.  This class is
    also the protocol type used for type annotations.
    """

    def assess(self, text: str) -> SensitivityAssessment:
        """Assess the sensitivity of *text*.

        Args:
            text: Input text to evaluate.

        Returns:
            A :class:`SensitivityAssessment` describing the sensitivity level.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Default implementation
# ---------------------------------------------------------------------------


class DefaultSensitivityScorer(SensitivityScorer):
    """Hybrid sensitivity scorer using regex, keywords, banwords, and Presidio.

    Args:
        config: Scorer configuration.  Uses default values when omitted.
    """

    def __init__(self, config: DefaultScorerConfig | None = None) -> None:
        self._config = config or DefaultScorerConfig()
        self._detector = PresidioDetector(config=self._config.detector)
        self._compiled_regex = self._compile_regex()
        self._compiled_keywords = self._compile_keywords()
        self._compiled_banwords = self._compile_banwords()

    def assess(self, text: str) -> SensitivityAssessment:
        """Return a sensitivity assessment for the given text."""
        if not text or not text.strip():
            return SensitivityAssessment(
                is_sensitive=False,
                score=0.0,
                level="low",
                summary="No content to analyze.",
            )

        signals: dict[str, float] = defaultdict(float)

        # Regex hits
        regex_weight = 0.0
        for pattern, weight, _label in self._compiled_regex:
            if pattern.search(text):
                regex_weight += weight
        signals["regex"] = min(regex_weight, _COMPONENT_CAPS["regex"])

        # Keyword hits
        kw_weight = 0.0
        for pattern, weight in self._compiled_keywords:
            if pattern.search(text):
                kw_weight += weight
        signals["keyword"] = min(kw_weight, _COMPONENT_CAPS["keyword"])

        # Banword hits (very high penalty)
        bw_weight = 0.0
        for pattern in self._compiled_banwords:
            if pattern.search(text):
                bw_weight += 0.50
        signals["banword"] = min(bw_weight, _COMPONENT_CAPS["banword"])

        # Presidio entities
        presidio_entities = self._detector.detect(text)
        presidio_weight = sum(self._config.entity_weights.get(e.entity_type, 0.05) * e.score for e in presidio_entities)
        signals["presidio"] = min(presidio_weight, _COMPONENT_CAPS["presidio"])

        # Heuristics
        heuristic_weight = 0.0
        if any(p.search(text) for p in _DISCLOSURE_PATTERNS):
            heuristic_weight += 0.10
        digit_count = sum(c.isdigit() for c in text)
        if digit_count >= 8 and digit_count / max(len(text), 1) >= 0.18:
            heuristic_weight += 0.08
        token_matches = list(_SECRET_TOKEN_PATTERN.finditer(text))
        if token_matches:
            heuristic_weight += 0.12 * min(len(token_matches), 2)
        signals["heuristic"] = min(heuristic_weight, _COMPONENT_CAPS["heuristic"])

        # Diversity boost
        active_categories = sum(1 for v in signals.values() if v > 0.0)
        diversity_boost = min(0.15, max(0, active_categories - 1) * 0.025)

        score = round(min(1.0, sum(signals.values()) + diversity_boost), 3)
        level = self._level(score)

        return SensitivityAssessment(
            is_sensitive=score >= self._config.sensitivity_threshold,
            score=score,
            level=level,
            detected_entities=presidio_entities,
            summary=self._summary(score, signals),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compile_regex(self) -> list[tuple[re.Pattern[str], float, str]]:
        result = []
        for cfg in self._config.regex_patterns:
            try:
                result.append((re.compile(cfg.pattern, re.IGNORECASE | re.DOTALL), cfg.weight, cfg.label))
            except re.error:
                pass
        return result

    def _compile_keywords(self) -> list[tuple[re.Pattern[str], float]]:
        result = []
        for group in self._config.keyword_groups.values():
            escaped = [re.escape(term) for term in group.terms]
            pattern_str = r"\b(?:" + "|".join(escaped) + r")\b"
            try:
                result.append((re.compile(pattern_str, re.IGNORECASE), group.weight))
            except re.error:
                pass
        return result

    def _compile_banwords(self) -> list[re.Pattern[str]]:
        result = []
        for word in self._config.banwords:
            escaped = re.escape(word)
            try:
                result.append(re.compile(escaped, re.IGNORECASE))
            except re.error:
                pass
        return result

    @staticmethod
    def _level(score: float) -> Literal["low", "medium", "high", "critical"]:
        if score < 0.25:
            return "low"
        if score < 0.50:
            return "medium"
        if score < 0.75:
            return "high"
        return "critical"

    @staticmethod
    def _summary(score: float, signals: dict[str, float]) -> str:
        active = [k for k, v in signals.items() if v > 0.0]
        if not active:
            return "No strong sensitivity signals detected."
        return f"Score {score:.2f} — active detectors: {', '.join(sorted(active))}."
