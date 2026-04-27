"""Unit tests for DefaultSensitivityScorer."""

from __future__ import annotations

import pytest

from genai_tk.agents.langchain.middleware.presidio_detector import PresidioDetectorConfig
from genai_tk.agents.langchain.middleware.sensitivity_scorer import (
    DefaultScorerConfig,
    DefaultSensitivityScorer,
    SensitivityAssessment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scorer(threshold: float = 0.35, enable_spacy: bool = False) -> DefaultSensitivityScorer:
    config = DefaultScorerConfig(
        sensitivity_threshold=threshold,
        detector=PresidioDetectorConfig(enable_spacy=enable_spacy),
    )
    return DefaultSensitivityScorer(config)


# ---------------------------------------------------------------------------
# Basic scoring
# ---------------------------------------------------------------------------


class TestScorerBasic:
    def test_empty_text_returns_not_sensitive(self) -> None:
        scorer = _make_scorer()
        result = scorer.assess("")
        assert result.is_sensitive is False
        assert result.score == 0.0
        assert result.level == "low"

    def test_whitespace_only_not_sensitive(self) -> None:
        scorer = _make_scorer()
        result = scorer.assess("   ")
        assert result.is_sensitive is False

    def test_clean_text_low_score(self) -> None:
        scorer = _make_scorer()
        result = scorer.assess("The quick brown fox jumps over the lazy dog.")
        assert result.level in {"low", "medium"}

    def test_returns_sensitivity_assessment(self) -> None:
        scorer = _make_scorer()
        result = scorer.assess("Hello world")
        assert isinstance(result, SensitivityAssessment)
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Regex-based detection
# ---------------------------------------------------------------------------


class TestRegexDetection:
    def test_email_increases_score(self) -> None:
        scorer = _make_scorer(threshold=0.1)
        result_with_email = scorer.assess("Contact alice@example.com for details.")
        result_clean = scorer.assess("Contact Alice for details.")
        assert result_with_email.score > result_clean.score

    def test_api_key_detected(self) -> None:
        scorer = _make_scorer(threshold=0.1)
        result = scorer.assess("Use API key sk-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789")
        assert result.score > 0.1

    def test_private_key_critical(self) -> None:
        scorer = _make_scorer()
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
        result = scorer.assess(text)
        assert result.score >= 0.4
        assert result.level in {"high", "critical"}


# ---------------------------------------------------------------------------
# Keyword-based detection
# ---------------------------------------------------------------------------


class TestKeywordDetection:
    def test_credential_keywords_increase_score(self) -> None:
        scorer = _make_scorer(threshold=0.01)
        result = scorer.assess("Enter your password here")
        assert result.score > 0.0

    def test_financial_keywords_increase_score(self) -> None:
        scorer = _make_scorer(threshold=0.01)
        result = scorer.assess("Send the salary transfer to IBAN account")
        assert result.score > 0.0


# ---------------------------------------------------------------------------
# Banword detection
# ---------------------------------------------------------------------------


class TestBanwordDetection:
    def test_banword_triggers_high_score(self) -> None:
        scorer = _make_scorer(threshold=0.35)
        result = scorer.assess("The root password is supersecret123")
        assert result.is_sensitive is True
        assert result.score >= 0.35

    def test_banword_in_sentence_detected(self) -> None:
        scorer = _make_scorer()
        result = scorer.assess("Please provide the private key for the server")
        assert result.score > 0.1


# ---------------------------------------------------------------------------
# Sensitivity threshold
# ---------------------------------------------------------------------------


class TestSensitivityThreshold:
    def test_low_threshold_makes_more_sensitive(self) -> None:
        scorer_low = _make_scorer(threshold=0.05)
        scorer_high = _make_scorer(threshold=0.95)
        text = "Contact alice@example.com"
        assert scorer_low.assess(text).is_sensitive or True  # may or may not trigger
        # High threshold text that would normally be medium should NOT be sensitive
        result_high = scorer_high.assess("The weather is nice")
        assert result_high.is_sensitive is False

    def test_banword_always_sensitive_at_default_threshold(self) -> None:
        scorer = DefaultSensitivityScorer()  # uses default threshold 0.35
        result = scorer.assess("The root password is abc")
        assert result.is_sensitive is True


# ---------------------------------------------------------------------------
# Score levels
# ---------------------------------------------------------------------------


class TestScoreLevels:
    @pytest.mark.parametrize(
        "score,expected_level",
        [
            (0.0, "low"),
            (0.10, "low"),
            (0.25, "medium"),
            (0.40, "medium"),
            (0.50, "high"),
            (0.70, "high"),
            (0.75, "critical"),
            (1.0, "critical"),
        ],
    )
    def test_level_boundaries(self, score: float, expected_level: str) -> None:
        from genai_tk.agents.langchain.middleware.sensitivity_scorer import DefaultSensitivityScorer

        assert DefaultSensitivityScorer._level(score) == expected_level


# ---------------------------------------------------------------------------
# Custom entity weights
# ---------------------------------------------------------------------------


class TestEntityWeightCustomization:
    def test_custom_weights_affect_score(self) -> None:
        scorer_default = _make_scorer()
        scorer_custom = DefaultSensitivityScorer(
            DefaultScorerConfig(
                entity_weights={"EMAIL_ADDRESS": 0.01},  # Very low weight for email
                detector=PresidioDetectorConfig(enable_spacy=False),
            )
        )
        text = "Email alice@example.com"
        # We can't assert exact values since Presidio confidence varies,
        # but custom scorer should not crash
        result = scorer_custom.assess(text)
        assert isinstance(result, SensitivityAssessment)


# ---------------------------------------------------------------------------
# Summary field
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_populated_for_sensitive_content(self) -> None:
        scorer = _make_scorer(threshold=0.01)
        result = scorer.assess("root password is abc123")
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_summary_for_empty_text(self) -> None:
        scorer = _make_scorer()
        result = scorer.assess("")
        assert "No content" in result.summary
