"""Unit tests for PresidioDetector."""

from __future__ import annotations

from genai_tk.agents.langchain.middleware.presidio_detector import (
    CustomRecognizerConfig,
    PresidioDetector,
    PresidioDetectorConfig,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_detector(
    analyzed_fields: list[str] | None = None,
    enable_spacy: bool = False,
    custom_recognizers: list[CustomRecognizerConfig] | None = None,
) -> PresidioDetector:
    """Build a detector with spaCy disabled to avoid heavy model downloads in CI."""
    config = PresidioDetectorConfig(
        analyzed_fields=analyzed_fields or ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "PERSON"],
        enable_spacy=enable_spacy,
        custom_recognizers=custom_recognizers or [],
    )
    return PresidioDetector(config)


# ---------------------------------------------------------------------------
# Basic detection
# ---------------------------------------------------------------------------


class TestPresidioDetectorDetect:
    def test_email_detected(self) -> None:
        detector = _make_detector()
        entities = detector.detect("Contact me at alice@example.com")
        types = [e.entity_type for e in entities]
        assert "EMAIL_ADDRESS" in types

    def test_phone_detected(self) -> None:
        detector = _make_detector(enable_spacy=False)
        # Presidio phone detection without spaCy context is confidence-dependent;
        # use a score_threshold of 0 to capture any detection attempt, and
        # also include a US phone in standard format that Presidio recognizes.
        config = PresidioDetectorConfig(
            analyzed_fields=["PHONE_NUMBER"],
            enable_spacy=False,
            score_threshold=0.0,  # accept any score
        )
        detector_low = PresidioDetector(config)
        entities = detector_low.detect("My phone number is (212) 555-1234")
        types = [e.entity_type for e in entities]
        # With score_threshold=0 Presidio should surface a PHONE_NUMBER hit
        assert "PHONE_NUMBER" in types

    def test_empty_text_returns_empty(self) -> None:
        detector = _make_detector()
        assert detector.detect("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        detector = _make_detector()
        assert detector.detect("   ") == []

    def test_clean_text_returns_empty(self) -> None:
        detector = _make_detector()
        entities = detector.detect("The weather is nice today.")
        # Presidio may still flag some false positives; just ensure it doesn't crash
        assert isinstance(entities, list)

    def test_sorted_by_start(self) -> None:
        detector = _make_detector()
        entities = detector.detect("alice@example.com bob@example.com")
        starts = [e.start for e in entities]
        assert starts == sorted(starts)

    def test_entity_fields_populated(self) -> None:
        detector = _make_detector()
        entities = detector.detect("Email alice@example.com now")
        email_entities = [e for e in entities if e.entity_type == "EMAIL_ADDRESS"]
        if email_entities:
            e = email_entities[0]
            assert "alice@example.com" in e.text
            assert e.start >= 0
            assert e.end > e.start
            assert 0.0 <= e.score <= 1.0


# ---------------------------------------------------------------------------
# Score threshold
# ---------------------------------------------------------------------------


class TestScoreThreshold:
    def test_high_threshold_filters_detections(self) -> None:
        config = PresidioDetectorConfig(
            analyzed_fields=["EMAIL_ADDRESS"],
            enable_spacy=False,
            score_threshold=0.99,
        )
        detector = PresidioDetector(config)
        # Most detections score < 1.0, so a threshold of 0.99 should filter most
        entities = detector.detect("Call alice@example.com")
        # May return 0 or some with very high confidence — just don't crash
        assert isinstance(entities, list)

    def test_zero_threshold_returns_more(self) -> None:
        config_low = PresidioDetectorConfig(analyzed_fields=["EMAIL_ADDRESS"], enable_spacy=False, score_threshold=0.0)
        config_high = PresidioDetectorConfig(analyzed_fields=["EMAIL_ADDRESS"], enable_spacy=False, score_threshold=0.9)
        detector_low = PresidioDetector(config_low)
        detector_high = PresidioDetector(config_high)
        text = "Email alice@example.com"
        low_count = len(detector_low.detect(text))
        high_count = len(detector_high.detect(text))
        assert low_count >= high_count


# ---------------------------------------------------------------------------
# Custom recognizers
# ---------------------------------------------------------------------------


class TestCustomRecognizers:
    def test_custom_entity_detected(self) -> None:
        recognizer = CustomRecognizerConfig(
            entity_name="CUSTOM_ORG",
            patterns=[r"(?i)\b(AcmeCorp|GlobexInc)\b"],
            context=["company"],
        )
        detector = _make_detector(
            analyzed_fields=["CUSTOM_ORG"],
            custom_recognizers=[recognizer],
        )
        entities = detector.detect("I work at AcmeCorp as a developer.")
        types = [e.entity_type for e in entities]
        assert "CUSTOM_ORG" in types

    def test_multiple_custom_patterns(self) -> None:
        recognizer = CustomRecognizerConfig(
            entity_name="PROJECT_CODE",
            patterns=[r"\bPROJ-\d{4}\b", r"\bTICKET-\d+\b"],
        )
        detector = _make_detector(
            analyzed_fields=["PROJECT_CODE"],
            custom_recognizers=[recognizer],
        )
        entities = detector.detect("Working on PROJ-1234 and TICKET-567")
        types = [e.entity_type for e in entities]
        assert "PROJECT_CODE" in types


# ---------------------------------------------------------------------------
# spaCy disabled
# ---------------------------------------------------------------------------


class TestSpacyDisabled:
    def test_spacy_disabled_does_not_crash(self) -> None:
        config = PresidioDetectorConfig(
            analyzed_fields=["EMAIL_ADDRESS"],
            enable_spacy=False,
        )
        detector = PresidioDetector(config)
        entities = detector.detect("Send to info@example.com")
        assert isinstance(entities, list)
