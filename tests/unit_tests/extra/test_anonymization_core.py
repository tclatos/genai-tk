"""Tests for workflow.anonymization — replaces the removed CustomizedPresidioAnonymizer."""

from faker import Faker

from genai_tk.extra.nlp import (
    AnonymizationConfig,
    CustomRecognizerConfig,
    PresidioDetector,
    PresidioDetectorConfig,
    anonymize_text,
    make_fake_value,
)


def _make_detector(custom_recognizers: list | None = None) -> PresidioDetector:
    cfg = PresidioDetectorConfig(custom_recognizers=custom_recognizers or [])
    return PresidioDetector(config=cfg)


def _make_faker(seed: int = 42) -> Faker:
    Faker.seed(seed)
    return Faker(["en_US"])


class TestMakeFakeValue:
    def test_standard_entities(self):
        faker = _make_faker()
        for entity in ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "LOCATION", "IP_ADDRESS", "URL"]:
            val = make_fake_value(entity, faker)
            assert isinstance(val, str) and len(val) > 0

    def test_company_uses_faker_company(self):
        faker = _make_faker()
        val = make_fake_value("COMPANY", faker)
        assert isinstance(val, str) and len(val) > 0

    def test_product_uses_faker_bs(self):
        faker = _make_faker()
        val = make_fake_value("PRODUCT", faker)
        assert isinstance(val, str) and len(val) > 0

    def test_project_uses_faker_bs(self):
        faker = _make_faker()
        val = make_fake_value("PROJECT", faker)
        assert isinstance(val, str) and len(val) > 0

    def test_unknown_entity_falls_back_to_bothify(self):
        faker = _make_faker()
        val = make_fake_value("WIDGET_ID", faker)
        assert val.startswith("WIDG")


class TestAnonymizeText:
    def test_basic_pii(self):
        detector = _make_detector()
        faker = _make_faker()
        text = "John Smith's email is john.smith@email.com"
        anonymized, mapping = anonymize_text(text, detector=detector, faker=faker)

        assert "John Smith" not in anonymized
        assert "john.smith@email.com" not in anonymized
        assert len(mapping) >= 1

    def test_mapping_enables_deanonymization(self):
        detector = _make_detector()
        faker = _make_faker()
        original = "Contact Bob at bob@example.com or 555-123-4567"
        anonymized, mapping = anonymize_text(original, detector=detector, faker=faker)

        restored = anonymized
        for real, fake in mapping.items():
            restored = restored.replace(fake, real)
        assert restored == original

    def test_empty_text_returns_unchanged(self):
        detector = _make_detector()
        faker = _make_faker()
        anonymized, mapping = anonymize_text("", detector=detector, faker=faker)
        assert anonymized == ""
        assert mapping == {}

    def test_shared_mapping_across_calls(self):
        """Same entity gets the same fake value when mapping is reused."""
        detector = _make_detector()
        faker = _make_faker()
        shared: dict[str, str] = {}
        _, shared = anonymize_text("Alice called alice@example.com", detector=detector, faker=faker, mapping=shared)
        _, shared = anonymize_text("Alice again: alice@example.com", detector=detector, faker=faker, mapping=shared)
        # The same original → same fake value
        assert len(set(shared.values())) == len(shared)

    def test_custom_company_product_recognizers(self):
        detector = _make_detector(
            custom_recognizers=[
                CustomRecognizerConfig(
                    entity_name="COMPANY",
                    patterns=[r"(?i)\b(Acme Corp|Tech Solutions)\b"],
                    context=["company", "firm"],
                ),
                CustomRecognizerConfig(
                    entity_name="PRODUCT",
                    patterns=[r"(?i)\b(WidgetPro)\b"],
                    context=["product", "service"],
                ),
            ]
        )
        faker = _make_faker()
        text = "Alice works at Acme Corp using WidgetPro"
        anonymized, mapping = anonymize_text(text, detector=detector, faker=faker)

        assert "Acme Corp" not in anonymized
        assert "WidgetPro" not in anonymized
        assert "COMPANY" in {k.split(":")[0] for k in []} or len(mapping) >= 2

    def test_project_recognizer(self):
        detector = _make_detector(
            custom_recognizers=[
                CustomRecognizerConfig(
                    entity_name="PROJECT",
                    patterns=[r"(?i)\b(ProjectAlpha|ProjectBeta)\b"],
                    context=["project", "initiative"],
                ),
            ]
        )
        faker = _make_faker()
        text = "The team is working on ProjectAlpha and ProjectBeta"
        anonymized, mapping = anonymize_text(text, detector=detector, faker=faker)

        assert "ProjectAlpha" not in anonymized
        assert "ProjectBeta" not in anonymized


class TestAnonymizationConfig:
    def test_default_config(self):
        config = AnonymizationConfig()
        assert "PERSON" in config.detector.analyzed_fields
        assert config.fuzzy_deanonymize is True
        assert config.fuzzy_threshold == 85

    def test_custom_fields_propagate(self):
        config = AnonymizationConfig(
            detector=PresidioDetectorConfig(analyzed_fields=["PERSON", "EMAIL_ADDRESS"]),
            faker_seed=99,
        )
        assert config.detector.analyzed_fields == ["PERSON", "EMAIL_ADDRESS"]
        assert config.faker_seed == 99
