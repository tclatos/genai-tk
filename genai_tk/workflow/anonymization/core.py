"""Stateless PII anonymization core shared by Prefect flows and LangChain middleware.

This module is the single source of truth for anonymization logic.  It has no
dependency on Prefect or LangChain, so it can be imported from either context
without pulling in the other's runtime.

Two consumers:

- **ETL / batch** — :mod:`genai_tk.workflow.prefect.flows.anonymize_flow` calls
  :func:`anonymize_text` inside a Prefect task to scrub files before they enter
  the vector store.

- **Agent runtime** — :class:`~genai_tk.agents.langchain.middleware.anonymization_middleware.AnonymizationMiddleware`
  calls :func:`anonymize_text` on every human message before forwarding it to the LLM.

The :class:`AnonymizationConfig` Pydantic model is shared too, so the same configuration
object can drive both use cases.

Example — standalone (stateless) usage::

    ```python
    from faker import Faker
    from genai_tk.workflow.anonymization.core import AnonymizationConfig, anonymize_text
    from genai_tk.agents.langchain.middleware.presidio_detector import PresidioDetector

    config = AnonymizationConfig()
    detector = PresidioDetector(config=config.detector)
    Faker.seed(config.faker_seed or 0)
    faker = Faker(config.faker_locales)

    anonymized, mapping = anonymize_text(
        "Contact Alice at alice@example.com",
        detector=detector,
        faker=faker,
    )
    ```
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from genai_tk.workflow.anonymization.presidio_detector import (
    DetectedEntity,
    PresidioDetector,
    PresidioDetectorConfig,
)


class AnonymizationConfig(BaseModel):
    """Configuration for PII anonymization.

    Shared between :class:`~genai_tk.agents.langchain.middleware.anonymization_middleware.AnonymizationMiddleware`
    and :func:`~genai_tk.workflow.prefect.flows.anonymize_flow.anonymize_files_flow`.

    Can be passed directly or expanded as keyword arguments when instantiated via the YAML
    middleware system.
    """

    detector: PresidioDetectorConfig = Field(default_factory=PresidioDetectorConfig)
    faker_seed: int | None = Field(default=None, description="Seed for deterministic Faker output")
    faker_locales: list[str] = Field(default_factory=lambda: ["en_US"], description="Faker locales")
    fuzzy_deanonymize: bool = Field(default=True, description="Use fuzzy matching when deanonymizing LLM output")
    fuzzy_threshold: int = Field(default=85, ge=0, le=100, description="rapidfuzz ratio threshold (0-100)")


def make_fake_value(entity_type: str, faker: Any) -> str:
    """Generate a Faker replacement for a given Presidio entity type.

    Args:
        entity_type: Presidio entity type string (e.g. ``"PERSON"``, ``"EMAIL_ADDRESS"``).
        faker: A ``Faker`` instance used to generate replacement values.

    Returns:
        A fake string suitable as a PII replacement.
    """
    generators: dict[str, Any] = {
        "PERSON": faker.name,
        "EMAIL_ADDRESS": faker.email,
        "PHONE_NUMBER": faker.phone_number,
        "CREDIT_CARD": faker.credit_card_number,
        "LOCATION": faker.city,
        "IBAN_CODE": faker.iban,
        "US_SSN": faker.ssn,
        "IP_ADDRESS": faker.ipv4,
        "URL": faker.url,
        "DATE_TIME": lambda: faker.date(),
        "NRP": faker.name,
        "ORG": lambda: faker.company(),
        # Domain-specific business entities — use faker.company() / faker.bs() as
        # placeholders; callers can override make_fake_value() for richer replacements.
        "COMPANY": faker.company,
        "PRODUCT": faker.bs,
        "PROJECT": faker.bs,
    }
    gen = generators.get(entity_type)
    if gen:
        return gen()
    return faker.bothify(text=f"{entity_type[:4].upper()}####")


def anonymize_text(
    text: str,
    *,
    detector: PresidioDetector,
    faker: Any,
    mapping: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Detect and replace PII in *text*, returning the sanitized string and the replacement mapping.

    The mapping can be shared across multiple calls (e.g. across all chunks of a document) to
    ensure the same entity always gets the same fake replacement within a set.

    Args:
        text: Raw input text to anonymize.
        detector: Pre-configured :class:`~genai_tk.agents.langchain.middleware.presidio_detector.PresidioDetector`.
        faker: ``Faker`` instance for generating replacement values.
        mapping: Existing original→fake mapping to extend.  ``None`` starts a fresh mapping.

    Returns:
        Tuple of ``(anonymized_text, mapping)`` where *mapping* maps original PII strings to their
        fake replacements.
    """
    if mapping is None:
        mapping = {}

    entities = detector.detect(text)
    if not entities:
        return text, mapping

    entities = _deduplicate_entities(entities)
    result = text
    offset = 0

    for entity in entities:
        original = text[entity.start : entity.end]
        if original not in mapping:
            mapping[original] = make_fake_value(entity.entity_type, faker)
        fake = mapping[original]

        start = entity.start + offset
        end = entity.end + offset
        result = result[:start] + fake + result[end:]
        offset += len(fake) - len(original)

    logger.debug("Anonymized {} entities", len(entities))
    return result, mapping


def _deduplicate_entities(entities: list[DetectedEntity]) -> list[DetectedEntity]:
    """Remove overlapping Presidio detections, keeping the highest-confidence entity per span."""
    if not entities:
        return entities
    sorted_entities = sorted(entities, key=lambda e: (-e.score, e.start))
    kept: list[DetectedEntity] = []
    for entity in sorted_entities:
        overlaps = any(entity.start < kept_entity.end and entity.end > kept_entity.start for kept_entity in kept)
        if not overlaps:
            kept.append(entity)
    return sorted(kept, key=lambda e: e.start)
