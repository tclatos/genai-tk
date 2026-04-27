"""Anonymization middleware for LangChain agents.

Detects PII in human messages before they reach the LLM, replaces each entity
with a deterministic fake value (via Faker), and deanonymizes the LLM response
before it is returned to the caller.

The PII→fake mapping is scoped per ``thread_id`` so concurrent conversations
are isolated from each other.

Streaming is **not** supported: the ``after_model`` hook only runs on complete
``AIMessage`` objects.  When streaming is used, deanonymization is silently
skipped and the anonymized text will appear in the stream.

Example YAML config::

    middlewares:
      - class: genai_tk.agents.langchain.middleware.anonymization_middleware:AnonymizationMiddleware
        analyzed_fields: ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"]
        fuzzy_deanonymize: true

Example programmatic usage::

    ```python
    from genai_tk.agents.langchain.middleware.anonymization_middleware import (
        AnonymizationConfig,
        AnonymizationMiddleware,
    )
    from genai_tk.agents.langchain.middleware.presidio_detector import PresidioDetectorConfig

    config = AnonymizationConfig(
        detector=PresidioDetectorConfig(analyzed_fields=["PERSON", "EMAIL_ADDRESS"]),
        faker_seed=42,
    )
    middleware = AnonymizationMiddleware(config=config)
    ```
"""

from __future__ import annotations

import string
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.runtime import Runtime
from loguru import logger
from pydantic import BaseModel, Field

from genai_tk.agents.langchain.middleware.presidio_detector import (
    DetectedEntity,
    PresidioDetector,
    PresidioDetectorConfig,
)

_DEFAULT_THREAD = "__default__"


class AnonymizationConfig(BaseModel):
    """Configuration for :class:`AnonymizationMiddleware`.

    Can be passed directly to the constructor or expanded as keyword arguments
    when instantiated via the YAML middleware system.
    """

    detector: PresidioDetectorConfig = Field(default_factory=PresidioDetectorConfig)
    faker_seed: int | None = Field(default=None, description="Seed for deterministic Faker output")
    faker_locales: list[str] = Field(default_factory=lambda: ["en_US"], description="Faker locales")
    fuzzy_deanonymize: bool = Field(default=True, description="Use fuzzy matching when deanonymizing LLM output")
    fuzzy_threshold: int = Field(default=85, ge=0, le=100, description="rapidfuzz ratio threshold (0-100)")


class AnonymizationMiddleware(AgentMiddleware):
    """Reversible PII anonymization middleware for LangChain agents.

    Detects PII in the last human message using Presidio, replaces each entity
    with a Faker-generated fake value, and restores original values in the
    AI response via deanonymization.

    The mapping is keyed by ``thread_id`` (from ``runtime.config``), so
    parallel conversations remain isolated.

    Args:
        config: Full anonymization config.  Alternatively, pass keyword
            arguments matching :class:`AnonymizationConfig` fields directly
            (used when instantiated from YAML).
    """

    def __init__(self, config: AnonymizationConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            # Allow YAML-style flat kwargs: analyzed_fields=..., faker_seed=..., etc.
            detector_fields = {k: v for k, v in kwargs.items() if k in PresidioDetectorConfig.model_fields}
            anon_fields = {k: v for k, v in kwargs.items() if k in AnonymizationConfig.model_fields and k != "detector"}
            if detector_fields:
                anon_fields["detector"] = PresidioDetectorConfig(**detector_fields)
            config = AnonymizationConfig(**anon_fields)
        self._config = config
        self._detector = PresidioDetector(config=config.detector)
        self._faker = self._build_faker()
        # mapping[thread_id][original_text] = fake_value
        self._mapping: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Middleware hooks
    # ------------------------------------------------------------------

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # type: ignore[override]
        """Anonymize PII in the last human message before the LLM is called."""
        messages: list[AnyMessage] = state.get("messages", [])
        if not messages:
            return None

        thread_id = self._thread_id(runtime)

        # Find and anonymize the last human message
        updated: list[AnyMessage] = list(messages)
        changed = False
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, HumanMessage):
                new_msg, did_change = self._anonymize_message(msg, thread_id)
                if did_change:
                    updated[i] = new_msg
                    changed = True
                break  # only last human message

        if not changed:
            return None
        return {"messages": updated}

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # type: ignore[override]
        """Deanonymize PII in the last AI message after the LLM has responded."""
        messages: list[AnyMessage] = state.get("messages", [])
        if not messages:
            return None

        thread_id = self._thread_id(runtime)
        mapping = self._mapping.get(thread_id)
        if not mapping:
            return None  # Nothing was anonymized for this thread

        # Find and deanonymize the last AI message
        updated: list[AnyMessage] = list(messages)
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, AIMessage):
                new_msg, did_change = self._deanonymize_message(msg, thread_id)
                if did_change:
                    updated[i] = new_msg
                    return {"messages": updated}
                return None
        return None

    # ------------------------------------------------------------------
    # Public helper
    # ------------------------------------------------------------------

    def cleanup(self, thread_id: str) -> None:
        """Remove the anonymization mapping for a completed conversation.

        Args:
            thread_id: The thread identifier to clean up.
        """
        self._mapping.pop(thread_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _thread_id(self, runtime: Any) -> str:
        try:
            from langgraph.config import get_config  # noqa: PLC0415

            cfg = get_config()
            return cfg.get("configurable", {}).get("thread_id", _DEFAULT_THREAD)
        except Exception:
            pass
        try:
            cfg = runtime.config if runtime is not None else {}
            return cfg.get("configurable", {}).get("thread_id", _DEFAULT_THREAD)
        except Exception:
            return _DEFAULT_THREAD

    def _anonymize_message(self, msg: HumanMessage, thread_id: str) -> tuple[HumanMessage, bool]:
        content = msg.content
        if isinstance(content, str):
            new_text = self._anonymize_text(content, thread_id)
            if new_text == content:
                return msg, False
            return HumanMessage(content=new_text, id=msg.id, name=msg.name), True

        if isinstance(content, list):
            new_parts: list[Any] = []
            changed = False
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    new_text = self._anonymize_text(part["text"], thread_id)
                    if new_text != part["text"]:
                        new_parts.append({**part, "text": new_text})
                        changed = True
                        continue
                new_parts.append(part)
            if changed:
                return HumanMessage(content=new_parts, id=msg.id, name=msg.name), True

        return msg, False

    def _deanonymize_message(self, msg: AIMessage, thread_id: str) -> tuple[AIMessage, bool]:
        content = msg.content
        if isinstance(content, str):
            new_text = self._deanonymize_text(content, thread_id)
            if new_text == content:
                return msg, False
            return msg.model_copy(update={"content": new_text}), True

        if isinstance(content, list):
            new_parts: list[Any] = []
            changed = False
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    new_text = self._deanonymize_text(part["text"], thread_id)
                    if new_text != part["text"]:
                        new_parts.append({**part, "text": new_text})
                        changed = True
                        continue
                new_parts.append(part)
            if changed:
                return msg.model_copy(update={"content": new_parts}), True

        return msg, False

    def _anonymize_text(self, text: str, thread_id: str) -> str:
        """Detect PII in *text*, replace with Faker values, store mapping."""
        entities = self._detector.detect(text)
        if not entities:
            return text

        # Deduplicate overlapping spans — keep highest-score entity per span
        entities = _deduplicate_entities(entities)

        mapping = self._mapping.setdefault(thread_id, {})
        result = text
        offset = 0

        for entity in entities:
            original = text[entity.start : entity.end]
            if original not in mapping:
                mapping[original] = self._fake_value(entity.entity_type)
            fake = mapping[original]

            start = entity.start + offset
            end = entity.end + offset
            result = result[:start] + fake + result[end:]
            offset += len(fake) - len(original)

        logger.debug(f"[Anonymization] Anonymized {len(entities)} entities for thread '{thread_id}'")
        return result

    def _deanonymize_text(self, text: str, thread_id: str) -> str:
        """Replace Faker values in *text* with original PII using the stored mapping."""
        mapping = self._mapping.get(thread_id, {})
        if not mapping:
            return text

        reverse = {v: k for k, v in mapping.items()}
        result = text

        # Exact replacements first
        for fake, original in reverse.items():
            result = result.replace(fake, original)

        # Fuzzy matching for values not caught by exact match
        if self._config.fuzzy_deanonymize:
            result = self._fuzzy_deanonymize(result, reverse)

        return result

    def _fuzzy_deanonymize(self, text: str, reverse_mapping: dict[str, str]) -> str:
        """Try to restore PII values that were slightly altered by the LLM."""
        try:
            from rapidfuzz import fuzz
        except ImportError:
            return text

        words = text.split()
        for i, word in enumerate(words):
            clean = word.strip(string.punctuation)
            if not clean:
                continue
            for fake, original in reverse_mapping.items():
                if fake in text:
                    continue  # already replaced by exact pass
                if fuzz.ratio(clean, fake) >= self._config.fuzzy_threshold:
                    words[i] = word.replace(clean, original)
                    break
        return " ".join(words)

    def _fake_value(self, entity_type: str) -> str:
        """Generate a Faker replacement for a given entity type."""
        faker = self._faker
        mapping = {
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
            "NRP": faker.name,  # Nationality/Religion/Political group → use name
            "ORG": lambda: faker.company(),
        }
        generator = mapping.get(entity_type)
        if generator:
            return generator()
        # Generic fallback: TAG####
        return faker.bothify(text=f"{entity_type[:4].upper()}####")

    def _build_faker(self) -> Any:
        from faker import Faker

        if self._config.faker_seed is not None:
            Faker.seed(self._config.faker_seed)
        return Faker(locale=self._config.faker_locales)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------


def _deduplicate_entities(entities: list[DetectedEntity]) -> list[DetectedEntity]:
    """Remove overlapping entities, keeping highest-score entity per span."""
    if not entities:
        return entities
    sorted_entities = sorted(entities, key=lambda e: (-e.score, e.start))
    kept: list[DetectedEntity] = []
    for entity in sorted_entities:
        overlaps = False
        for kept_entity in kept:
            if entity.start < kept_entity.end and entity.end > kept_entity.start:
                overlaps = True
                break
        if not overlaps:
            kept.append(entity)
    return sorted(kept, key=lambda e: e.start)
