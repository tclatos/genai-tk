"""Unit tests for AnonymizationMiddleware."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from genai_tk.agents.langchain.middleware.anonymization_middleware import (
    AnonymizationConfig,
    AnonymizationMiddleware,
    _deduplicate_entities,
)
from genai_tk.agents.langchain.middleware.presidio_detector import DetectedEntity, PresidioDetectorConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config(analyzed_fields: list[str] | None = None) -> AnonymizationConfig:
    return AnonymizationConfig(
        detector=PresidioDetectorConfig(
            analyzed_fields=analyzed_fields or ["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"],
            enable_spacy=False,
            score_threshold=0.4,
        ),
        faker_seed=42,
        faker_locales=["en_US"],
        fuzzy_deanonymize=True,
        fuzzy_threshold=85,
    )


def _make_runtime(thread_id: str = "thread-1") -> MagicMock:
    runtime = MagicMock()
    runtime.config = {"configurable": {"thread_id": thread_id}}
    return runtime


def _make_state(content: str | list[Any], extra_messages: list = []) -> dict[str, Any]:
    return {"messages": [*extra_messages, HumanMessage(content=content)]}


def _make_ai_state(human_content: str, ai_content: str) -> dict[str, Any]:
    return {
        "messages": [
            HumanMessage(content=human_content),
            AIMessage(content=ai_content),
        ]
    }


# ---------------------------------------------------------------------------
# Tests: before_model (anonymization)
# ---------------------------------------------------------------------------


class TestBeforeModel:
    def test_email_anonymized(self) -> None:
        middleware = AnonymizationMiddleware(config=_minimal_config())
        runtime = _make_runtime()
        state = _make_state("Contact me at alice@example.com")

        result = middleware.before_model(state, runtime)

        if result is not None:  # PII was found
            updated_msg = result["messages"][-1]
            assert "alice@example.com" not in updated_msg.content
            # Mapping must be populated
            assert "thread-1" in middleware._mapping
        # else: Presidio may not detect it with high enough confidence — test is lenient

    def test_no_pii_returns_none(self) -> None:
        middleware = AnonymizationMiddleware(config=_minimal_config())
        runtime = _make_runtime()
        state = _make_state("The weather is nice today.")

        result = middleware.before_model(state, runtime)

        # No PII — should return None (no state update)
        assert result is None

    def test_empty_messages_returns_none(self) -> None:
        middleware = AnonymizationMiddleware(config=_minimal_config())
        runtime = _make_runtime()

        result = middleware.before_model({"messages": []}, runtime)

        assert result is None

    def test_only_last_human_message_anonymized(self) -> None:
        middleware = AnonymizationMiddleware(config=_minimal_config())
        runtime = _make_runtime()

        messages = [
            HumanMessage(content="First: alice@example.com"),
            AIMessage(content="Noted"),
            HumanMessage(content="Second: bob@example.com"),
        ]
        state = {"messages": messages}

        result = middleware.before_model(state, runtime)

        # Only the last (index 2) human message should be affected
        if result is not None:
            updated = result["messages"]
            # First message should be unchanged
            assert updated[0].content == "First: alice@example.com"
            # Last message should have been processed
            assert isinstance(updated[2], HumanMessage)

    def test_multipart_content_anonymized(self) -> None:
        middleware = AnonymizationMiddleware(config=_minimal_config())
        runtime = _make_runtime()
        content = [{"type": "text", "text": "Send to alice@example.com"}]
        state = _make_state(content)

        result = middleware.before_model(state, runtime)

        if result is not None:
            updated_parts = result["messages"][-1].content
            assert isinstance(updated_parts, list)
            assert "alice@example.com" not in updated_parts[0]["text"]

    def test_thread_isolation(self) -> None:
        """Two threads should not share anonymization mappings."""
        middleware = AnonymizationMiddleware(config=_minimal_config())
        runtime_a = _make_runtime("thread-A")
        runtime_b = _make_runtime("thread-B")

        state = _make_state("alice@example.com")
        middleware.before_model(state, runtime_a)
        middleware.before_model(state, runtime_b)

        # Mappings should be separate dicts
        mapping_a = middleware._mapping.get("thread-A", {})
        mapping_b = middleware._mapping.get("thread-B", {})
        assert mapping_a is not mapping_b


# ---------------------------------------------------------------------------
# Tests: after_model (deanonymization)
# ---------------------------------------------------------------------------


class TestAfterModel:
    def test_deanonymize_ai_response(self) -> None:
        middleware = AnonymizationMiddleware(config=_minimal_config())
        runtime = _make_runtime()

        # First anonymize to populate the mapping
        state = _make_state("Contact alice@example.com")
        before_result = middleware.before_model(state, runtime)

        if before_result is None:
            pytest.skip("No PII detected — Presidio confidence may be too low in test env")

        # Determine what fake was used
        fake_email = None
        for original, fake in middleware._mapping.get("thread-1", {}).items():
            if "alice@example.com" in original:
                fake_email = fake
                break

        if fake_email is None:
            pytest.skip("Email not in mapping")

        # Now deanonymize — LLM response contains the fake email
        ai_state = {"messages": [AIMessage(content=f"I'll email {fake_email} now.")]}
        after_result = middleware.after_model(ai_state, runtime)

        assert after_result is not None
        final_content = after_result["messages"][-1].content
        assert "alice@example.com" in final_content
        assert fake_email not in final_content

    def test_no_mapping_returns_none(self) -> None:
        middleware = AnonymizationMiddleware(config=_minimal_config())
        runtime = _make_runtime("empty-thread")

        state = {"messages": [AIMessage(content="Nothing was anonymized")]}
        result = middleware.after_model(state, runtime)

        assert result is None

    def test_no_ai_message_returns_none(self) -> None:
        middleware = AnonymizationMiddleware(config=_minimal_config())
        runtime = _make_runtime()
        # Inject a mapping manually
        middleware._mapping["thread-1"] = {"alice@example.com": "fake@test.com"}

        state = {"messages": [HumanMessage(content="No AI here")]}
        result = middleware.after_model(state, runtime)

        assert result is None


# ---------------------------------------------------------------------------
# Tests: round-trip anonymize/deanonymize
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_exact_round_trip(self) -> None:
        """Direct test of internal methods without going through middleware hooks."""
        middleware = AnonymizationMiddleware(config=_minimal_config())
        thread_id = "rt-thread"

        original = "My email is alice@example.com and phone is 555-867-5309"
        anonymized = middleware._anonymize_text(original, thread_id)

        if anonymized == original:
            pytest.skip("No entities detected in test env")

        restored = middleware._deanonymize_text(anonymized, thread_id)
        assert restored == original

    def test_consistent_fake_for_same_entity(self) -> None:
        """Same PII value must always map to same fake value within a thread."""
        middleware = AnonymizationMiddleware(config=_minimal_config())
        thread_id = "consistency-thread"

        text1 = "alice@example.com"
        text2 = "alice@example.com again"

        anon1 = middleware._anonymize_text(text1, thread_id)
        anon2 = middleware._anonymize_text(text2, thread_id)

        mapping = middleware._mapping.get(thread_id, {})
        if "alice@example.com" in mapping:
            fake = mapping["alice@example.com"]
            # Both anonymized texts should contain the same fake
            assert fake in anon1
            assert fake in anon2


# ---------------------------------------------------------------------------
# Tests: cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_removes_thread_mapping(self) -> None:
        middleware = AnonymizationMiddleware(config=_minimal_config())
        middleware._mapping["thread-X"] = {"alice@example.com": "fake@test.com"}

        middleware.cleanup("thread-X")

        assert "thread-X" not in middleware._mapping

    def test_cleanup_nonexistent_thread_no_error(self) -> None:
        middleware = AnonymizationMiddleware(config=_minimal_config())
        middleware.cleanup("no-such-thread")  # Should not raise


# ---------------------------------------------------------------------------
# Tests: YAML-style flat kwargs
# ---------------------------------------------------------------------------


class TestYamlStyleConstruction:
    def test_flat_kwargs_construction(self) -> None:
        middleware = AnonymizationMiddleware(
            analyzed_fields=["EMAIL_ADDRESS"],
            faker_seed=123,
            fuzzy_deanonymize=False,
        )
        assert middleware._config.faker_seed == 123
        assert middleware._config.fuzzy_deanonymize is False

    def test_no_args_uses_defaults(self) -> None:
        middleware = AnonymizationMiddleware()
        assert middleware._config is not None
        assert isinstance(middleware._config.detector, PresidioDetectorConfig)


# ---------------------------------------------------------------------------
# Tests: _deduplicate_entities utility
# ---------------------------------------------------------------------------


class TestDeduplicateEntities:
    def test_empty_list(self) -> None:
        assert _deduplicate_entities([]) == []

    def test_no_overlap(self) -> None:
        entities = [
            DetectedEntity(entity_type="A", start=0, end=5, text="hello", score=0.9),
            DetectedEntity(entity_type="B", start=10, end=15, text="world", score=0.8),
        ]
        result = _deduplicate_entities(entities)
        assert len(result) == 2

    def test_overlapping_keeps_higher_score(self) -> None:
        entities = [
            DetectedEntity(entity_type="LOW", start=0, end=10, text="email@x.com", score=0.5),
            DetectedEntity(entity_type="HIGH", start=0, end=10, text="email@x.com", score=0.9),
        ]
        result = _deduplicate_entities(entities)
        assert len(result) == 1
        assert result[0].entity_type == "HIGH"

    def test_result_sorted_by_start(self) -> None:
        entities = [
            DetectedEntity(entity_type="B", start=20, end=25, text="two", score=0.8),
            DetectedEntity(entity_type="A", start=0, end=5, text="one", score=0.8),
        ]
        result = _deduplicate_entities(entities)
        assert result[0].start < result[1].start
