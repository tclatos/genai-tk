"""Tests for embeddings functionality.

This module contains basic regression tests for the embeddings factory and utilities.
"""

from genai_tk.core.embeddings_factory import EmbeddingsFactory

# Fake model constants
FAKE_EMBEDDINGS_ID = "embeddings_768@fake"
FAKE_EMBEDDINGS_DIMENSION = 768
LOCAL_FASTEMBED_ID = "bge-small-en@local"
LOCAL_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDINGS_CACHE_DIR = "data/hf_models"

SENTENCE_1 = "Tokenization is the process of breaking down a text into individual units."
SENTENCE_2 = "Tokens can be words, phrases, or even individual characters."


def test_fake_embeddings(fake_embeddings) -> None:
    """Test that default embeddings can be created and used."""
    vectors = fake_embeddings.embed_documents([SENTENCE_1, SENTENCE_2])

    # Basic validation of embeddings
    assert len(vectors) == 2
    assert len(vectors[0]) == FAKE_EMBEDDINGS_DIMENSION
    assert len(vectors[1]) == FAKE_EMBEDDINGS_DIMENSION


def test_known_embeddings_list() -> None:
    """Test that known embeddings list is not empty."""
    embeddings_list = EmbeddingsFactory.known_items()
    assert len(embeddings_list) > 0
    assert isinstance(embeddings_list, list)
    assert FAKE_EMBEDDINGS_ID in embeddings_list
    assert LOCAL_FASTEMBED_ID in embeddings_list


def test_embeddings_with_single_text(fake_embeddings) -> None:
    """Test embedding a single text."""
    vector = fake_embeddings.embed_query(SENTENCE_1)
    assert len(vector) == FAKE_EMBEDDINGS_DIMENSION


def test_embeddings_consistency(fake_embeddings) -> None:
    """Test that embeddings are consistent for the same input."""
    vector1 = fake_embeddings.embed_query(SENTENCE_1)
    vector2 = fake_embeddings.embed_query(SENTENCE_1)
    assert vector1 == vector2  # Fake embeddings should be deterministic


def test_fastembed_local_model_uses_configured_cache(monkeypatch) -> None:
    """Test FastEmbed local model initialization uses configured model and cache directory."""
    captured_kwargs: dict[str, str] = {}

    class FakeFastEmbedEmbeddings:
        """Simple fake replacement for FastEmbedEmbeddings constructor capture."""

        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * 384 for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [0.0] * 384

    import langchain_community.embeddings.fastembed as fastembed_module

    monkeypatch.setattr(fastembed_module, "FastEmbedEmbeddings", FakeFastEmbedEmbeddings)

    embedder = EmbeddingsFactory(embeddings=LOCAL_FASTEMBED_ID).get()

    assert isinstance(embedder, FakeFastEmbedEmbeddings)
    assert captured_kwargs["model_name"] == LOCAL_FASTEMBED_MODEL
    assert captured_kwargs["cache_dir"] == EMBEDDINGS_CACHE_DIR
