"""Tests for vector store registry with fake models.

This module contains tests for vector store creation and functionality
using fake embeddings to ensure fast, reliable testing.
"""

import pytest
from langchain_core.documents import Document

from genai_tk.core.embeddings_factory import EmbeddingsFactory
from genai_tk.core.embeddings_store import EmbeddingsStore

# Fake model constants
FAKE_EMBEDDINGS_ID = "embeddings_768_fake"


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="The quick brown fox jumps over the lazy dog"),
        Document(page_content="Python is a powerful programming language"),
        Document(page_content="Machine learning is transforming many industries"),
        Document(page_content="Vector search uses embeddings for similarity"),
        Document(page_content="Artificial intelligence enables smart systems"),
    ]


@pytest.fixture
def fresh_embeddings_store():
    """Create a fresh in-memory embeddings store for each test.

    This fixture ensures test isolation by creating a unique store
    instance for each test, preventing document pollution between tests.
    """
    # Use in_memory_chroma config which creates a fresh in-memory store
    embeddings_store = EmbeddingsStore.create_from_config("in_memory_chroma")
    yield embeddings_store


@pytest.mark.parametrize("config_name", ["default"])  # Use existing config
def test_vector_store_creation_and_search(sample_documents, config_name) -> None:
    """Test vector store creation, document addition, and similarity search.

    Args:
        sample_documents: Fixture providing test documents
        config_name: Configuration name to test
    """
    # Create vector store factory from config
    embeddings_store = EmbeddingsStore.create_from_config(config_name)

    # Add documents
    db = embeddings_store.get_vector_store()
    db.add_documents(sample_documents)

    # Perform similarity search
    query = "programming language"
    results = db.similarity_search(query, k=2)

    # Basic validation
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)

    # With fake embeddings, we just check that we get some results
    # The similarity might not be semantically accurate with fake embeddings
    result_contents = [doc.page_content for doc in results]
    assert len(result_contents) == 2
    assert all(isinstance(content, str) for content in result_contents)


def test_vector_store_with_fake_embeddings(sample_documents) -> None:
    """Test vector store specifically with fake embeddings."""
    embeddings_store = EmbeddingsStore.create_from_config("default")

    db = embeddings_store.get_vector_store()
    db.add_documents(sample_documents)

    # Test search with different queries
    queries = ["machine learning", "artificial intelligence", "vector search"]

    for query in queries:
        results = db.similarity_search(query, k=3)
        assert len(results) <= 3
        assert all(isinstance(doc, Document) for doc in results)


def test_vector_store_max_marginal_relevance_search(sample_documents) -> None:
    """Test max marginal relevance search functionality."""
    embeddings_store = EmbeddingsStore.create_from_config("default")

    db = embeddings_store.get_vector_store()
    db.add_documents(sample_documents)

    # Test MMR search
    results = db.max_marginal_relevance_search("programming language", k=2, fetch_k=4)

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


def test_direct_instantiation_blocked() -> None:
    """Test that direct instantiation is blocked."""
    with pytest.raises(RuntimeError, match="EmbeddingsStore cannot be instantiated directly"):
        EmbeddingsStore(
            backend="InMemory",
            embeddings_factory=EmbeddingsFactory(embeddings=FAKE_EMBEDDINGS_ID),
        )


def test_vector_store_similarity_search_by_vector(sample_documents) -> None:
    """Test similarity search using vector input."""
    embeddings_store = EmbeddingsStore.create_from_config("default")
    embeddings_factory = embeddings_store.embeddings_factory

    db = embeddings_store.get_vector_store()
    db.add_documents(sample_documents)

    # Create a query vector using the same embeddings
    query_vector = embeddings_factory.get().embed_query("programming language")

    results = db.similarity_search_by_vector(query_vector, k=2)

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


def test_vector_store_factory_known_items() -> None:
    """Test that vector store factory has correct known items."""
    known_items = EmbeddingsStore.known_items()
    assert isinstance(known_items, list)
    assert len(known_items) > 0
    assert "InMemory" in known_items
    assert "Chroma" in known_items
    assert "Sklearn" in known_items
    assert "PgVector" in known_items
    # Ensure deprecated Chroma_in_memory is no longer in known items
    assert "Chroma_in_memory" not in known_items


def test_vector_store_empty_search(fresh_embeddings_store) -> None:
    """Test vector store behavior with empty document set."""
    db = fresh_embeddings_store.get_vector_store()

    # Search on empty database should not crash
    results = db.similarity_search("test query", k=2)
    assert len(results) == 0


def test_vector_store_large_k_parameter(sample_documents, fresh_embeddings_store) -> None:
    """Test vector store behavior when k exceeds document count."""
    db = fresh_embeddings_store.get_vector_store()
    db.add_documents(sample_documents)

    # Request more results than available documents
    results = db.similarity_search("test", k=100)
    assert len(results) <= len(sample_documents)


@pytest.mark.performance_tests
def test_vector_store_performance(sample_documents, performance_threshold) -> None:
    """Test vector store performance with fake embeddings."""
    import time

    embeddings_store = EmbeddingsStore.create_from_config("default")

    db = embeddings_store.get_vector_store()

    # Measure document addition time
    start_time = time.time()
    db.add_documents(sample_documents)
    add_time = time.time() - start_time

    # Measure search time
    start_time = time.time()
    results = db.similarity_search("test query", k=2)
    search_time = time.time() - start_time

    # Fake embeddings should make operations very fast
    assert add_time < performance_threshold
    assert search_time < performance_threshold
    assert len(results) == 2


def test_chroma_memory_storage(sample_documents) -> None:
    """Test Chroma with in-memory storage using new storage field."""
    embeddings_store = EmbeddingsStore.create_from_config("in_memory_chroma")
    assert embeddings_store.backend == "Chroma"
    assert embeddings_store.config.get("storage") == "::memory::"

    db = embeddings_store.get_vector_store()
    db.add_documents(sample_documents)

    # Verify we can search
    results = db.similarity_search("test", k=2)
    assert len(results) == 2


@pytest.mark.skip(reason="PostgreSQL tests temporarily disabled")
def test_postgres_vector_store() -> None:
    """Test PgVector store - skipped for now."""
    # This test would require a running PostgreSQL instance
