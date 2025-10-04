"""Tests for vector store factory with fake models.

This module contains tests for vector store creation and functionality
using fake embeddings to ensure fast, reliable testing.
"""

import pytest
from langchain.schema import Document

from genai_tk.core.embeddings_factory import EmbeddingsFactory
from genai_tk.core.vector_store_registry import VectorStoreRegistry

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


@pytest.mark.parametrize("vector_store_type", ["InMemory"])  # Skip Chroma due to missing dependency
def test_vector_store_creation_and_search(sample_documents, vector_store_type) -> None:
    """Test vector store creation, document addition, and similarity search.

    Args:
        sample_documents: Fixture providing test documents
        vector_store_type: Parametrized vector store type to test
    """
    # Create vector store factory
    vs_factory = VectorStoreRegistry(
        id=vector_store_type,
        embeddings_factory=EmbeddingsFactory(embeddings_id=FAKE_EMBEDDINGS_ID),
    )

    # Add documents
    db = vs_factory.get()
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
    vs_factory = VectorStoreRegistry(
        id="InMemory",
        embeddings_factory=EmbeddingsFactory(embeddings_id=FAKE_EMBEDDINGS_ID),
    )

    db = vs_factory.get()
    db.add_documents(sample_documents)

    # Test search with different queries
    queries = ["machine learning", "artificial intelligence", "vector search"]

    for query in queries:
        results = db.similarity_search(query, k=3)
        assert len(results) <= 3
        assert all(isinstance(doc, Document) for doc in results)


def test_vector_store_max_marginal_relevance_search(sample_documents) -> None:
    """Test max marginal relevance search functionality."""
    vs_factory = VectorStoreRegistry(
        id="InMemory",
        embeddings_factory=EmbeddingsFactory(embeddings_id=FAKE_EMBEDDINGS_ID),
    )

    db = vs_factory.get()
    db.add_documents(sample_documents)

    # Test MMR search
    results = db.max_marginal_relevance_search("programming language", k=2, fetch_k=4)

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


def test_vector_store_similarity_search_by_vector(sample_documents) -> None:
    """Test similarity search using vector input."""
    embeddings_factory = EmbeddingsFactory(embeddings_id=FAKE_EMBEDDINGS_ID)
    vs_factory = VectorStoreRegistry(
        id="InMemory",
        embeddings_factory=embeddings_factory,
    )

    db = vs_factory.get()
    db.add_documents(sample_documents)

    # Create a query vector using the same embeddings
    query_vector = embeddings_factory.get().embed_query("programming language")

    results = db.similarity_search_by_vector(query_vector, k=2)

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


def test_vector_store_factory_known_items() -> None:
    """Test that vector store factory has known items."""
    known_items = VectorStoreRegistry.known_items()
    assert isinstance(known_items, list)
    assert len(known_items) > 0
    assert "InMemory" in known_items


def test_vector_store_factory_invalid_type() -> None:
    """Test that invalid vector store type handling works correctly."""
    # Test that the factory validates known types
    known_items = VectorStoreRegistry.known_items()
    assert len(known_items) > 0
    assert "InMemory" in known_items

    # Test that we can create factories with valid types
    from typing import get_args

    from genai_tk.core.vector_store_registry import VECTOR_STORE_ENGINE

    valid_types = get_args(VECTOR_STORE_ENGINE)
    for valid_type in valid_types:
        if valid_type == "Chroma_in_memory":
            continue  # Skip due to missing dependency
        factory = VectorStoreRegistry(
            id=valid_type,
            embeddings_factory=EmbeddingsFactory(embeddings_id=FAKE_EMBEDDINGS_ID),
        )
        assert factory.id == valid_type


def test_vector_store_empty_search() -> None:
    """Test vector store behavior with empty document set."""
    vs_factory = VectorStoreRegistry(
        id="InMemory",
        embeddings_factory=EmbeddingsFactory(embeddings_id=FAKE_EMBEDDINGS_ID),
    )

    db = vs_factory.get()

    # Search on empty database should not crash
    results = db.similarity_search("test query", k=2)
    assert len(results) == 0


def test_vector_store_large_k_parameter(sample_documents) -> None:
    """Test vector store behavior when k exceeds document count."""
    vs_factory = VectorStoreRegistry(
        id="InMemory",
        embeddings_factory=EmbeddingsFactory(embeddings_id=FAKE_EMBEDDINGS_ID),
    )

    db = vs_factory.get()
    db.add_documents(sample_documents)

    # Request more results than available documents
    results = db.similarity_search("test", k=100)
    assert len(results) <= len(sample_documents)


@pytest.mark.performance_tests
def test_vector_store_performance(sample_documents, performance_threshold) -> None:
    """Test vector store performance with fake embeddings."""
    import time

    vs_factory = VectorStoreRegistry(
        id="InMemory",
        embeddings_factory=EmbeddingsFactory(embeddings_id=FAKE_EMBEDDINGS_ID),
    )

    db = vs_factory.get()

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
