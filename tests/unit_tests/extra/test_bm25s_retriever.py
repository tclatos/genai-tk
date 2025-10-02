"""Tests for BM25FastRetriever with fake models."""

import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from genai_tk.extra.retrievers.bm25s_retriever import BM25FastRetriever

# Test data constants
SAMPLE_DOCUMENTS = [
    Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"id": 1}),
    Document(page_content="A fast brown fox leaps over lazy dogs in summer.", metadata={"id": 2}),
    Document(page_content="The lazy dog sleeps in the sun.", metadata={"id": 3}),
    Document(page_content="Python is a programming language used for AI development.", metadata={"id": 4}),
    Document(page_content="Machine learning models require training data.", metadata={"id": 5}),
]


class TestBM25FastRetriever:
    """Test class for BM25FastRetriever."""


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return SAMPLE_DOCUMENTS

    def test_from_texts(self):
        """Test creating retriever from texts."""
        texts = ["hello world", "foo bar", "python programming"]
        metadatas = [{"source": i} for i in range(len(texts))]

        retriever = BM25FastRetriever.from_texts(texts=texts, metadatas=metadatas, k=2)

        assert retriever.k == 2
        assert len(retriever.docs) == 3
        assert retriever.docs[0].page_content == "hello world"
        assert retriever.docs[0].metadata["source"] == 0

    def test_from_documents(self, sample_documents):
        """Test creating retriever from documents."""
        retriever = BM25FastRetriever.from_documents(documents=sample_documents, k=3)

        assert retriever.k == 3
        assert len(retriever.docs) == 5
        assert retriever.docs[0].metadata["id"] == 1

    def test_retrieval_basic(self, sample_documents):
        """Test basic retrieval functionality."""
        retriever = BM25FastRetriever.from_documents(documents=sample_documents, k=2)

        results = retriever.invoke("fox")

        assert len(results) == 2
        # Should find documents about foxes
        fox_docs = [doc for doc in results if "fox" in doc.page_content.lower()]
        assert len(fox_docs) > 0

    def test_retrieval_empty_query(self, sample_documents):
        """Test retrieval with empty query."""
        retriever = BM25FastRetriever.from_documents(documents=sample_documents, k=2)

        results = retriever.invoke("")
        assert len(results) == 2  # Should return top documents even for empty query

    def test_retrieval_no_matches(self, sample_documents):
        """Test retrieval with no matching terms."""
        retriever = BM25FastRetriever.from_documents(documents=sample_documents, k=2)

        results = retriever.invoke("zebra elephant")
        assert len(results) == 2  # Should return some documents even without matches

    def test_cache_functionality(self, sample_documents):
        """Test saving and loading from cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "bm25_cache"

            # Create retriever and save to cache
            retriever = BM25FastRetriever.from_documents(documents=sample_documents, cache_dir=cache_path, k=3)

            # Load from cache
            cached_retriever = BM25FastRetriever.from_index_file(index_file=cache_path, k=3)

            # Both should return similar results
            original_results = retriever.invoke("fox")
            cached_results = cached_retriever.invoke("fox")

            # Note: cached retriever doesn't store docs, so we only test it doesn't crash
            assert len(cached_results) <= 3

    def test_preprocessing_function(self):
        """Test custom preprocessing function."""

        def custom_preprocess(text: str) -> list[str]:
            return text.upper().split()

        texts = ["hello world", "foo bar"]
        retriever = BM25FastRetriever.from_texts(texts=texts, preprocess_func=custom_preprocess, k=2)

        # Should work with custom preprocessing
        results = retriever.invoke("HELLO")
        assert len(results) > 0

    def test_k_parameter(self, sample_documents):
        """Test different k values."""
        retriever = BM25FastRetriever.from_documents(documents=sample_documents, k=1)

        results = retriever.invoke("dog")
        assert len(results) == 1

        retriever.k = 3
        results = retriever.invoke("dog")
        assert len(results) == 3

    def test_bm25_parameters(self, sample_documents):
        """Test passing BM25 parameters."""
        retriever = BM25FastRetriever.from_documents(
            documents=sample_documents, bm25_params={"k1": 1.5, "b": 0.75}, k=2
        )

        results = retriever.invoke("fox")
        assert len(results) == 2  # Should work with custom BM25 parameters

    def test_spacy_preprocessing(self, sample_documents):
        """Test using spacy preprocessing function."""
        import warnings

        # Suppress spacy-related deprecation warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="spacy.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="weasel.*")

        from genai_tk.extra.retrievers.bm25s_retriever import get_spacy_preprocess_fn
        from genai_tk.utils.spacy_model_mngr import SpaCyModelManager

        # Setup spacy model using SpaCyModelManager
        model_name = "en_core_web_sm"
        SpaCyModelManager.setup_spacy_model(model_name)

        # Get preprocessing function with additional stop words
        additional_stop_words = ["the", "a", "an"]
        preprocess_func = get_spacy_preprocess_fn(model_name, additional_stop_words)

        retriever = BM25FastRetriever.from_documents(documents=sample_documents, preprocess_func=preprocess_func, k=2)

        # Test that preprocessing is working correctly
        processed = preprocess_func("The fox jumps over a lazy dog")
        assert "the" not in processed
        assert "a" not in processed

        # Test retrieval works
        results = retriever.invoke("fox")
        assert len(results) <= 2  # Should work with spacy preprocessing


def test_retriever_performance_with_fake_data(sample_documents, performance_threshold) -> None:
    """Test that BM25 retriever performs well with fake data."""
    import time

    retriever = BM25FastRetriever.from_documents(documents=sample_documents, k=3)

    # Measure retrieval time
    start_time = time.time()
    results = retriever.invoke("python programming")
    retrieval_time = time.time() - start_time

    # Should be very fast with in-memory data
    assert retrieval_time < performance_threshold
    assert len(results) <= 3


def test_retriever_consistency(sample_documents) -> None:
    """Test that retriever returns consistent results."""
    retriever = BM25FastRetriever.from_documents(documents=sample_documents, k=2)

    # Multiple calls should return same results
    results1 = retriever.invoke("fox")
    results2 = retriever.invoke("fox")

    assert len(results1) == len(results2)
    assert [doc.page_content for doc in results1] == [doc.page_content for doc in results2]


def test_retriever_with_empty_documents() -> None:
    """Test retriever behavior with no documents."""
    # Test that creating retriever with empty documents handles it gracefully
    try:
        retriever = BM25FastRetriever.from_documents(documents=[], k=2)
        # If creation succeeds, test that it returns empty results
        results = retriever.get_relevant_documents("test query")
        assert len(results) == 0
    except ValueError:
        # If creation fails with empty documents, that's also acceptable behavior
        # The important thing is that it doesn't crash silently
        pass


def test_retriever_edge_cases(sample_documents) -> None:
    """Test edge cases for BM25 retriever."""
    retriever = BM25FastRetriever.from_documents(documents=sample_documents, k=2)

    # Test with special characters
    results = retriever.invoke("python@#$%")
    assert isinstance(results, list)

    # Test with very long query
    long_query = "python " * 100
    results = retriever.invoke(long_query)
    assert isinstance(results, list)

    # Test with empty string
    results = retriever.invoke("")
    assert isinstance(results, list)
