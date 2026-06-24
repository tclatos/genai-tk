"""Integration tests for end-to-end workflows using fake models.

Each test verifies a single, concrete behavior of the pipeline.
All tests use fake models (deterministic, zero-cost, no network).
"""

import pytest
from langchain_core.documents import Document

from genai_tk.core.embeddings_store import EmbeddingsStore

KNOWN_DOCS = [
    Document(page_content="Python is a high-level programming language created by Guido van Rossum."),
    Document(page_content="Python is widely used for web development, data science, and artificial intelligence."),
    Document(page_content="The language emphasizes code readability and simple syntax."),
]


# ---------------------------------------------------------------------------
# Vector store / retrieval behavior
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.fake_models
def test_vector_store_add_and_retrieve_returns_k_results(sample_documents) -> None:
    """Adding documents to the vector store then searching returns exactly k results."""
    embeddings_store = EmbeddingsStore.create_from_config("default")
    vector_store = embeddings_store.get_vector_store()
    vector_store.add_documents(sample_documents)

    results = vector_store.similarity_search("programming language", k=2)

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


@pytest.mark.integration
@pytest.mark.fake_models
def test_vector_store_returns_document_objects(sample_documents) -> None:
    """Similarity search results are LangChain Document instances."""
    embeddings_store = EmbeddingsStore.create_from_config("default")
    vector_store = embeddings_store.get_vector_store()
    vector_store.add_documents(sample_documents)

    results = vector_store.similarity_search("neural networks", k=3)

    assert all(isinstance(doc, Document) for doc in results)
    assert all(isinstance(doc.page_content, str) for doc in results)


# ---------------------------------------------------------------------------
# Embedding behavior
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.fake_models
def test_fake_embeddings_produce_correct_dimension(fake_embeddings) -> None:
    """Fake embeddings produce 768-dimensional vectors."""
    vectors = fake_embeddings.embed_documents(["sentence one", "sentence two"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 768
    assert len(vectors[1]) == 768


@pytest.mark.integration
@pytest.mark.fake_models
def test_fake_embeddings_are_deterministic(fake_embeddings) -> None:
    """Fake embeddings produce identical vectors for the same text."""
    text = "deterministic embedding check"
    assert fake_embeddings.embed_query(text) == fake_embeddings.embed_query(text)


# ---------------------------------------------------------------------------
# LLM + retrieval collaboration
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.fake_models
def test_llm_generates_response_from_retrieved_context(fake_llm, sample_documents) -> None:
    """LLM produces non-empty output when given retrieved document context."""
    embeddings_store = EmbeddingsStore.create_from_config("default")
    vector_store = embeddings_store.get_vector_store()
    vector_store.add_documents(sample_documents)

    retrieved = vector_store.similarity_search("programming language", k=2)
    context = "\n".join(doc.page_content for doc in retrieved)
    response = fake_llm.invoke(f"Context:\n{context}\n\nQuestion: What is programming?")

    assert response is not None
    assert len(response.content) > 0


@pytest.mark.integration
@pytest.mark.fake_models
@pytest.mark.parametrize(
    "question",
    [
        "Who created Python?",
        "What is Python used for?",
        "What are Python's key features?",
    ],
)
def test_qa_pipeline_produces_answer_for_each_question(fake_llm, question) -> None:
    """Each question produces a non-empty answer from the fake LLM."""
    embeddings_store = EmbeddingsStore.create_from_config("default")
    vector_store = embeddings_store.get_vector_store()
    vector_store.add_documents(KNOWN_DOCS)

    retrieved = vector_store.similarity_search(question, k=2)
    context = "\n".join(doc.page_content for doc in retrieved)
    answer = fake_llm.invoke(f"Context: {context}\n\nQuestion: {question}\n\nAnswer:")

    assert answer is not None
    assert len(answer.content) > 0
