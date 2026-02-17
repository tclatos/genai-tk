"""End-to-end integration tests using fake models.

This module contains integration tests that verify complete workflows
work correctly with fake models, ensuring fast and reliable testing.
"""

import pytest
from langchain_core.documents import Document

from genai_tk.core.embeddings_factory import EmbeddingsFactory, get_embeddings
from genai_tk.core.embeddings_store import EmbeddingsStore
from genai_tk.core.llm_factory import get_llm

# Constants
FAKE_LLM_ID = "parrot_local@fake"
FAKE_EMBEDDINGS_ID = "embeddings_768@fake"


@pytest.mark.integration
@pytest.mark.fake_models
def test_rag_pipeline_with_fake_models(sample_documents) -> None:
    """Test complete RAG pipeline with fake models."""
    # Create vector store from configuration
    embeddings_store = EmbeddingsStore.create_from_config("default")
    vector_store = embeddings_store.get_vector_store()

    # Add documents to vector store
    vector_store.add_documents(sample_documents)

    # Create LLM
    llm = get_llm(llm=FAKE_LLM_ID)

    # Test retrieval
    query = "programming language"
    retrieved_docs = vector_store.similarity_search(query, k=2)

    assert len(retrieved_docs) == 2
    assert all(isinstance(doc, Document) for doc in retrieved_docs)

    # Test generation
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"Based on this context:\n{context}\n\nAnswer: What is {query}?"

    response = llm.invoke(prompt)
    assert response is not None
    assert hasattr(response, "content")
    assert len(response.content) > 0


@pytest.mark.integration
@pytest.mark.fake_models
def test_document_processing_pipeline(sample_documents) -> None:
    """Test document processing pipeline with fake models."""
    # Create embeddings factory
    embeddings_factory = EmbeddingsFactory(embeddings=FAKE_EMBEDDINGS_ID)
    embeddings = embeddings_factory.get()

    # Test embedding generation
    documents = sample_documents[:3]
    embedded_docs = []

    for doc in documents:
        vector = embeddings.embed_query(doc.page_content)
        embedded_docs.append((doc, vector))

    assert len(embedded_docs) == len(documents)
    assert all(len(vector) == 768 for _, vector in embedded_docs)  # Fake embedding dimension

    # Create vector store and add embedded documents
    embeddings_store = EmbeddingsStore.create_from_config("default")
    vector_store = embeddings_store.get_vector_store()
    vector_store.add_documents(documents)

    # Test semantic search
    results = vector_store.similarity_search("neural networks", k=2)
    assert len(results) == 2

    # Test LLM processing of retrieved documents
    llm = get_llm(llm=FAKE_LLM_ID)
    context = "\n".join([doc.page_content for doc in results])

    summary_prompt = f"Summarize this information:\n{context}"
    summary = llm.invoke(summary_prompt)

    assert summary is not None
    assert len(summary.content) > 0


@pytest.mark.integration
@pytest.mark.fake_models
def test_multi_step_reasoning_workflow() -> None:
    """Test multi-step reasoning workflow with fake models."""
    llm = get_llm(llm=FAKE_LLM_ID)

    # Step 1: Generate analysis
    analysis_prompt = "Analyze the benefits of using fake models for testing"
    analysis = llm.invoke(analysis_prompt)
    assert analysis is not None

    # Step 2: Generate recommendations based on analysis
    recommendation_prompt = f"Based on this analysis: {analysis.content}\n\nProvide recommendations"
    recommendations = llm.invoke(recommendation_prompt)
    assert recommendations is not None

    # Step 3: Generate final summary
    summary_prompt = (
        f"Analysis: {analysis.content}\n\nRecommendations: {recommendations.content}\n\nProvide a final summary"
    )
    final_summary = llm.invoke(summary_prompt)
    assert final_summary is not None

    # Verify all steps produced content
    assert len(analysis.content) > 0
    assert len(recommendations.content) > 0
    assert len(final_summary.content) > 0


@pytest.mark.integration
@pytest.mark.fake_models
def test_question_answering_pipeline() -> None:
    """Test question answering pipeline with fake models."""
    # Knowledge base
    knowledge_docs = [
        Document(page_content="Python is a high-level programming language created by Guido van Rossum."),
        Document(page_content="Python is widely used for web development, data science, and artificial intelligence."),
        Document(page_content="The language emphasizes code readability and simple syntax."),
    ]

    # Create vector store from configuration
    embeddings_store = EmbeddingsStore.create_from_config("default")
    vector_store = embeddings_store.get_vector_store()
    vector_store.add_documents(knowledge_docs)

    # Test questions
    questions = [
        "Who created Python?",
        "What is Python used for?",
        "What are Python's key features?",
    ]

    llm = get_llm(llm=FAKE_LLM_ID)

    for question in questions:
        # Retrieve relevant documents
        retrieved_docs = vector_store.similarity_search(question, k=2)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Generate answer
        qa_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        answer = llm.invoke(qa_prompt)

        assert answer is not None
        assert len(answer.content) > 0
        # Answer should be relevant to the question (basic check with fake models)
        question_words = [word.lower() for word in question.split() if len(word) > 2]
        assert any(word in answer.content.lower() for word in question_words)


@pytest.mark.integration
@pytest.mark.fake_models
def test_content_generation_pipeline() -> None:
    """Test content generation pipeline with fake models."""
    llm = get_llm(llm=FAKE_LLM_ID)

    # Generate different types of content
    content_types = [
        ("blog post", "Write a blog post about the benefits of AI testing"),
        ("summary", "Summarize the key points of machine learning"),
        ("code example", "Write a simple Python function that calculates factorial"),
        ("explanation", "Explain the concept of vector embeddings"),
    ]

    generated_content = []

    for content_type, prompt in content_types:
        response = llm.invoke(prompt)
        assert response is not None
        assert len(response.content) > 0

        generated_content.append({"type": content_type, "content": response.content})

    # Verify all content was generated
    assert len(generated_content) == len(content_types)

    # Verify content diversity (fake models should still produce different responses)
    contents = [item["content"] for item in generated_content]
    assert len(set(contents)) == len(contents)  # All should be unique


@pytest.mark.integration
@pytest.mark.fake_models
def test_performance_benchmark() -> None:
    """Test performance of complete workflows with fake models."""
    import time

    # Setup components
    embeddings = get_embeddings(embeddings=FAKE_EMBEDDINGS_ID)
    llm = get_llm(llm=FAKE_LLM_ID)

    # Test embedding performance
    start_time = time.time()
    texts = ["test text"] * 10
    embedded_vectors = embeddings.embed_documents(texts)
    embedding_time = time.time() - start_time

    assert len(embedded_vectors) == 10
    assert all(len(vector) == 768 for vector in embedded_vectors)

    # Test LLM performance
    start_time = time.time()
    responses = []
    for i in range(5):
        response = llm.invoke(f"Test prompt {i}")
        responses.append(response)
    llm_time = time.time() - start_time

    assert len(responses) == 5
    assert all(response is not None for response in responses)

    # Performance assertions (fake models should be fast)
    assert embedding_time < 2.0  # Should embed 10 texts in under 2 seconds
    assert llm_time < 3.0  # Should generate 5 responses in under 3 seconds


@pytest.mark.integration
@pytest.mark.fake_models
def test_error_handling_pipeline() -> None:
    """Test error handling in pipelines with fake models."""
    llm = get_llm(llm=FAKE_LLM_ID)

    # Test with empty input
    response = llm.invoke("")
    assert response is not None

    # Test with very long input
    long_input = "test " * 1000
    response = llm.invoke(long_input)
    assert response is not None

    # Test with special characters
    special_input = "Test with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
    response = llm.invoke(special_input)
    assert response is not None

    # Test with unicode characters
    unicode_input = "Test with unicode: ñáéíóú 🤖 AI testing"
    response = llm.invoke(unicode_input)
    assert response is not None


@pytest.mark.integration
@pytest.mark.fake_models
def test_consistency_validation() -> None:
    """Test that fake models provide consistent responses."""
    llm = get_llm(llm=FAKE_LLM_ID)

    # Test multiple calls with same input
    test_prompt = "Generate a consistent response"
    responses = []

    for _ in range(5):
        response = llm.invoke(test_prompt)
        responses.append(response.content)

    # Fake models should be deterministic
    assert len(set(responses)) == 1  # All responses should be identical

    # Test with different parameters
    llm_temp = get_llm(llm=FAKE_LLM_ID, temperature=0.5)
    response_diff_temp = llm_temp.invoke(test_prompt)

    # Even with different temperature, fake models might be consistent
    # This test documents the current behavior
    assert response_diff_temp is not None
