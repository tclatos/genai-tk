# Test Quick Reference Guide

This guide provides quick examples for writing tests in genai-tk.

## Using Fake Models (Recommended for Unit Tests)

### Basic LLM Test
```python
def test_my_llm_feature(fake_llm):
    """Test feature using fake LLM."""
    response = fake_llm.invoke("Tell me a joke")
    assert response is not None
    assert len(response.content) > 0
```

### Basic Embeddings Test
```python
def test_my_embeddings_feature(fake_embeddings):
    """Test feature using fake embeddings."""
    text = "Test document"
    embedding = fake_embeddings.embed_query(text)
    assert len(embedding) == 768  # fake embeddings are 768-dimensional
```

### Using Sample Documents
```python
def test_document_processing(sample_documents):
    """Test with sample documents."""
    assert len(sample_documents) == 5
    for doc in sample_documents:
        assert doc.page_content
        assert "source" in doc.metadata
```

## Creating Test Files

### Temporary Directory
```python
def test_file_processing(temp_test_dir):
    """Test file operations."""
    from tests.utils.test_data import create_test_file
    
    test_file = create_test_file(temp_test_dir, "test.txt", "content")
    assert test_file.exists()
```

### Multiple Test Files
```python
def test_batch_processing(temp_test_dir):
    """Test batch file processing."""
    from tests.utils.test_data import create_sample_markdown_files
    
    files = create_sample_markdown_files(temp_test_dir, count=5)
    assert len(files) == 5
```

## Vector Store Tests

### Basic Vector Store Test
```python
def test_vector_store_search(sample_documents):
    """Test vector store search."""
    from genai_tk.core.embeddings_store import EmbeddingsStore
    
    # Create in-memory vector store
    embeddings_store = EmbeddingsStore.create_from_config("default")
    db = embeddings_store.get_vector_store()
    
    # Add documents
    db.add_documents(sample_documents)
    
    # Search
    results = db.similarity_search("programming", k=2)
    assert len(results) == 2
```

### Using Fresh Embeddings Store (Isolated Tests)
```python
@pytest.fixture
def fresh_embeddings_store():
    """Create a fresh in-memory embeddings store for each test."""
    from genai_tk.core.embeddings_store import EmbeddingsStore
    return EmbeddingsStore.create_from_config("in_memory_chroma")

def test_isolated_search(fresh_embeddings_store, sample_documents):
    """Test with isolated vector store."""
    db = fresh_embeddings_store.get_vector_store()
    db.add_documents(sample_documents)
    
    results = db.similarity_search("test", k=1)
    assert len(results) <= 1
```

## Async Tests

```python
import pytest

@pytest.mark.asyncio
async def test_async_operation(fake_llm):
    """Test async operations."""
    from langchain_core.runnables import RunnablePassthrough
    
    chain = RunnablePassthrough.assign(response=fake_llm)
    result = await chain.ainvoke({"query": "test"})
    assert "response" in result
```

## Mocking External Dependencies

### Mock Vector Store for Tool Tests
```python
from unittest.mock import Mock, AsyncMock, patch

@patch("genai_tk.tools.langchain.rag_tool_factory.EmbeddingsStore")
@pytest.mark.asyncio
async def test_rag_tool(mock_embeddings_store_class):
    """Test RAG tool with mocked vector store."""
    from langchain_core.documents import Document
    from genai_tk.tools.langchain.rag_tool_factory import RAGToolFactory, RAGToolConfig
    
    # Mock embeddings store
    mock_embeddings_store = Mock()
    mock_embeddings_store.query = AsyncMock(
        return_value=[Document(page_content="Test result")]
    )
    mock_embeddings_store_class.create_from_config.return_value = mock_embeddings_store
    
    # Create and test tool
    config = RAGToolConfig(embeddings_store="test", top_k=1)
    factory = RAGToolFactory(llm=Mock())
    tool = factory.create_tool(config)
    
    result = await tool.ainvoke({"query": "test"})
    assert "Test result" in result
```

## Test Data Generators

```python
from tests.utils.test_data import (
    generate_sample_documents,
    generate_sample_texts,
    generate_sample_queries,
)

def test_with_custom_data():
    """Test with custom generated data."""
    # Generate 10 documents
    docs = generate_sample_documents(count=10)
    assert len(docs) == 10
    
    # Generate texts
    texts = generate_sample_texts(count=5)
    assert len(texts) == 5
    
    # Get search queries
    queries = generate_sample_queries()
    assert len(queries) > 0
```

## Configuration Tests

```python
def test_config_setting():
    """Test configuration changes."""
    from genai_tk.utils.config_mngr import global_config
    
    # Get current value
    original = global_config().get("llm.models.default")
    
    # Change value
    global_config().set("llm.models.default", "test_model")
    assert global_config().get("llm.models.default") == "test_model"
    
    # Restore (fixture does this automatically)
    global_config().set("llm.models.default", original)
```

## Common Patterns

### Testing Error Handling
```python
def test_error_handling(fake_llm):
    """Test error handling."""
    from genai_tk.core.llm_factory import get_llm
    import pytest
    
    # Test invalid model
    with pytest.raises(ValueError, match="Unknown LLM"):
        get_llm(llm="nonexistent_model")
```

### Testing File Processing
```python
def test_file_processing(temp_test_dir):
    """Test file processing with manifest."""
    from datetime import datetime, timezone
    from tests.utils.test_data import create_sample_markdown_files
    
    # Create test files
    files = create_sample_markdown_files(temp_test_dir, count=3)
    
    # Process files
    for file in files:
        content = file.read_text()
        assert "# Document" in content
```

### Testing Pydantic Models
```python
def test_config_validation():
    """Test Pydantic model validation."""
    from genai_tk.tools.langchain.rag_tool_factory import RAGToolConfig
    
    # Valid config
    config = RAGToolConfig(embeddings_store="test")
    assert config.embeddings_store == "test"
    assert config.top_k == 4  # default value
    
    # Test defaults
    assert config.default_filter is None
```

## Markers

```python
# Mark as unit test
@pytest.mark.unit
def test_unit_feature():
    pass

# Mark as integration test
@pytest.mark.integration
def test_integration_workflow():
    pass

# Mark as slow test
@pytest.mark.slow
def test_slow_operation():
    pass

# Skip test conditionally
@pytest.mark.skipif(condition, reason="explanation")
def test_conditional():
    pass

# Mark test for specific functionality
@pytest.mark.fake_models
def test_with_fake_models(fake_llm):
    pass
```

## Best Practices

1. **Use Fixtures**: Reuse common setup via fixtures
2. **Use Fake Models**: Default to fake models for speed and reliability
3. **Isolate Tests**: Each test should be independent
4. **Clear Names**: Test names should describe what's being tested
5. **One Assertion Focus**: Each test should verify one thing
6. **Use Shared Data**: Import from `tests.utils.test_data`
7. **Clean Up**: Use fixtures or context managers for cleanup
8. **Test What Matters**: Focus on behavior, not implementation

## Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run unit tests only
uv run pytest tests/unit_tests/

# Run specific module
uv run pytest tests/unit_tests/core/test_llm_factory.py

# Run specific test
uv run pytest tests/unit_tests/core/test_llm_factory.py::test_basic_call

# Run with verbose output
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=genai_tk

# Run and stop on first failure
uv run pytest tests/ -x

# Run matching pattern
uv run pytest tests/ -k "llm"
```

## Troubleshooting

### Tests Hanging
- Check for network calls (use fake models)
- Check for infinite loops in code under test
- Use `pytest --timeout=10` to set timeout

### Import Errors
- Ensure virtual environment is activated
- Run `uv sync` to install dependencies
- Check PYTHONPATH includes project root

### Fixture Not Found
- Ensure fixture is in conftest.py or imported module
- Check fixture scope matches test scope
- Import fixture explicitly if needed

### Tests Fail Randomly
- Check for test interdependencies
- Ensure proper cleanup in fixtures
- Use fresh fixtures (e.g., `fresh_embeddings_store`)
- Check for shared state in test data
