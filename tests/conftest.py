"""Pytest configuration and shared fixtures.

This module provides shared fixtures and configuration for all tests,
ensuring consistent use of fake models and test setup.
"""

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from genai_tk.core.embeddings_factory import get_embeddings
from genai_tk.core.llm_factory import get_llm
from genai_tk.utils.config_mngr import global_config

# Constants for fake models
FAKE_LLM_ID = "parrot_local_fake"
FAKE_EMBEDDINGS_ID = "embeddings_768_fake"
PYTEST_CONFIG_NAME = "pytest"


@pytest.fixture(scope="session", autouse=True)
def setup_test_config():
    """Set up test configuration for all tests.

    This fixture runs automatically for all tests and ensures:
    - Pytest configuration is selected
    - Fake models are set as defaults
    - Test environment is properly configured
    """
    # Select pytest configuration
    global_config().select_config(PYTEST_CONFIG_NAME)

    # Ensure fake models are defaults for all tests
    global_config().set("llm.models.default", FAKE_LLM_ID)
    global_config().set("embeddings.models.default", FAKE_EMBEDDINGS_ID)

    # Configure test-specific settings
    global_config().set("llm_cache.method", "memory")
    global_config().set("kv_store.engine", "memory")

    yield

    # Cleanup if needed
    pass


@pytest.fixture
def fake_llm() -> BaseChatModel:
    """Provide a fake LLM instance for testing.

    Returns:
        BaseChatModel: Fake LLM instance using parrot_local_fake
    """
    return get_llm(llm_id=FAKE_LLM_ID)


@pytest.fixture
def fake_embeddings():
    """Provide a fake embeddings instance for testing.

    Returns:
        Embeddings: Fake embeddings instance using embeddings_768_fake
    """
    return get_embeddings(embeddings_id=FAKE_EMBEDDINGS_ID)


@pytest.fixture
def fake_llm_with_streaming():
    """Provide a fake LLM instance with streaming enabled.

    Returns:
        BaseChatModel: Fake LLM instance with streaming
    """
    return get_llm(llm_id=FAKE_LLM_ID, streaming=True)


@pytest.fixture
def fake_llm_with_json_mode():
    """Provide a fake LLM instance with JSON mode enabled.

    Returns:
        BaseChatModel: Fake LLM instance with JSON mode
    """
    return get_llm(llm_id=FAKE_LLM_ID, json_mode=True)


@pytest.fixture
def fake_llm_with_cache():
    """Provide a fake LLM instance with caching enabled.

    Returns:
        BaseChatModel: Fake LLM instance with memory cache
    """
    return get_llm(llm_id=FAKE_LLM_ID, cache="memory")


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing.

    Returns:
        List[Document]: List of sample documents with metadata
    """
    from langchain_core.documents import Document

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a powerful programming language for AI development.",
        "Machine learning models require training data to perform well.",
        "Vector databases enable efficient similarity search.",
        "Embeddings represent text as numerical vectors.",
    ]

    return [
        Document(page_content=texts[i % len(texts)], metadata={"id": i, "source": f"doc_{i}", "page": i // 2 + 1})
        for i in range(5)
    ]


@pytest.fixture
def sample_texts():
    """Provide sample text strings for testing.

    Returns:
        List[str]: List of sample text strings
    """
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a powerful programming language for AI development.",
        "Machine learning models require training data to perform well.",
        "Vector databases enable efficient similarity search.",
        "Embeddings represent text as numerical vectors.",
    ]


@pytest.fixture
def sample_queries():
    """Provide sample search queries for testing.

    Returns:
        List[str]: List of sample search queries
    """
    return [
        "programming language",
        "machine learning",
        "similarity search",
        "artificial intelligence",
        "data science",
    ]


@pytest.fixture
def test_config():
    """Provide a test configuration dictionary.

    Returns:
        Dict[str, Any]: Test configuration with fake models
    """
    return {
        "llm": {
            "models": {
                "default": FAKE_LLM_ID,
                "fake": FAKE_LLM_ID,
            },
            "cache": "memory",
        },
        "embeddings": {
            "models": {
                "default": FAKE_EMBEDDINGS_ID,
                "fake": FAKE_EMBEDDINGS_ID,
            },
            "cache": True,
        },
        "vector_store": {
            "default": "InMemory",
        },
        "kv_store": {
            "engine": "memory",
        },
    }


@pytest.fixture
def tool_definitions():
    """Provide sample tool definitions for testing.

    Returns:
        List[Dict[str, Any]]: List of tool definition dictionaries
    """
    return [
        {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        },
    ]


# Performance testing fixtures
@pytest.fixture
def performance_threshold():
    """Provide performance threshold for fake model tests.

    Fake models should be very fast, so we set strict thresholds.

    Returns:
        float: Maximum response time in seconds
    """
    return 1.0  # Fake models should respond in under 1 second


# Error testing fixtures
@pytest.fixture
def invalid_llm_id():
    """Provide an invalid LLM ID for error testing.

    Returns:
        str: Invalid LLM ID that should raise an error
    """
    return "nonexistent_llm_model"


@pytest.fixture
def invalid_embeddings_id():
    """Provide an invalid embeddings ID for error testing.

    Returns:
        str: Invalid embeddings ID that should raise an error
    """
    return "nonexistent_embeddings_model"


# Markers for different test types
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "network: mark test as requiring network access")
    config.addinivalue_line("markers", "fake_models: mark test as using fake models only")


# Skip tests that require real models unless explicitly requested
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip real model tests by default."""
    if not config.getoption("--include-real-models"):
        skip_real_models = pytest.mark.skip(reason="Test requires real models. Use --include-real-models to run.")
        for item in items:
            if "real_models" in item.keywords:
                item.add_marker(skip_real_models)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--include-real-models",
        action="store_true",
        default=False,
        help="Include tests that require real AI models (may incur costs and require API keys)",
    )
    parser.addoption(
        "--performance-tests", action="store_true", default=False, help="Include performance benchmark tests"
    )
