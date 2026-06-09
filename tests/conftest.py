"""Pytest configuration and shared fixtures.

This module provides shared fixtures and configuration for all tests,
ensuring consistent use of fake models and test setup.
"""

from pathlib import Path
from typing import Generator

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from genai_tk.config_mgmt.config_mngr import switch_profile
from genai_tk.config_mgmt.features import is_available
from genai_tk.core.factories.embeddings_factory import get_embeddings
from genai_tk.core.factories.llm_factory import get_llm

# Constants for fake models
FAKE_LLM_ID = "parrot_local@fake"
FAKE_EMBEDDINGS_ID = "embeddings_768@fake"
PYTEST_PROFILE = "pytest"


# ---------------------------------------------------------------------------
# Optional-feature marker: @pytest.mark.requires_feature("<name>")
# Tests with this marker are automatically skipped when the feature is absent.
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_feature(name): skip test when the named optional feature is not installed",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    include_real_models = config.getoption("--include-real-models", default=False)
    include_docker = config.getoption("--include-docker", default=False)

    for item in items:
        # Skip real-model tests unless opted in
        if not include_real_models and item.get_closest_marker("real_models"):
            item.add_marker(pytest.mark.skip(reason="Skipping real model tests (use --include-real-models to run)"))

        # Skip docker tests unless opted in
        if not include_docker and item.get_closest_marker("docker"):
            item.add_marker(pytest.mark.skip(reason="Skipping Docker tests (use --include-docker to run)"))

        # Skip tests for missing optional features
        for marker in item.iter_markers("requires_feature"):
            feature: str = marker.args[0]
            if not is_available(feature):
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"Optional feature '{feature}' not installed — run: uv sync --extra {feature}"
                    )
                )


@pytest.fixture(scope="session", autouse=True)
def setup_test_config():
    """Set up test configuration for all tests.

    This fixture runs automatically for all tests and ensures:
    - Pytest profile is loaded (fake models, memory cache, etc.)
    - Test environment is properly configured
    """
    switch_profile(PYTEST_PROFILE)

    yield


@pytest.fixture
def fake_llm() -> BaseChatModel:
    """Provide a fake LLM instance for testing.

    Returns:
        BaseChatModel: Fake LLM instance using parrot_local_fake
    """
    return get_llm(llm=FAKE_LLM_ID)


@pytest.fixture
def fake_embeddings():
    """Provide a fake embeddings instance for testing.

    Returns:
        Embeddings: Fake embeddings instance using embeddings_768_fake
    """
    return get_embeddings(embeddings=FAKE_EMBEDDINGS_ID)


@pytest.fixture
def fake_llm_with_streaming():
    """Provide a fake LLM instance with streaming enabled.

    Returns:
        BaseChatModel: Fake LLM instance with streaming
    """
    return get_llm(llm=FAKE_LLM_ID, streaming=True)


@pytest.fixture
def fake_llm_with_json_mode():
    """Provide a fake LLM instance with JSON mode enabled.

    Returns:
        BaseChatModel: Fake LLM instance with JSON mode
    """
    return get_llm(llm=FAKE_LLM_ID, json_mode=True)


@pytest.fixture
def fake_llm_with_cache():
    """Provide a fake LLM instance with caching enabled.

    Returns:
        BaseChatModel: Fake LLM instance with memory cache
    """
    return get_llm(llm=FAKE_LLM_ID, cache="memory")


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing.

    Returns:
        List[Document]: List of sample documents with metadata
    """
    from tests.utils.test_data import generate_sample_documents

    return generate_sample_documents(5)


@pytest.fixture
def sample_texts():
    """Provide sample text strings for testing.

    Returns:
        List[str]: List of sample text strings
    """
    from tests.utils.test_data import generate_sample_texts

    return generate_sample_texts(5)


@pytest.fixture
def sample_queries():
    """Provide sample search queries for testing.

    Returns:
        List[str]: List of sample search queries
    """
    from tests.utils.test_data import generate_sample_queries

    return generate_sample_queries()


@pytest.fixture
def temp_test_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary directory for test files.

    Args:
        tmp_path: pytest's temporary path fixture

    Returns:
        Path to temporary test directory
    """
    test_dir = tmp_path / "test_data"
    test_dir.mkdir(exist_ok=True)
    yield test_dir


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


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--include-real-models",
        action="store_true",
        default=False,
        help="Include tests that require real AI models (may incur costs and require API keys)",
    )
    parser.addoption(
        "--include-docker",
        action="store_true",
        default=False,
        help="Include tests that require Docker (starts real containers)",
    )
    parser.addoption(
        "--performance-tests", action="store_true", default=False, help="Include performance benchmark tests"
    )
