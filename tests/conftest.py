"""Pytest configuration and shared fixtures.

All test settings (fake model IDs, cache strategy, vector store backend) are
derived from the ``pytest`` profile in ``config/app_conf.yaml`` via the typed
``PytestConfig`` model.  Individual test files should **not** hardcode model IDs.
"""

from pathlib import Path
from typing import Generator

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from genai_tk.config_mgmt.config_mngr import switch_profile
from genai_tk.config_mgmt.features import is_available
from genai_tk.config_mgmt.test_config import PytestConfig, get_pytest_config
from genai_tk.core.factories.embeddings_factory import get_embeddings
from genai_tk.core.factories.llm_factory import get_llm

# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def setup_test_config():
    """Activate the ``pytest`` profile for all tests."""
    switch_profile("pytest")
    yield


@pytest.fixture(scope="session")
def test_cfg(setup_test_config) -> PytestConfig:
    """Typed test configuration derived from the active pytest profile."""
    return get_pytest_config()


# ---------------------------------------------------------------------------
# Fake model ID string fixtures (for tests that need the raw ID, e.g. CLI args)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fake_llm_id(test_cfg: PytestConfig) -> str:
    """Fake LLM model ID string (e.g. for CLI --llm arguments)."""
    return test_cfg.fake_llm


@pytest.fixture(scope="session")
def fake_embeddings_id(test_cfg: PytestConfig) -> str:
    """Fake embeddings model ID string (e.g. for CLI --model arguments)."""
    return test_cfg.fake_embeddings


# ---------------------------------------------------------------------------
# Fake model fixtures (resolved from typed config)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_llm(test_cfg: PytestConfig) -> BaseChatModel:
    """Fake LLM instance for testing."""
    return get_llm(llm=test_cfg.fake_llm)


@pytest.fixture
def fake_embeddings(test_cfg: PytestConfig):
    """Fake embeddings instance for testing."""
    return get_embeddings(embeddings=test_cfg.fake_embeddings)


@pytest.fixture
def fake_llm_with_streaming(test_cfg: PytestConfig):
    """Fake LLM instance with streaming enabled."""
    return get_llm(llm=test_cfg.fake_llm, streaming=True)


@pytest.fixture
def fake_llm_with_json_mode(test_cfg: PytestConfig):
    """Fake LLM instance with JSON mode enabled."""
    return get_llm(llm=test_cfg.fake_llm, json_mode=True)


@pytest.fixture
def fake_llm_with_cache(test_cfg: PytestConfig):
    """Fake LLM instance with caching enabled."""
    return get_llm(llm=test_cfg.fake_llm, cache="memory")


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    from tests.utils.test_data import generate_sample_documents

    return generate_sample_documents(5)


@pytest.fixture
def sample_texts():
    """Provide sample text strings for testing."""
    from tests.utils.test_data import generate_sample_texts

    return generate_sample_texts(5)


@pytest.fixture
def sample_queries():
    """Provide sample search queries for testing."""
    from tests.utils.test_data import generate_sample_queries

    return generate_sample_queries()


@pytest.fixture
def temp_test_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir(exist_ok=True)
    yield test_dir


# ---------------------------------------------------------------------------
# Utility fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool_definitions():
    """Provide sample tool definitions for testing."""
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


@pytest.fixture
def performance_threshold():
    """Maximum response time (seconds) for fake model tests."""
    return 1.0


@pytest.fixture
def invalid_llm_id():
    """An invalid LLM ID for error testing."""
    return "nonexistent_llm_model"


@pytest.fixture
def invalid_embeddings_id():
    """An invalid embeddings ID for error testing."""
    return "nonexistent_embeddings_model"
