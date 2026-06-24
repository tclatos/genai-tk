"""Shared fixtures for agent integration tests."""

import pytest

from genai_tk.config_mgmt.test_config import PytestConfig


@pytest.fixture(scope="module")
def agent_fake_llm_id(test_cfg: PytestConfig) -> str:
    """Fake LLM ID for agent tests, module-scoped for performance."""
    return test_cfg.fake_llm
