"""Test utilities for GenAI Toolkit.

This module provides shared constants, fixtures, and utilities for testing
with fake models to ensure fast, reliable, and cost-effective tests.
"""

from .constants import (
    FAKE_EMBEDDINGS_ID,
    FAKE_EMBEDDINGS_PROVIDER,
    FAKE_LLM_ID,
    FAKE_LLM_PROVIDER,
    PYTEST_CONFIG_NAME,
)
from .factories import FakeLLMResponseFactory, FakeTestDataFactory

__all__ = [
    "FAKE_LLM_ID",
    "FAKE_EMBEDDINGS_ID",
    "FAKE_LLM_PROVIDER",
    "FAKE_EMBEDDINGS_PROVIDER",
    "PYTEST_CONFIG_NAME",
    "FakeLLMResponseFactory",
    "FakeTestDataFactory",
]
