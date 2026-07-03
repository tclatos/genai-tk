"""Unit tests for the harness registry (profile resolution across LangChain/DeerFlow)."""

import pytest

from genai_tk.agents.harness import create_harness, list_harness_profiles
from genai_tk.agents.harness.deerflow_harness import DeerFlowHarness
from genai_tk.agents.harness.langchain_harness import LangChainHarness


def test_list_harness_profiles_includes_both_harnesses() -> None:
    refs = list_harness_profiles()
    harnesses = {r.harness for r in refs}
    assert "langchain" in harnesses
    assert "deerflow" in harnesses


def test_create_harness_resolves_langchain_profile() -> None:
    harness = create_harness("research")
    assert isinstance(harness, LangChainHarness)
    assert harness.name == "langchain"


def test_create_harness_resolves_deerflow_profile() -> None:
    harness = create_harness("Web Browser")
    assert isinstance(harness, DeerFlowHarness)
    assert harness.name == "deerflow"


def test_create_harness_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="not found"):
        create_harness("definitely-not-a-real-profile")
