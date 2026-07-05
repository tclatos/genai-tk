"""Tests for the unified agent-profile loader (single ``agents:`` dict).

Covers both the unified ``agents.yaml`` form (discriminated by ``harness:``
key) and the legacy split fallback (``langchain_agents:`` dict +
``deerflow_agents:`` list). The unified loader is what
:func:`create_harness` and :func:`list_harness_profiles` now use.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_tk.agents.deer_flow.profile import DeerFlowProfile
from genai_tk.agents.harness import list_harness_profiles
from genai_tk.agents.harness.langchain_harness import LangChainHarness
from genai_tk.agents.harness.profiles import load_agent_profiles
from genai_tk.agents.langchain.config import AgentProfileConfig

_BUNDLED_AGENTS_YAML = Path(__file__).resolve().parents[4] / "config" / "examples" / "agents" / "agents.yaml"


def test_unified_loader_reads_bundled_agents_yaml() -> None:
    """The bundled unified ``agents.yaml`` loads with both harness variants."""
    profiles, defaults, default_key = load_agent_profiles(str(_BUNDLED_AGENTS_YAML))
    assert "simple" in profiles
    assert "research_deep" in profiles
    assert "Research Assistant" in profiles
    assert "Web Browser" in profiles

    assert isinstance(profiles["simple"], AgentProfileConfig)
    assert profiles["simple"].harness == "langchain"
    assert profiles["simple"].type == "react"

    assert isinstance(profiles["Research Assistant"], DeerFlowProfile)
    assert profiles["Research Assistant"].harness == "deerflow"
    assert profiles["Research Assistant"].mode == "pro"

    # defaults are read from the agent_defaults block
    assert defaults is not None
    assert defaults.default_profile == "simple"
    assert default_key == "simple"


def test_unified_loader_applies_langchain_defaults() -> None:
    """Langchain profiles inherit inheritable fields from agent_defaults."""
    profiles, defaults, _ = load_agent_profiles(str(_BUNDLED_AGENTS_YAML))
    assert defaults is not None
    # `simple` didn't specify middlewares; it inherits the default stack (the
    # profile's entries are parsed into MiddlewareConfig objects, while the
    # defaults hold the raw dict form, so compare by class_path).
    profile_mw = profiles["simple"].middlewares
    assert len(profile_mw) == len(defaults.middlewares)
    assert profile_mw[0].class_path.endswith("RichToolCallMiddleware")
    # `research_deep` didn't specify backend/checkpointer; should inherit defaults.
    lc = profiles["research_deep"]
    assert isinstance(lc, AgentProfileConfig)
    assert lc.checkpointer is not None
    assert lc.checkpointer.type == "none"


def test_unified_loader_exposes_deerflow_profiles_unchanged() -> None:
    """DeerFlow profiles in the unified file parse without langchain defaults applied."""
    profiles, _defaults, _ = load_agent_profiles(str(_BUNDLED_AGENTS_YAML))
    wb = profiles["Web Browser"]
    assert isinstance(wb, DeerFlowProfile)
    assert wb.mode == "flash"
    assert wb.sandbox == "local"
    assert wb.tool_groups == ["web"]


def test_registry_list_harness_profiles_works_without_unified_file() -> None:
    """The legacy split fallback still resolves profiles when no agents.yaml is preferred.

    In the repo there is no project-level ``config/agents.yaml``; the loader
    finds the bundled example OR falls back to the legacy split (langchain dir +
    deerflow.yaml). Either way, `list_harness_profiles()` returns both
    harnesses.
    """
    refs = list_harness_profiles()
    harnesses = {r.harness for r in refs}
    assert "langchain" in harnesses
    assert "deerflow" in harnesses


def test_create_harness_resolves_unified_lookup_case_insensitive(tmp_path: Path, monkeypatch) -> None:
    """``create_harness`` does one dict lookup against the unified profiles."""
    from genai_tk.agents.harness.registry import create_harness

    profiles, _defaults, _ = load_agent_profiles(str(_BUNDLED_AGENTS_YAML))

    import genai_tk.agents.harness.registry as registry_mod

    monkeypatch.setattr(
        registry_mod,
        "load_agent_profiles",
        lambda *a, **k: (profiles, None, "simple"),
        raising=True,
    )

    # No matching key → ValueError listing the unified profile set, no probing.
    with pytest.raises(ValueError):
        create_harness("nonexistent_profile_xyz")

    # Case-insensitive match for a langchain profile slug → LangChainHarness.
    h = create_harness("SIMPLE")
    assert isinstance(h, LangChainHarness)
    assert h.name == "langchain"
