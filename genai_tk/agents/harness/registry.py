"""Harness registry — resolve a profile key against a single unified profile
dict and build the matching :class:`~genai_tk.agents.harness.base.BaseHarness`.

Profiles live under one ``agents:`` YAML dict keyed by profile slug/name,
discriminated by the ``harness`` field on each profile model, and loaded by
:func:`~genai_tk.agents.harness.profiles.load_agent_profiles`.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel

from genai_tk.agents.harness.base import BaseHarness
from genai_tk.agents.harness.profiles import load_agent_profiles


class HarnessProfileRef(BaseModel):
    """A profile discovered across either harness config tree."""

    key: str
    harness: str
    name: str
    description: str = ""
    llm: str | None = None


def _profile_ref(key: str, profile) -> HarnessProfileRef:
    return HarnessProfileRef(
        key=key,
        harness=profile.harness,
        name=profile.name,
        description=getattr(profile, "description", "") or "",
        llm=getattr(profile, "llm", None),
    )


def list_harness_profiles() -> list[HarnessProfileRef]:
    """Return all agent profiles as one list, across both harnesses."""
    try:
        profiles, _defaults, _default_key = load_agent_profiles()
    except Exception as e:
        logger.debug(f"Could not load agent profiles: {e}")
        return []
    return [_profile_ref(key, p) for key, p in profiles.items()]


def create_harness(
    key: str,
    *,
    llm_override: str | None = None,
    force_memory_checkpointer: bool = False,
) -> BaseHarness:
    """Resolve a profile key against the unified profile dict and build its harness.

    Single dict lookup keyed by profile slug (LangChain) or profile name
    (DeerFlow). Matching is case-insensitive.

    Args:
        key: Profile key/slug or name.
        llm_override: LLM identifier overriding the profile's configured LLM.
        force_memory_checkpointer: LangChain-only; use an in-process ``MemorySaver``.

    Returns:
        A ready-to-stream :class:`BaseHarness` instance.

    Raises:
        ValueError: If no profile matches *key*.
    """
    profiles, _defaults, _default_key = load_agent_profiles()
    profile = _lookup(profiles, key)
    if profile is None:
        available = list(profiles.keys())
        raise ValueError(f"Profile '{key}' not found. Available profiles ({len(available)}): {available}")

    if profile.harness == "langchain":
        from genai_tk.agents.harness.langchain_harness import LangChainHarness

        return LangChainHarness(profile, llm_override=llm_override, force_memory_checkpointer=force_memory_checkpointer)
    if profile.harness == "deerflow":
        from genai_tk.agents.harness.deerflow_harness import DeerFlowHarness

        return DeerFlowHarness(profile.name, llm_override=llm_override)
    raise ValueError(f"Unknown harness '{profile.harness}' for profile '{key}'")


def _lookup(profiles: dict, key: str):
    """Case-insensitive single-dict profile lookup."""
    if key in profiles:
        return profiles[key]
    lowered = key.lower()
    for k, p in profiles.items():
        if k.lower() == lowered or getattr(p, "name", "").lower() == lowered:
            return p
    return None
