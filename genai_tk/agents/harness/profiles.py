"""Unified agent-profile model and loader for all harnesses.

Both LangChain (react | deep | custom — incl. DeepAgents SDK) and DeerFlow
profiles are presented as one discriminated-union type keyed by the
``harness`` field, so :func:`create_harness` and the CLI can do a single
dict lookup instead of probing two separate config trees.

Recommended canonical source — a single ``agents.yaml`` file (or directory)
holding every profile under one ``agents:`` top-level dict, optionally with a
``agent_defaults:`` block (langchain-only inheritable defaults)::

    agent_defaults:
      type: react
      middlewares:
        - class: genai_tk.agents.langchain.middleware.rich_middleware.RichToolCallMiddleware
      ...

    agents:
      research:
        harness: langchain
        name: "Research"
        type: deep
        ...
      "Research Assistant":
        harness: deerflow
        mode: pro
        ...

Legacy split form — ``langchain_agents:`` (dict + ``defaults:``) and
``deerflow_agents:`` (list) — is still supported as a fallback when no
``agents:`` file is present. New deployments should prefer ``agents.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Union

from loguru import logger
from pydantic import BaseModel, Field, TypeAdapter

from genai_tk.agents.deer_flow.profile import DeerFlowProfile, load_deer_flow_profiles
from genai_tk.agents.langchain.config import (
    AgentProfileConfig,
    LangchainAgentsConfig,
    load_unified_config,
    resolve_profile,
)
from genai_tk.config_mgmt.config_mngr import load_yaml_configs, paths_config

# Single discriminated-union profile model — the discriminator is the
# ``harness`` field on each variant. ``AgentProfileConfig.harness`` is always
# ``"langchain"`` and ``DeerFlowProfile.harness`` is always ``"deerflow"``.
AgentProfile = Annotated[
    Union[AgentProfileConfig, DeerFlowProfile],
    Field(discriminator="harness"),
]
"""Discriminated union of every agent-profile variant, keyed by ``harness``."""


class AgentDefaultsConfig(BaseModel):
    """Top-level wrapper for the optional ``agent_defaults:`` block (langchain-only).

    Holds the inheritable defaults applied to ``harness: langchain`` profiles
    when they live in the unified ``agents:`` dict ( DeerFlow has no inheritable
    defaults concept). Mirrors the legacy ``langchain_agents.defaults`` block.
    """

    type: Literal["react", "deep", "custom"] = "react"
    llm: str | None = None
    middlewares: list = Field(default_factory=list)
    checkpointer: dict[str, Any] | None = None
    backend: dict[str, Any] | None = None
    enable_planning: bool = True
    enable_file_system: bool = True
    skills: dict[str, Any] = Field(default_factory=dict)
    default_profile: str = ""
    model_config = {"extra": "allow"}


_adapter = TypeAdapter(AgentProfile)


def _resolve_unified_agents_path(config_path: str | None = None) -> Path | None:
    """Return the path to a unified ``agents.yaml`` file/dir, or ``None``.

    Looks for, in order:
    - ``config_path`` if given explicitly (returned only if it exists),
    - ``{paths.config}/agents.yaml`` (project-specific single file), or
    - ``{paths.config}/agents/`` (project-specific directory of files).

    The bundled ``config/examples/agents/agents.yaml`` is a *template* — it is
    **not** auto-preferred, so existing legacy profile sets under ``langchain_agents:`` /
    ``deerflow_agents:`` continue to resolve until a user adopts the unified
    canonical form by creating a project-level ``config/agents.yaml``.
    """
    if config_path is not None:
        p = Path(config_path)
        return p if p.exists() else None
    cfg_dir = paths_config().config
    single = cfg_dir / "agents.yaml"
    if single.exists():
        return single
    agents_dir = cfg_dir / "agents"
    if agents_dir.is_dir():
        return agents_dir
    return None


def _apply_langchain_defaults(profile_raw: dict, defaults: AgentDefaultsConfig | None) -> dict:
    """Apply langchain defaults to a raw langchain profile dict (shallow merge).

    Mirrors the field-by-field fallback of the legacy ``resolve_profile`` so a
    profile in the unified ``agents:`` dict inherits the same defaults without
    listing every field.
    """
    if defaults is None:
        return profile_raw
    merged = dict(profile_raw)
    merged.setdefault("type", defaults.type)
    if merged.get("llm") is None:
        merged["llm"] = defaults.llm
    merged.setdefault("middlewares", defaults.middlewares)
    if merged.get("checkpointer") is None and defaults.checkpointer is not None:
        merged["checkpointer"] = defaults.checkpointer
    if merged.get("backend") is None and defaults.backend is not None:
        merged["backend"] = defaults.backend
    merged.setdefault("enable_planning", defaults.enable_planning)
    merged.setdefault("enable_file_system", defaults.enable_file_system)
    if not merged.get("skill_directories") and defaults.skills:
        merged["skill_directories"] = list(defaults.skills.get("directories", []))
    return merged


def load_agent_profiles(
    config_path: str | None = None,
) -> tuple[dict[str, Any], AgentDefaultsConfig | None, str]:
    """Load all agent profiles into a single dict, discriminated by ``harness``.

    Returns a tuple ``(profiles, defaults, default_profile_key)`` where:
    - *profiles* maps the profile key/slug → validated :data:`AgentProfile`
      instance (``AgentProfileConfig`` or ``DeerFlowProfile``).
    - *defaults* is the optional langchain inheritable defaults block
      (``None`` when absent).
    - *default_profile_key* is the configured default profile key, or ``""``.

    Resolution order:
    1. A unified ``agents:`` top-level dict (from ``agents.yaml`` or an
       explicit *config_path*). Defaults block (``agent_defaults:``) is
       applied to langchain profiles before discriminated-union validation.
    2. Otherwise, the legacy split sources — LangChain's
       ``load_unified_config`` and DeerFlow's ``load_deer_flow_profiles`` —
       are read and flattened into the same dict so callers see one shape.

    Args:
        config_path: Optional path to a YAML file/dir holding ``agents:``.
    """
    unified_path = _resolve_unified_agents_path(config_path)
    if unified_path is not None:
        return _load_unified_file(unified_path)
    return _load_legacy_split()


def _load_unified_file(path: Path) -> tuple[dict[str, Any], AgentDefaultsConfig | None, str]:
    raw = load_yaml_configs(path, "agents")
    if not isinstance(raw, dict):
        raw = {}
    # Read optional defaults block (may live alongside `agents:`).
    defaults_raw: dict[str, Any] | None = None
    if path.is_file():
        loaded = load_yaml_configs(path, "agent_defaults", model=None)
        if isinstance(loaded, dict):
            defaults_raw = dict(loaded)
    defaults = AgentDefaultsConfig.model_validate(defaults_raw) if defaults_raw else None
    default_profile_key = (defaults.default_profile if defaults else "") or ""

    profiles: dict[str, Any] = {}
    for key, val in raw.items():
        if not isinstance(val, dict):
            continue
        if key in {"defaults", "default_profile"}:
            continue
        candidate = dict(val)
        # Default langchain if discriminator missing (backward-compat for
        # LangChain-style profiles migrated without an explicit `harness:` key).
        harness = candidate.get("harness", "langchain")
        if harness == "langchain":
            candidate = _apply_langchain_defaults(candidate, defaults)
        if "name" not in candidate:
            candidate["name"] = key
        try:
            profiles[key] = _adapter.validate_python(candidate)
        except Exception as e:
            logger.warning(f"Skipping agent profile '{key}': {e}")
    return profiles, defaults, default_profile_key


def _load_legacy_split() -> tuple[dict[str, Any], AgentDefaultsConfig | None, str]:
    profiles: dict[str, Any] = {}
    defaults: AgentDefaultsConfig | None = None
    default_profile_key = ""

    # LangChain profiles (defaults merged per-resolve_profile, then re-validated
    # into the union shape as a plain AgentProfileConfig — harness defaults to
    # langchain so the union discriminator resolves cleanly).
    try:
        from genai_tk.agents.langchain.commands import _get_config_path

        lc_cfg: LangchainAgentsConfig = load_unified_config(_get_config_path())
        default_profile_key = lc_cfg.default_profile
        for key in lc_cfg.profiles_dict:
            try:
                resolved = resolve_profile(lc_cfg, key)
                profiles[key] = resolved
            except Exception as e:
                logger.debug(f"LangChain profile '{key}' skipped: {e}")
    except Exception as e:
        logger.debug(f"LangChain profiles unavailable: {e}")

    # DeerFlow profiles (currently a list; key by profile name).
    try:
        from genai_tk.agents.deer_flow.runtime import resolve_deerflow_config_path

        for p in load_deer_flow_profiles(resolve_deerflow_config_path()):
            profiles[p.name] = p
    except Exception as e:
        logger.debug(f"DeerFlow profiles unavailable: {e}")

    return profiles, defaults, default_profile_key
