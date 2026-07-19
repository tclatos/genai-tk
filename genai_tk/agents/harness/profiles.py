"""Unified agent-profile model and loader for all harnesses.

Both LangChain (react | deep | custom — incl. DeepAgents SDK) and DeerFlow
profiles are presented as one discriminated-union type keyed by the
``harness`` field, so :func:`create_harness` and the CLI can do a single
dict lookup instead of probing two separate config trees.

Canonical source — a directory (or single file) holding every profile under
one ``agents:`` top-level dict, optionally with a ``agent_defaults:`` block
(langchain-only inheritable defaults)::

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

Resolution order: a project-level ``config/agents.yaml`` / ``config/agents/``
directory, then the bundled ``config/examples/agents/`` directory. The legacy
split form (``langchain_agents:`` dict + ``deerflow_agents:`` list) is no
longer supported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Union

from loguru import logger
from pydantic import BaseModel, Field, TypeAdapter

from genai_tk.agents.deer_flow.profile import DeerFlowProfile
from genai_tk.agents.langchain.config import AgentProfileConfig
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


def _contains_yaml_files(path: Path) -> bool:
    """Return True when *path* is a YAML file or a directory containing YAML files."""
    if path.is_file():
        return path.suffix.lower() in {".yaml", ".yml"}
    if not path.is_dir():
        return False
    return any(path.glob("*.yaml")) or any(path.glob("*.yml"))


def _resolve_unified_agents_path(config_path: str | None = None) -> Path | None:
    """Return the path to a unified ``agents:`` file/dir, or ``None``.

    Looks for, in order:
    - *config_path* if given explicitly (returned only if it exists),
    - ``{paths.config}/agents.yaml`` (project-specific single file),
    - ``{paths.config}/agents/`` (project-specific directory of files),
    - ``{paths.config}/examples/agents/`` (bundled directory of category files),
    - ``{paths.config}/examples/agents.yaml`` (bundled single file).
    """
    if config_path is not None:
        p = Path(config_path)
        return p if p.exists() else None
    cfg_dir = paths_config().config
    single = cfg_dir / "agents.yaml"
    if single.exists():
        return single
    agents_dir = cfg_dir / "agents"
    if _contains_yaml_files(agents_dir):
        return agents_dir
    examples_dir = cfg_dir / "examples" / "agents"
    if _contains_yaml_files(examples_dir):
        return examples_dir
    examples_single = cfg_dir / "examples" / "agents.yaml"
    if _contains_yaml_files(examples_single):
        return examples_single
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

    The ``agents:`` top-level dict is read from the resolved file or directory
    (see :func:`_resolve_unified_agents_path`). The optional ``agent_defaults:``
    block is applied to ``harness: langchain`` profiles before discriminated-union
    validation. Returns an empty profile set when no unified source is found.

    Args:
        config_path: Optional path to a YAML file/dir holding ``agents:``.
    """
    unified_path = _resolve_unified_agents_path(config_path)
    if unified_path is None:
        return {}, None, ""
    return _load_unified_file(unified_path)


def load_langchain_profiles(config_path: str | None = None) -> dict[str, AgentProfileConfig]:
    """Return all ``harness: langchain`` profiles keyed by slug (defaults applied).

    Convenience wrapper over :func:`load_agent_profiles` for callers that only
    care about LangChain profiles (the LangChain CLI, ``LangchainAgent``, the
    ReAct webapp page, the MCP agent tool). Defaults are already merged in.
    """
    profiles, _defaults, _ = load_agent_profiles(config_path)
    return {k: p for k, p in profiles.items() if p.harness == "langchain"}


def load_deerflow_profiles(config_path: str | None = None) -> list[DeerFlowProfile]:
    """Return all ``harness: deerflow`` profiles (fully self-describing).

    Convenience wrapper over :func:`load_agent_profiles` for callers that only
    care about DeerFlow profiles (the DeerFlow CLI ``--list``, the DeerFlow
    webapp page, the runtime profile preparation).
    """
    profiles, _defaults, _ = load_agent_profiles(config_path)
    return [p for p in profiles.values() if p.harness == "deerflow"]


def _load_unified_file(path: Path) -> tuple[dict[str, Any], AgentDefaultsConfig | None, str]:
    raw = load_yaml_configs(path, "agents")
    if not isinstance(raw, dict):
        raw = {}
    # Read optional defaults block (may live alongside `agents:` in any file).
    defaults_raw: dict[str, Any] | None = None
    try:
        loaded = load_yaml_configs(path, "agent_defaults", model=None)
        if isinstance(loaded, dict):
            defaults_raw = dict(loaded)
    except Exception:
        # No `agent_defaults:` key present anywhere — defaults stay None.
        defaults_raw = None
    defaults = AgentDefaultsConfig.model_validate(defaults_raw) if defaults_raw else None
    default_profile_key = (defaults.default_profile if defaults else "") or ""

    profiles: dict[str, Any] = {}
    for key, val in raw.items():
        if not isinstance(val, dict):
            continue
        if key in {"defaults", "default_profile"}:
            continue
        candidate = dict(val)
        # Default to langchain when the discriminator is omitted.
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
