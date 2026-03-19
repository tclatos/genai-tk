"""Deer-flow agent profile model.

Defines the Pydantic profile loaded from config/agents/deerflow.yaml.
Replaces the old DeerFlowAgentConfig dataclass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast, get_args

from loguru import logger
from pydantic import BaseModel, Field

from genai_tk.tools.tool_specs import ToolSpec

DeerFlowMode = Literal["flash", "thinking", "pro", "ultra"]
"""Valid agent reasoning-intensity modes."""

DeerFlowSandbox = Literal["local", "docker"]
"""Sandbox provider for code/file execution."""


class DeerFlowProfile(BaseModel):
    """Configuration for a single Deer-flow agent profile.

    Loaded from the ``deerflow_agents`` list in ``config/agents/deerflow.yaml``.
    """

    name: str
    description: str = ""
    mode: DeerFlowMode = "flash"
    llm: str | None = None
    tool_groups: list[str] = Field(default_factory=lambda: ["web"])
    mcp_servers: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    skill_directories: list[str] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    system_prompt: str | None = None

    # Embedded-client behaviour flags (can be set per-profile)
    subagent_enabled: bool = False
    plan_mode: bool = False

    sandbox: DeerFlowSandbox = "local"

    # --web mode: server lifecycle settings
    auto_start: bool = True
    deer_flow_path: str | None = None
    base_url: str = "http://localhost:2026"
    langgraph_url: str = "http://localhost:2024"
    gateway_url: str = "http://localhost:8001"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DeerFlowError(Exception):
    """Base exception for Deer-flow errors."""


class ProfileNotFoundError(DeerFlowError):
    """Raised when a requested profile name is not found."""

    def __init__(self, profile_name: str, available: list[str]) -> None:
        self.profile_name = profile_name
        self.available_profiles = available
        super().__init__(f"Profile '{profile_name}' not found. Available: {', '.join(available)}")


class MCPServerNotFoundError(DeerFlowError):
    """Raised when requested MCP server names are not configured."""

    def __init__(self, invalid: list[str], available: list[str] | None = None) -> None:
        self.invalid_servers = invalid
        self.available_servers = available
        msg = f"MCP server(s) not found: {', '.join(invalid)}"
        if available:
            msg += f". Available: {', '.join(available)}"
        super().__init__(msg)


class DockerSandboxError(DeerFlowError):
    """Raised when Docker sandbox prerequisites are not met."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        detail = "; ".join(reasons)
        super().__init__(
            f"Docker sandbox is not available: {detail}. "
            "Either install the missing dependencies and ensure Docker is running, "
            "or set 'sandbox: local' in the profile."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_available_modes() -> list[str]:
    """Return the list of valid agent modes."""
    return list(get_args(DeerFlowMode))


def validate_mode(mode: str) -> DeerFlowMode:
    """Validate and normalize a Deer-flow mode value.

    Args:
        mode: Raw mode string.

    Returns:
        Normalized mode value.

    Raises:
        ValueError: If mode is not one of the supported values.
    """
    normalized = mode.strip().lower()
    if normalized not in get_available_modes():
        raise ValueError(f"Invalid mode '{mode}'. Available: {', '.join(get_available_modes())}")
    return cast(DeerFlowMode, normalized)


def get_available_profile_names(profiles: list[DeerFlowProfile]) -> list[str]:
    """Return profile names from a list."""
    return [p.name for p in profiles]


def validate_profile_name(name: str, profiles: list[DeerFlowProfile]) -> DeerFlowProfile:
    """Find and return a profile by name (case-insensitive).

    Args:
        name: Profile name to look for.
        profiles: List of loaded profiles.

    Returns:
        The matching DeerFlowProfile.
    """
    for p in profiles:
        if p.name.lower() == name.lower():
            return p
    raise ProfileNotFoundError(name, get_available_profile_names(profiles))


def validate_mcp_servers(server_names: list[str]) -> list[str]:
    """Validate MCP server names against configuration.

    Args:
        server_names: Server names to validate.

    Returns:
        Validated server names.
    """
    if not server_names:
        return []

    from genai_tk.utils.config_mngr import global_config

    try:
        mcp_config = global_config().get("mcp", {})
        available = list((mcp_config.get("servers") or {}).keys())
    except Exception:
        available = []

    if not available:
        # Cannot validate without config — pass through
        return server_names

    invalid = [s for s in server_names if s not in available]
    if invalid:
        raise MCPServerNotFoundError(invalid, available)
    return server_names


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_deer_flow_profiles(config_path: str | None = None) -> list[DeerFlowProfile]:
    """Load Deer-flow profiles from a YAML file or directory.

    When *config_path* is ``None`` the loader looks for
    ``{paths.config}/agents/deerflow/`` (directory) first, then
    ``{paths.config}/agents/deerflow.yaml`` (single file).  In directory mode
    all ``*.yaml`` / ``*.yml`` files are loaded and their ``deerflow_agents``
    lists are concatenated in alphabetical file order.

    Args:
        config_path: Explicit path to a YAML file or directory.  ``None`` uses
            the default location derived from ``paths.config``.

    Returns:
        List of ``DeerFlowProfile`` instances.
    """
    from genai_tk.utils.config_mngr import load_yaml_configs, paths_config

    if config_path is None:
        agents_dir = paths_config().config / "agents"
        dir_path = agents_dir / "deerflow"
        path = dir_path if dir_path.is_dir() else agents_dir / "deerflow.yaml"
    else:
        path = Path(config_path)

    profiles: list[DeerFlowProfile] = load_yaml_configs(path, "deerflow_agents", model=DeerFlowProfile)  # type: ignore[assignment]
    logger.debug(f"Loaded {len(profiles)} Deer-flow profiles from {path}")
    return profiles
