"""Deer-flow agent profile model.

Defines the Pydantic profile loaded from config/agents/deerflow.yaml.
Replaces the old DeerFlowAgentConfig dataclass.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel, Field

VALID_MODES = ("flash", "thinking", "pro", "ultra")


class DeerFlowProfile(BaseModel):
    """Configuration for a single Deer-flow agent profile.

    Loaded from the ``deerflow_agents`` list in ``config/agents/deerflow.yaml``.
    """

    name: str
    description: str = ""
    mode: str = "flash"
    llm: str | None = None
    tool_groups: list[str] = Field(default_factory=lambda: ["web"])
    mcp_servers: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    skill_directories: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    system_prompt: str | None = None

    # HTTP client settings (can be overridden per-profile)
    base_url: str = "http://localhost:2026"
    langgraph_url: str = "http://localhost:2024"
    gateway_url: str = "http://localhost:8001"

    # Sandbox provider for code/file execution: "local" (no Docker) or "docker"
    sandbox: str = "local"

    # Server lifecycle
    auto_start: bool = True
    deer_flow_path: str | None = None


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


class InvalidModeError(DeerFlowError):
    """Raised when an invalid mode string is given."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.valid_modes = list(VALID_MODES)
        super().__init__(f"Invalid mode '{mode}'. Valid modes: {', '.join(VALID_MODES)}")


class MCPServerNotFoundError(DeerFlowError):
    """Raised when requested MCP server names are not configured."""

    def __init__(self, invalid: list[str], available: list[str] | None = None) -> None:
        self.invalid_servers = invalid
        self.available_servers = available
        msg = f"MCP server(s) not found: {', '.join(invalid)}"
        if available:
            msg += f". Available: {', '.join(available)}"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_available_modes() -> list[str]:
    """Return the list of valid agent modes."""
    return list(VALID_MODES)


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


def validate_mode(mode: str) -> str:
    """Validate and normalise a mode string.

    Args:
        mode: Raw mode string from user input.

    Returns:
        Lower-cased validated mode string.
    """
    normalised = mode.lower()
    if normalised not in VALID_MODES:
        raise InvalidModeError(mode)
    return normalised


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
        mcp_config = global_config().get_nested("mcp", {})
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
    """Load Deer-flow profiles from a YAML file.

    Args:
        config_path: Path to ``deerflow.yaml``. Defaults to ``config/agents/deerflow.yaml``
            resolved via the global config manager.

    Returns:
        List of DeerFlowProfile instances.
    """
    if config_path is None:
        from genai_tk.utils.config_mngr import global_config

        config_dir = global_config().get_dir_path("paths.config")
        config_path = str(config_dir / "agents" / "deerflow.yaml")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Deer-flow config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw or "deerflow_agents" not in raw:
        raise ValueError(f"Config {path} is missing 'deerflow_agents' section")

    profiles = []
    for entry in raw.get("deerflow_agents", []):
        profile = DeerFlowProfile(
            name=entry.get("name", "Unnamed"),
            description=entry.get("description", ""),
            mode=entry.get("mode", "flash"),
            llm=entry.get("llm"),
            tool_groups=entry.get("tool_groups", ["web"]),
            mcp_servers=entry.get("mcp_servers", []),
            skills=entry.get("skills", []),
            skill_directories=entry.get("skill_directories", []),
            features=entry.get("features", []),
            examples=entry.get("examples", []),
            system_prompt=entry.get("system_prompt"),
            base_url=entry.get("base_url", "http://localhost:2026"),
            langgraph_url=entry.get("langgraph_url", "http://localhost:2024"),
            gateway_url=entry.get("gateway_url", "http://localhost:8001"),
            auto_start=entry.get("auto_start", True),
            deer_flow_path=entry.get("deer_flow_path"),
        )
        profiles.append(profile)

    logger.debug(f"Loaded {len(profiles)} Deer-flow profiles from {config_path}")
    return profiles
