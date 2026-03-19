"""Pydantic models for deepagent-cli configuration.

Defines the profile and global config structures loaded from
``config/basic/agents/deepagent/`` (directory) or ``deepagent.yaml``.

Docker sandbox settings (image, host, port, etc.) are defined in
``config/basic/sandbox.yaml`` and loaded via
:mod:`genai_tk.agents.sandbox.config`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from genai_tk.agents.sandbox.models import DockerAioSettings

# Backward-compatible alias — prefer DockerAioSettings directly
AioSandboxConfig = DockerAioSettings


class DeepagentProfile(BaseModel):
    """A named preset that bundles deepagent-cli launch configuration.

    Loaded from the ``deepagent.profiles`` list in ``deepagent.yaml``.
    Any field left as ``None`` falls back to the parent ``DeepagentConfig`` default.
    """

    name: str
    description: str = ""
    llm: str | None = None
    auto_approve: bool = False
    enable_memory: bool = True
    enable_skills: bool = True
    enable_shell: bool = True
    shell_allow_list: list[str] = Field(default_factory=list)
    sandbox: str = "none"
    sandbox_config: DockerAioSettings | None = None
    system_prompt: str | None = None
    tools: list[str] = Field(default_factory=list)


class DeepagentConfig(BaseModel):
    """Global configuration for the deepagent CLI integration.

    Loaded from the ``deepagent`` key in ``config/basic/agents/deepagent.yaml``.
    """

    default_model: str | None = None
    default_profile: str | None = None
    auto_approve: bool = False
    enable_memory: bool = True
    enable_skills: bool = True
    enable_shell: bool = True
    shell_allow_list: list[str] = Field(default_factory=list)
    sandbox: str = "none"
    sandbox_config: DockerAioSettings | None = None
    system_prompt: str | None = None
    switcher_models: list[str] = Field(default_factory=list)
    profiles: list[DeepagentProfile] = Field(default_factory=list)

    def get_profile(self, name: str) -> DeepagentProfile | None:
        """Return the profile with the given name, or None if not found.

        Args:
            name: Profile name to look up (case-insensitive).
        """
        name_lower = name.lower()
        for profile in self.profiles:
            if profile.name.lower() == name_lower:
                return profile
        return None


def load_deepagent_config(config_path: str | Path | None = None) -> DeepagentConfig:
    """Load and validate the deepagent configuration from a YAML file or directory.

    When *config_path* is ``None`` the loader looks for
    ``{paths.config}/agents/deepagent/`` (directory) first, then
    ``{paths.config}/agents/deepagent.yaml`` (single file).  A directory allows
    splitting global settings and individual profiles into separate files — they
    are deep-merged with the ``profiles`` lists concatenated in alphabetical
    file order.

    Args:
        config_path: Explicit path to a YAML file or directory.  ``None`` uses
            the default location derived from ``paths.config``.

    Returns:
        Validated ``DeepagentConfig`` instance.

    Example:
        ```python
        config = load_deepagent_config()
        for profile in config.profiles:
            print(profile.name)
        ```
    """
    from genai_tk.utils.config_exceptions import yaml_config_validation
    from genai_tk.utils.config_mngr import load_yaml_configs, paths_config

    if config_path is None:
        agents_dir = paths_config().config / "agents"
        dir_path = agents_dir / "deepagent"
        path = dir_path if dir_path.is_dir() else agents_dir / "deepagent.yaml"
    else:
        path = Path(config_path)

    raw: dict = load_yaml_configs(path, "deepagent", list_merge_keys=["profiles"])  # type: ignore[assignment]

    with yaml_config_validation(file_path=str(path), context="deepagent config"):
        return DeepagentConfig.model_validate(raw)
