"""Pydantic models for deepagent-cli configuration.

Defines the profile and global config structures loaded from
``config/basic/agents/deepagent.yaml`` via OmegaConf.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Keep in sync with AioSandboxBackendConfig defaults
_SANDBOX_IMAGE_DEFAULT = "ghcr.io/tclatos/agent-sandbox:latest"


class AioSandboxConfig(BaseModel):
    """Optional Docker-based sandbox configuration for the ``aio`` sandbox type.

    Mirrors ``AioSandboxBackendConfig`` so all settings can be overridden from
    ``deepagent.yaml`` without importing the backend at config-load time.  Any
    field left ``None`` falls back to the ``AioSandboxBackendConfig`` built-in
    default when the backend is instantiated.
    """

    image: str | None = None
    host: str | None = None
    host_port: int | None = None
    startup_timeout: float | None = None
    work_dir: str | None = None
    env_vars: dict[str, str] = Field(default_factory=dict)


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
    sandbox_config: AioSandboxConfig | None = None
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
    sandbox_config: AioSandboxConfig | None = None
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


def load_deepagent_config() -> DeepagentConfig:
    """Load and validate the deepagent configuration from the global config.

    Reads the ``deepagent`` section from the OmegaConf singleton and validates
    it as a ``DeepagentConfig`` Pydantic model. Falls back to defaults if the
    section is absent.

    Returns:
        Validated ``DeepagentConfig`` instance.

    Example:
        ```python
        config = load_deepagent_config()
        if config.default_model:
            print(f"Default model: {config.default_model}")
        for profile in config.profiles:
            print(f"  Profile: {profile.name}")
        ```
    """
    from genai_tk.utils.config_mngr import global_config

    try:
        raw = global_config().get("deepagent", {})
        if not raw:
            return DeepagentConfig()
        # OmegaConf returns a DictConfig — convert to a plain dict first
        from omegaconf import OmegaConf

        if hasattr(raw, "_metadata"):  # is OmegaConf node
            raw = OmegaConf.to_container(raw, resolve=True)
        return DeepagentConfig.model_validate(raw)
    except Exception:
        return DeepagentConfig()
