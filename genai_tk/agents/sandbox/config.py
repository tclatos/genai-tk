"""Sandbox configuration loader.

Reads ``config/basic/sandbox.yaml`` via the OmegaConf singleton and returns
typed Pydantic models.

Example:
    ```python
    from genai_tk.agents.sandbox.config import load_sandbox_config, get_docker_aio_settings

    cfg = load_sandbox_config()
    print(cfg.default)  # "local"

    aio = get_docker_aio_settings()
    print(aio.image, aio.host_port)
    ```
"""

from __future__ import annotations

from genai_tk.agents.sandbox.models import (
    DockerAioSettings,
    DockerSmolSettings,
    E2bSandboxSettings,
    SandboxConfig,
)

_SANDBOX_YAML_KEY = "sandbox"
_SANDBOX_YAML_FILE = "basic/sandbox.yaml"


def load_sandbox_config() -> SandboxConfig:
    """Load and validate the unified sandbox configuration.

    Reads the ``sandbox`` section from the global OmegaConf config (which
    includes ``config/basic/sandbox.yaml``).  Falls back to defaults when the
    section is absent.

    Returns:
        Validated ``SandboxConfig`` instance.
    """
    try:
        from omegaconf import OmegaConf

        from genai_tk.utils.config_mngr import global_config

        raw = global_config().get(_SANDBOX_YAML_KEY, {})
        if not raw:
            return SandboxConfig()
        if hasattr(raw, "_metadata"):
            raw = OmegaConf.to_container(raw, resolve=True)
        return SandboxConfig.model_validate(raw)
    except Exception:
        return SandboxConfig()


def get_docker_aio_settings() -> DockerAioSettings:
    """Return the AioSandboxBackend Docker settings from the shared config.

    Returns:
        Resolved ``DockerAioSettings`` instance.
    """
    return load_sandbox_config().docker.aio


def get_docker_smol_settings() -> DockerSmolSettings:
    """Return the SmolAgents Docker executor settings from the shared config.

    Returns:
        Resolved ``DockerSmolSettings`` instance.
    """
    return load_sandbox_config().docker.smolagents


def get_e2b_settings() -> E2bSandboxSettings:
    """Return the E2B sandbox settings from the shared config.

    Returns:
        Resolved ``E2bSandboxSettings`` instance.
    """
    return load_sandbox_config().e2b


def resolve_sandbox_name(name: str | None, framework_default: str | None = None) -> str:
    """Resolve a sandbox name, falling back through defaults.

    Priority: explicit ``name`` → ``framework_default`` → global config default → ``"local"``.

    Args:
        name: Explicitly requested sandbox name (may be ``None``).
        framework_default: Framework-level override (e.g. from a profile field).

    Returns:
        Resolved sandbox name string.
    """
    if name:
        return name
    if framework_default:
        return framework_default
    try:
        return load_sandbox_config().default
    except Exception:
        return "local"
