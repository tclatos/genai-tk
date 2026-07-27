"""Sandbox configuration loader.

Reads ``config/sandbox.yaml`` via the OmegaConf singleton and returns
typed Pydantic models.

Example:
    ```python
    from genai_tk.agents.sandbox.config import load_sandbox_config, get_docker_aio_settings

    cfg = load_sandbox_config()
    print(cfg.default)  # "local"

    aio = get_docker_aio_settings()
    print(aio.image, aio.opensandbox_server_url)
    ```
"""

from __future__ import annotations

from pathlib import Path

from genai_tk.agents.sandbox.models import (
    DockerAioSettings,
    DockerSmolSettings,
    E2bSandboxSettings,
    SandboxConfig,
)
from genai_tk.config_mgmt.config_mngr import global_config

_SANDBOX_YAML_KEY = "sandbox"
_SANDBOX_YAML_FILE = "basic/sandbox.yaml"

# execd image used by the minimal generated server config (single source of
# truth shared by ``cli sandbox start`` and ``AioSandboxBackend._ensure_server``).
_OPENSANDBOX_EXECD_IMAGE = "opensandbox/execd:v1.0.16"


def _default_allowed_host_paths() -> list[str]:
    """Return host-path prefixes to permit for bind mounts in a dev/test server.

    Covers the user home (where projects/skills usually live), ``/tmp``, and the
    configured project root so skill directories mount without per-path config.
    """
    paths: set[str] = {str(Path.home()), "/tmp"}
    try:
        from genai_tk.config_mgmt.config_mngr import paths_config  # noqa: PLC0415

        paths.add(str(paths_config().project))
    except Exception:
        pass
    return sorted(p for p in paths if p)


def write_server_config(
    port: int,
    *,
    path: Path | None = None,
    execd_image: str = _OPENSANDBOX_EXECD_IMAGE,
    allowed_host_paths: list[str] | None = None,
) -> Path:
    """Write a minimal opensandbox-server TOML config bound to *port*.

    The server is launched with ``SANDBOX_CONFIG_PATH`` pointing at the result
    so it does not depend on ``~/.sandbox.toml`` (which may be missing or bound
    to the wrong port) and boots non-interactively. Pass an explicit *path* for
    a long-lived daemon so ``cli sandbox stop`` can remove it; omit it for a
    concurrency-safe tempfile used by the short-lived test/agent server.

    Args:
        port: Port the opensandbox-server listens on.
        path: Optional explicit config destination.
        execd_image: execd image used by the docker runtime.
        allowed_host_paths: Host-path prefixes permitted for bind mounts. Defaults
            to the user home, ``/tmp``, and the project root — enough for local
            dev/test skill mounts. Pass an explicit list to override.

    Returns:
        Path of the written config.
    """
    import os  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    if path is None:
        fd, name = tempfile.mkstemp(suffix=".toml")
        os.close(fd)
        target = Path(name)
    else:
        target = path
    target.parent.mkdir(parents=True, exist_ok=True)
    # ``host`` must be a resolvable bind address: the opensandbox-server example
    # uses the loopback address, but a literal placeholder (e.g. asterisks) makes
    # uvicorn fail with ``Name or service not known`` on a fresh start, so the
    # loopback IP is resolved from ``localhost`` at write time.
    # ``[storage].allowed_host_paths`` gates bind mounts: opensandbox-server
    # (>=0.2) rejects every mount when the list is empty, and ``["/"]`` matches
    # nothing (the prefix check is ``path == p or path.startswith(p + "/")``),
    # so default to home + ``/tmp`` + project root.
    if allowed_host_paths is None:
        allowed_host_paths = _default_allowed_host_paths()
    import socket  # noqa: PLC0415

    bind_host = socket.gethostbyname("localhost")
    quoted = ", ".join(f'"{p}"' for p in allowed_host_paths)
    target.write_text(
        f'[server]\nhost = "{bind_host}"\nport = {port}\n\n'
        f'[runtime]\ntype = "docker"\nexecd_image = "{execd_image}"\n\n'
        f"[storage]\nallowed_host_paths = [{quoted}]\n"
    )
    return target


def load_sandbox_config() -> SandboxConfig:
    """Load and validate the unified sandbox configuration.

    Reads the ``sandbox`` section from the global OmegaConf config (which
    includes ``config/sandbox.yaml``).  Falls back to defaults when the
    section is absent.

    Returns:
        Validated ``SandboxConfig`` instance.
    """
    try:
        return global_config().section(_SANDBOX_YAML_KEY, SandboxConfig)
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
