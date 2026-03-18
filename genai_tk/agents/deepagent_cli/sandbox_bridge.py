"""Lifecycle management for the AioSandboxBackend inside deepagent-cli sessions.

Provides a single async context manager, ``sandbox_context``, that starts and
stops an :class:`~genai_tk.agents.sandbox.AioSandboxBackend` Docker container
when ``sandbox: docker`` (or legacy ``aio``) is configured, or yields ``None``
for all other sandbox values (where deepagents-cli manages the backend itself).

Standard sandbox name is ``"docker"``.  The legacy name ``"aio"`` is also
accepted for backward compatibility.

Example:
    ```python
    async with sandbox_context(profile, config) as backend:
        agent, da_backend = create_cli_agent(
            model=model,
            sandbox=backend,
            sandbox_type=effective_sandbox_type(profile, backend),
            ...
        )
    ```
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from genai_tk.agents.deepagent_cli.models import DeepagentConfig, DeepagentProfile
    from genai_tk.agents.sandbox.aio_backend import AioSandboxBackend

# Both names refer to the same AioSandboxBackend
_DOCKER_SANDBOX_TYPES = frozenset({"docker", "aio"})


@asynccontextmanager
async def sandbox_context(
    profile: DeepagentProfile,
    config: DeepagentConfig,
) -> AsyncIterator[AioSandboxBackend | None]:
    """Async context manager that starts an ``AioSandboxBackend`` when requested.

    When ``profile.sandbox`` is ``"docker"`` (or legacy ``"aio"``), merges the
    sandbox configuration (profile overrides shared config), starts the Docker
    container, and yields the running backend.  For all other sandbox values,
    yields ``None`` so deepagents-cli falls back to its built-in backends
    (local shell, Modal, Daytona, etc.).

    Args:
        profile: Active profile (provides ``sandbox`` and ``sandbox_config``).
        config: Global deepagent config (provides fallback ``sandbox_config``).

    Yields:
        Running ``AioSandboxBackend`` for ``docker``/``aio`` sandbox, or ``None`` otherwise.
    """
    if profile.sandbox not in _DOCKER_SANDBOX_TYPES:
        yield None
        return

    from genai_tk.agents.sandbox.aio_backend import AioSandboxBackend

    backend_config = _build_backend_config(profile, config)
    async with AioSandboxBackend(config=backend_config) as backend:
        yield backend


def effective_sandbox_type(
    profile: DeepagentProfile,
    sandbox_backend: AioSandboxBackend | None,
) -> str | None:
    """Return the ``sandbox_type`` string to pass to deepagents-cli.

    For ``docker``/``aio`` sandboxes this returns ``None`` — deepagents-cli
    does not know those types and the system prompt will be generated without a
    sandbox-specific section (fine since the profile usually overrides it).
    For all other recognised deepagents-cli sandbox types (``modal``,
    ``daytona``, ``runloop``, ``langsmith``) the original type string is
    forwarded unchanged.

    Args:
        profile: Active profile.
        sandbox_backend: Backend instance returned by ``sandbox_context``
            (``None`` when not running the Docker backend).

    Returns:
        Sandbox type string understood by deepagents-cli, or ``None``.
    """
    if sandbox_backend is not None:
        # Our custom Docker backend — deepagents-cli doesn't recognise it
        return None
    if profile.sandbox not in _DOCKER_SANDBOX_TYPES | {"none"}:
        return profile.sandbox
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_backend_config(
    profile: DeepagentProfile,
    config: DeepagentConfig,
):
    """Build a ``DockerAioSettings`` by merging shared config + global + profile overrides.

    Priority (highest to lowest):
    1. Profile-level ``sandbox_config`` fields (non-None values).
    2. Global deepagent config ``sandbox_config`` fields (non-None values).
    3. Shared ``sandbox.yaml`` docker.aio defaults.

    Args:
        profile: Active profile (may have ``sandbox_config``).
        config: Global deepagent config (may have ``sandbox_config``).

    Returns:
        Fully-resolved ``DockerAioSettings`` instance.
    """
    from genai_tk.agents.sandbox.config import get_docker_aio_settings

    # Start from shared sandbox.yaml settings
    base = get_docker_aio_settings().model_dump()

    # Layer global deepagent config on top (only fields explicitly set)
    if config.sandbox_config:
        explicit = config.sandbox_config.model_fields_set
        for k, v in config.sandbox_config.model_dump().items():
            if k not in explicit:
                continue
            if k == "env_vars":
                base["env_vars"] = {**base.get("env_vars", {}), **v}
            else:
                base[k] = v

    # Layer profile overrides on top (only fields explicitly set)
    if profile.sandbox_config:
        explicit = profile.sandbox_config.model_fields_set
        for k, v in profile.sandbox_config.model_dump().items():
            if k not in explicit:
                continue
            if k == "env_vars":
                base["env_vars"] = {**base.get("env_vars", {}), **v}
            else:
                base[k] = v

    from genai_tk.agents.sandbox.models import DockerAioSettings

    return DockerAioSettings(**base)
