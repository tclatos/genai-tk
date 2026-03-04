"""Lifecycle management for the AioSandboxBackend inside deepagent-cli sessions.

Provides a single async context manager, ``sandbox_context``, that starts and
stops an :class:`~genai_tk.agents.langchain.sandbox_backend.AioSandboxBackend`
Docker container when ``sandbox: aio`` is configured, or yields ``None`` for
all other sandbox values (where deepagents-cli manages the backend itself).

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
    from genai_tk.agents.deepagent.models import DeepagentConfig, DeepagentProfile
    from genai_tk.agents.langchain.sandbox_backend import AioSandboxBackend, AioSandboxBackendConfig

_AIO_SANDBOX_TYPE = "aio"


@asynccontextmanager
async def sandbox_context(
    profile: DeepagentProfile,
    config: DeepagentConfig,
) -> AsyncIterator[AioSandboxBackend | None]:
    """Async context manager that starts an ``AioSandboxBackend`` when requested.

    When ``profile.sandbox == "aio"``, merges the sandbox configuration
    (profile overrides global), starts the Docker container, and yields the
    running backend.  For all other sandbox values, yields ``None`` so
    deepagents-cli falls back to its built-in backends (local shell, Modal, etc.).

    Args:
        profile: Active profile (provides ``sandbox`` and ``sandbox_config``).
        config: Global deepagent config (provides fallback ``sandbox_config``).

    Yields:
        Running ``AioSandboxBackend`` for ``sandbox: aio``, or ``None`` otherwise.
    """
    if profile.sandbox != _AIO_SANDBOX_TYPE:
        yield None
        return

    from genai_tk.agents.langchain.sandbox_backend import AioSandboxBackend

    backend_config = _build_backend_config(profile, config)
    async with AioSandboxBackend(config=backend_config) as backend:
        yield backend


def effective_sandbox_type(
    profile: DeepagentProfile,
    sandbox_backend: AioSandboxBackend | None,
) -> str | None:
    """Return the ``sandbox_type`` string to pass to deepagents-cli.

    For ``aio`` sandboxes this returns ``None`` — deepagents-cli does not know
    the ``"aio"`` type and the system prompt will be generated without a
    sandbox-specific section (which is fine since the system prompt is usually
    overridden via the profile anyway).  For all other recognised deepagents-cli
    sandbox types (``modal``, ``daytona``, ``runloop``, ``langsmith``) the
    original type string is forwarded unchanged.

    Args:
        profile: Active profile.
        sandbox_backend: Backend instance returned by ``sandbox_context``
            (``None`` when not running ``aio``).

    Returns:
        Sandbox type string understood by deepagents-cli, or ``None``.
    """
    if sandbox_backend is not None:
        # Our custom backend — deepagents-cli doesn't recognise "aio"
        return None
    if profile.sandbox not in ("none", _AIO_SANDBOX_TYPE):
        return profile.sandbox
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_backend_config(
    profile: DeepagentProfile,
    config: DeepagentConfig,
) -> AioSandboxBackendConfig:
    """Merge global + profile sandbox configs and produce an ``AioSandboxBackendConfig``.

    Profile fields take priority. Only non-``None`` profile values override the
    global defaults so a partially-specified profile inherits the rest from
    the global ``sandbox_config``.

    Args:
        profile: Active profile (may have ``sandbox_config``).
        config: Global deepagent config (may have ``sandbox_config``).

    Returns:
        Fully-resolved ``AioSandboxBackendConfig`` instance.
    """
    from genai_tk.agents.langchain.sandbox_backend import AioSandboxBackendConfig

    # Start from global defaults
    base: dict = {}
    if config.sandbox_config:
        base = {k: v for k, v in config.sandbox_config.model_dump().items() if v is not None}

    # Layer profile overrides on top
    if profile.sandbox_config:
        for k, v in profile.sandbox_config.model_dump().items():
            if k == "env_vars":
                # Merge env dicts: global first, profile vars win on conflict
                merged_env = {**base.get("env_vars", {}), **v}
                base["env_vars"] = merged_env
            elif v is not None:
                base[k] = v

    return AioSandboxBackendConfig(**base)
