"""Unified sandbox module for all agent frameworks.

Provides sandbox configuration models, config loaders, and the
``AioSandboxBackend`` (Docker-based deepagents ``SandboxBackendProtocol``).

Standard sandbox names used across all frameworks:

- ``local``  — no isolation, runs in host process (default)
- ``docker`` — Docker-based sandbox
- ``e2b``    — E2B cloud sandbox (SmolAgents only)
- ``wasm``   — WebAssembly/Pyodide (SmolAgents only, reserved)

Example:
    ```python
    from genai_tk.agents.sandbox import AioSandboxBackend, load_sandbox_config

    cfg = load_sandbox_config()
    aio_settings = cfg.docker.aio

    async with AioSandboxBackend(config=aio_settings) as backend:
        resp = await backend.aexecute("echo hello")
        print(resp.output)
    ```
"""

from genai_tk.agents.sandbox.aio_backend import AioSandboxBackend, SandboxToolResult
from genai_tk.agents.sandbox.config import (
    get_docker_aio_settings,
    get_docker_smol_settings,
    get_e2b_settings,
    load_sandbox_config,
    resolve_sandbox_name,
)
from genai_tk.agents.sandbox.models import (
    DockerAioSettings,
    DockerSandboxSettings,
    DockerSmolSettings,
    E2bSandboxSettings,
    SandboxConfig,
    WasmSandboxSettings,
)

__all__ = [
    # Backend
    "AioSandboxBackend",
    "SandboxToolResult",
    # Config loaders
    "load_sandbox_config",
    "get_docker_aio_settings",
    "get_docker_smol_settings",
    "get_e2b_settings",
    "resolve_sandbox_name",
    # Models
    "SandboxConfig",
    "DockerAioSettings",
    "DockerSmolSettings",
    "DockerSandboxSettings",
    "E2bSandboxSettings",
    "WasmSandboxSettings",
]
