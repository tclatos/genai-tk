"""Pydantic models for the unified sandbox configuration.

All sandbox technology settings are defined here as a single source of truth.
Loaded from ``config/basic/sandbox.yaml`` via :func:`~genai_tk.agents.sandbox.config.load_sandbox_config`.

Example:
    ```python
    from genai_tk.agents.sandbox.config import load_sandbox_config

    cfg = load_sandbox_config()
    aio = cfg.docker.aio
    print(f"Docker image: {aio.image}  server: {aio.opensandbox_server_url}")
    ```
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DockerAioSettings(BaseModel):
    """AioSandboxBackend settings.

    Maps to the ``sandbox.docker.aio`` block in ``sandbox.yaml``.
    Requires a running ``opensandbox-server`` (``uv add opensandbox``).
    """

    image: str = "ghcr.io/agent-infra/sandbox:latest"
    startup_timeout: float = 60.0
    work_dir: str = "/home/user"
    env_vars: dict[str, str] = Field(default_factory=dict)
    opensandbox_server_url: str = "http://localhost:8080"
    entrypoint: list[str] = Field(default_factory=lambda: ["/opt/gem/run.sh"])


class DockerSmolSettings(BaseModel):
    """SmolAgents native Docker executor settings.

    Used only by SmolAgents (``executor_type="docker"``).
    Maps to the ``sandbox.docker.smolagents`` block in ``sandbox.yaml``.
    """

    image: str = "python:3.12-slim"
    mem_limit: str = "512m"
    cpu_quota: int = 50000
    pids_limit: int = 100


class DockerSandboxSettings(BaseModel):
    """Combined Docker sandbox settings grouping AioSandboxBackend and SmolAgents configs."""

    aio: DockerAioSettings = Field(default_factory=DockerAioSettings)
    smolagents: DockerSmolSettings = Field(default_factory=DockerSmolSettings)


class E2bSandboxSettings(BaseModel):
    """E2B cloud sandbox settings (SmolAgents only).

    Maps to the ``sandbox.e2b`` block in ``sandbox.yaml``.
    """

    api_key: str | None = None
    template: str | None = None
    timeout: int = 300


class WasmSandboxSettings(BaseModel):
    """WebAssembly / Pyodide sandbox settings (reserved slot).

    Maps to the ``sandbox.wasm`` block in ``sandbox.yaml``.
    Currently requires Deno installed on the host (SmolAgents ``executor_type="wasm"``).
    """

    enabled: bool = False


class SandboxConfig(BaseModel):
    """Top-level unified sandbox configuration.

    Loaded from the ``sandbox`` key in ``config/basic/sandbox.yaml``.
    """

    default: str = "local"
    docker: DockerSandboxSettings = Field(default_factory=DockerSandboxSettings)
    e2b: E2bSandboxSettings = Field(default_factory=E2bSandboxSettings)
    wasm: WasmSandboxSettings = Field(default_factory=WasmSandboxSettings)
