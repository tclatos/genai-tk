"""Unified Sandbox Session Manager.

Manages shared sandbox lifecycles so containers are reused across tool calls
and turns without reloading Docker, and provides context-aware binding for
both DeepAgents and DeerFlow harnesses.
"""

from __future__ import annotations

import asyncio
import contextvars
import shutil
import subprocess

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from genai_tk.agents.sandbox.aio_backend import AioSandboxBackend
from genai_tk.agents.sandbox.models import DockerAioSettings
from genai_tk.utils.singleton import once

# Context variable to bind an active sandbox backend from the enclosing harness
active_sandbox_backend: contextvars.ContextVar[AioSandboxBackend | None] = contextvars.ContextVar(
    "active_sandbox_backend", default=None
)


class DockerSandboxManager(BaseModel):
    """Singleton manager for shared Docker sandbox backend instances."""

    config: DockerAioSettings | None = Field(default=None)
    backend: AioSandboxBackend | None = Field(default=None, repr=False)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @once
    def singleton() -> DockerSandboxManager:
        """Returns the thread-safe singleton instance of DockerSandboxManager."""
        from genai_tk.agents.sandbox.config import get_docker_aio_settings

        cfg = get_docker_aio_settings()
        return DockerSandboxManager(config=cfg)

    @classmethod
    def is_docker_available(cls) -> bool:
        """Check if Docker CLI and daemon are available."""
        if not shutil.which("docker"):
            return False
        try:
            res = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3.0,
                check=False,
            )
            return res.returncode == 0
        except Exception:
            return False

    async def aget_backend(self, config: DockerAioSettings | None = None) -> AioSandboxBackend:
        """Get or lazily start the managed AioSandboxBackend.

        If a harness has bound an active backend into `active_sandbox_backend`,
        that backend is returned immediately.
        """
        # 1. Check context variable first (e.g. set by DeepAgentHarness / DeerFlow)
        ctx_backend = active_sandbox_backend.get()
        if ctx_backend is not None:
            if not getattr(ctx_backend, "_sandbox", None):
                await ctx_backend.start()
            return ctx_backend

        # 2. Re-use existing backend if active and healthy
        if self.backend is not None:
            if getattr(self.backend, "_sandbox", None) is not None:
                return self.backend

        # 3. Create and start a new backend
        from genai_tk.agents.sandbox.aio_backend import AioSandboxBackend
        from genai_tk.agents.sandbox.config import get_docker_aio_settings

        cfg = config or self.config or get_docker_aio_settings()
        self.config = cfg
        backend = AioSandboxBackend(config=cfg)
        logger.info("Starting shared Docker AioSandboxBackend...")
        await backend.start()
        self.backend = backend
        return backend

    def get_backend(self, config: DockerAioSettings | None = None) -> AioSandboxBackend:
        """Synchronous wrapper to get or lazily start the managed AioSandboxBackend."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.run(self.aget_backend(config))
        return asyncio.run(self.aget_backend(config))

    async def aclose_backend(self) -> None:
        """Stop and clean up the shared backend if running."""
        if self.backend is not None:
            backend = self.backend
            self.backend = None
            try:
                await backend.stop()
                logger.info("Shared Docker AioSandboxBackend stopped.")
            except Exception as exc:
                logger.debug(f"Error stopping shared sandbox: {exc}")

    def close_backend(self) -> None:
        """Synchronously stop the shared backend."""
        if self.backend is not None:
            try:
                asyncio.run(self.aclose_backend())
            except Exception as exc:
                logger.debug(f"Error closing shared backend sync: {exc}")

    @classmethod
    async def aget_shared_backend(
        cls,
        config: DockerAioSettings | None = None,
    ) -> AioSandboxBackend:
        """Get or lazily start the shared AioSandboxBackend from singleton."""
        mgr = cls.singleton()
        return await mgr.aget_backend(config)

    @classmethod
    def get_shared_backend(
        cls,
        config: DockerAioSettings | None = None,
    ) -> AioSandboxBackend:
        """Synchronous wrapper to get or lazily start the shared AioSandboxBackend from singleton."""
        mgr = cls.singleton()
        return mgr.get_backend(config)

    @classmethod
    async def aclose_shared_backend(cls) -> None:
        """Stop and clean up the shared backend if running."""
        mgr = cls.singleton()
        await mgr.aclose_backend()

    @classmethod
    def close_shared_backend(cls) -> None:
        """Synchronously stop the shared backend."""
        mgr = cls.singleton()
        mgr.close_backend()


def get_active_sandbox() -> AioSandboxBackend | None:
    """Return the currently active sandbox backend from context or shared singleton."""
    ctx_backend = active_sandbox_backend.get()
    if ctx_backend is not None:
        return ctx_backend
    mgr = DockerSandboxManager.singleton()
    return mgr.backend
