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

from genai_tk.agents.sandbox.aio_backend import AioSandboxBackend
from genai_tk.agents.sandbox.models import DockerAioSettings

# Context variable to bind an active sandbox backend from the enclosing harness
active_sandbox_backend: contextvars.ContextVar[AioSandboxBackend | None] = contextvars.ContextVar(
    "active_sandbox_backend", default=None
)


class DockerSandboxManager:
    """Singleton manager for shared Docker sandbox backend instances."""

    _shared_backend: AioSandboxBackend | None = None
    _lock: asyncio.Lock | None = None

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

    @classmethod
    async def aget_shared_backend(
        cls,
        config: DockerAioSettings | None = None,
    ) -> AioSandboxBackend:
        """Get or lazily start the shared AioSandboxBackend.

        If a harness has bound an active backend into `active_sandbox_backend`,
        that backend is returned immediately.
        """
        # 1. Check context variable first (e.g. set by DeepAgentHarness / DeerFlow)
        ctx_backend = active_sandbox_backend.get()
        if ctx_backend is not None:
            if not getattr(ctx_backend, "_sandbox", None):
                await ctx_backend.start()
            return ctx_backend

        # 2. Re-use existing shared backend if active and healthy
        if cls._shared_backend is not None:
            if getattr(cls._shared_backend, "_sandbox", None) is not None:
                return cls._shared_backend

        # 3. Create and start a new shared backend
        from genai_tk.agents.sandbox.aio_backend import AioSandboxBackend
        from genai_tk.agents.sandbox.config import get_docker_aio_settings

        cfg = config or get_docker_aio_settings()
        backend = AioSandboxBackend(config=cfg)
        logger.info("Starting shared Docker AioSandboxBackend...")
        await backend.start()
        cls._shared_backend = backend
        return backend

    @classmethod
    def get_shared_backend(
        cls,
        config: DockerAioSettings | None = None,
    ) -> AioSandboxBackend:
        """Synchronous wrapper to get or lazily start the shared AioSandboxBackend."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Running inside an existing event loop: run in executor or nest
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.run(cls.aget_shared_backend(config))
        return asyncio.run(cls.aget_shared_backend(config))

    @classmethod
    async def aclose_shared_backend(cls) -> None:
        """Stop and clean up the shared backend if running."""
        if cls._shared_backend is not None:
            backend = cls._shared_backend
            cls._shared_backend = None
            try:
                await backend.stop()
                logger.info("Shared Docker AioSandboxBackend stopped.")
            except Exception as exc:
                logger.debug(f"Error stopping shared sandbox: {exc}")

    @classmethod
    def close_shared_backend(cls) -> None:
        """Synchronously stop the shared backend."""
        if cls._shared_backend is not None:
            try:
                asyncio.run(cls.aclose_shared_backend())
            except Exception as exc:
                logger.debug(f"Error closing shared backend sync: {exc}")


def get_active_sandbox() -> AioSandboxBackend | None:
    """Return the currently active sandbox backend from context or shared singleton."""
    ctx_backend = active_sandbox_backend.get()
    if ctx_backend is not None:
        return ctx_backend
    return DockerSandboxManager._shared_backend
