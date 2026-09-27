"""Unified Sandbox Session Manager.

Manages shared sandbox lifecycles so containers and remote sandboxes are reused
across tool calls and turns without reloading, and provides context-aware binding for
both DeepAgents and DeerFlow harnesses.
"""

from __future__ import annotations

import asyncio
import contextvars
import shutil
import subprocess
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from genai_tk.utils.singleton import once

# Context variable to bind an active sandbox backend from the enclosing harness
active_sandbox_backend: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "active_sandbox_backend", default=None
)


class SandboxManager(BaseModel):
    """Singleton manager for shared sandbox backend instances across agent turns."""

    backend_type: str = Field(default="docker")
    config: Any | None = Field(default=None)
    backend: Any | None = Field(default=None, repr=False)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @once
    def singleton() -> SandboxManager:
        """Returns the thread-safe singleton instance of SandboxManager."""
        from genai_tk.agents.sandbox.config import get_docker_aio_settings

        cfg = get_docker_aio_settings()
        return SandboxManager(backend_type="docker", config=cfg)

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

    async def aget_backend(
        self,
        config: Any | None = None,
        backend_type: str | None = None,
    ) -> Any:
        """Get or lazily start the managed sandbox backend conforming to SandboxBackendProtocol.

        If a harness has bound an active backend into `active_sandbox_backend`,
        that backend is returned immediately.
        """
        # 1. Check context variable first (e.g. set by DeepAgentHarness / DeerFlow)
        ctx_backend = active_sandbox_backend.get()
        if ctx_backend is not None:
            if hasattr(ctx_backend, "start") and not getattr(ctx_backend, "_sandbox", None):
                await ctx_backend.start()
            return ctx_backend

        # 2. Re-use existing backend if active and healthy
        if self.backend is not None:
            if hasattr(self.backend, "_sandbox") and getattr(self.backend, "_sandbox", None) is not None:
                # If the event loop that created the backend was closed, reset it so we create a fresh one
                sandbox_obj = getattr(self.backend, "_sandbox", None)
                adapter = getattr(sandbox_obj, "command", None)
                client = getattr(adapter, "_client", None)
                if client is not None and getattr(client, "is_closed", False):
                    self.backend = None
                else:
                    return self.backend
            elif not hasattr(self.backend, "_sandbox"):
                return self.backend

        # 3. Create and start a new backend using SandboxBackendFactory
        from genai_tk.agents.sandbox.factory import SandboxBackendFactory

        b_type = backend_type or self.backend_type or "docker"
        cfg = config or self.config

        kwargs: dict[str, Any] = {}
        if cfg is not None:
            kwargs["config"] = cfg

        backend = SandboxBackendFactory.create(b_type, **kwargs)
        if hasattr(backend, "start"):
            logger.info(f"Starting shared {b_type} sandbox backend...")
            await backend.start()

        self.backend = backend
        return backend

    def get_backend(
        self,
        config: Any | None = None,
        backend_type: str | None = None,
    ) -> Any:
        """Synchronous wrapper to get or lazily start the managed sandbox backend."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.run(self.aget_backend(config, backend_type))
        return asyncio.run(self.aget_backend(config, backend_type))

    async def aclose_backend(self) -> None:
        """Stop and clean up the shared backend if running."""
        if self.backend is not None:
            backend = self.backend
            self.backend = None
            try:
                if hasattr(backend, "stop"):
                    await backend.stop()
                elif hasattr(backend, "close"):
                    await backend.close()
                logger.info("Shared sandbox backend stopped.")
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
        config: Any | None = None,
        backend_type: str | None = None,
    ) -> Any:
        """Get or lazily start the shared sandbox backend from singleton."""
        mgr = cls.singleton()
        return await mgr.aget_backend(config, backend_type)

    @classmethod
    def get_shared_backend(
        cls,
        config: Any | None = None,
        backend_type: str | None = None,
    ) -> Any:
        """Synchronous wrapper to get or lazily start the shared sandbox backend from singleton."""
        mgr = cls.singleton()
        return mgr.get_backend(config, backend_type)

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


# Backward compatibility alias
DockerSandboxManager = SandboxManager


def get_active_sandbox() -> Any | None:
    """Return the currently active sandbox backend from context or shared singleton."""
    ctx_backend = active_sandbox_backend.get()
    if ctx_backend is not None:
        return ctx_backend
    mgr = SandboxManager.singleton()
    return mgr.backend
