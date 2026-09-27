"""Factory for instantiating sandbox backends conforming to SandboxBackendProtocol."""

from __future__ import annotations

from typing import Any

from loguru import logger

from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.config_mgmt.import_utils import ImportResolver

BUILTIN_SANDBOX_BACKENDS: dict[str, str] = {
    "aio_sandbox": "genai_tk.agents.sandbox.aio_backend.AioSandboxBackend",
    "docker": "genai_tk.agents.sandbox.aio_backend.AioSandboxBackend",
    "local": "deepagents.backends.local_shell.LocalShellBackend",
    "filesystem": "deepagents.backends.local_shell.LocalShellBackend",
}


class SandboxBackendFactory:
    """Factory for creating and resolving sandbox backend instances by name or configuration.

    Supports built-in backends (OpenSandbox Docker, LocalShell) as well as custom
    or third-party providers (E2B, Modal, Daytona) configured via qualified Python class names
    in YAML configuration (e.g. ``sandbox.backends.<name>.class``).
    """

    _registry: dict[str, str] = {}

    @classmethod
    def register(cls, name: str, class_path: str) -> None:
        """Register a custom sandbox backend class path by name."""
        cls._registry[name.lower()] = class_path
        logger.debug(f"Registered sandbox backend '{name}' -> {class_path}")

    @classmethod
    def get_backend_class_path(cls, name: str) -> str:
        """Resolve the qualified class path for a sandbox backend name."""
        normalized = name.strip().lower()

        # 1. Check in-memory registry
        if normalized in cls._registry:
            return cls._registry[normalized]

        # 2. Check YAML configuration under sandbox.backends.<name> or sandbox_backends.<name>
        for key in (f"sandbox.backends.{normalized}", f"sandbox_backends.{normalized}"):
            try:
                cfg = global_config().get_dict(key)
                if cfg and isinstance(cfg, dict):
                    cp = cfg.get("class") or cfg.get("class_path")
                    if cp:
                        return cp
            except Exception:
                pass

        # 3. Fall back to built-in mapping
        if normalized in BUILTIN_SANDBOX_BACKENDS:
            return BUILTIN_SANDBOX_BACKENDS[normalized]

        raise KeyError(
            f"Unknown sandbox backend '{name}'. Available built-in: {sorted(BUILTIN_SANDBOX_BACKENDS.keys())}"
        )

    @classmethod
    def create(cls, name: str = "docker", **kwargs: Any) -> Any:
        """Instantiate a sandbox backend by name with optional override parameters.

        Args:
            name: Sandbox backend name (e.g. 'docker', 'aio_sandbox', 'local', 'e2b', 'modal').
            **kwargs: Arguments passed to the backend constructor (e.g. config=..., work_dir=...).

        Returns:
            Configured backend instance conforming to SandboxBackendProtocol.
        """
        normalized = name.strip().lower()
        class_path = cls.get_backend_class_path(normalized)

        # Handle docker / aio_sandbox default settings if not provided
        if normalized in ("docker", "aio_sandbox") and "config" not in kwargs:
            from genai_tk.agents.sandbox.config import get_docker_aio_settings

            kwargs["config"] = get_docker_aio_settings()

        logger.debug(f"Instantiating sandbox backend '{normalized}' using class {class_path}")
        backend_cls = ImportResolver.import_from_qualified(class_path)
        return backend_cls(**kwargs)
