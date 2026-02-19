"""Path setup for Deer-flow backend imports.

Deer-flow's backend is installed as a package dependency.
This module adds the deer-flow backend directory to sys.path and sets up
required environment variables.

Requires DEER_FLOW_PATH environment variable pointing to deer-flow root directory.
"""

import os
import sys
from pathlib import Path

from loguru import logger

# Cached path once resolved
_deer_flow_backend_path: Path | None = None


def get_deer_flow_backend_path() -> Path:
    """Get the deer-flow backend directory path from DEER_FLOW_PATH environment variable.

    Returns:
        Path to the deer-flow backend directory

    Raises:
        EnvironmentError: If DEER_FLOW_PATH is not set or invalid
    """
    global _deer_flow_backend_path
    if _deer_flow_backend_path is not None:
        return _deer_flow_backend_path

    env_path = os.environ.get("DEER_FLOW_PATH")
    if not env_path:
        raise EnvironmentError(
            "DEER_FLOW_PATH environment variable is not set.\n"
            "Please set it to the deer-flow root directory:\n"
            "  export DEER_FLOW_PATH=/path/to/deer-flow\n\n"
            "Example: export DEER_FLOW_PATH=/home/user/ext_prj/deer-flow"
        )

    backend_path = Path(env_path) / "backend"
    if not backend_path.exists():
        raise EnvironmentError(
            f"Deer-flow backend not found at: {backend_path}\n"
            f"DEER_FLOW_PATH is set to: {env_path}\n"
            "Please verify the path points to the deer-flow root directory."
        )

    if not (backend_path / "src" / "__init__.py").exists():
        raise EnvironmentError(
            f"Invalid deer-flow backend at: {backend_path}\n"
            "The backend/src/__init__.py file is missing.\n"
            "Please ensure deer-flow is properly installed."
        )

    _deer_flow_backend_path = backend_path.resolve()
    logger.debug(f"Deer-flow backend found at: {_deer_flow_backend_path}")
    return _deer_flow_backend_path


def setup_deer_flow_path() -> Path:
    """Add deer-flow backend to sys.path if not already present.

    This must be called before any `from src.xxx import ...` deer-flow imports.

    Returns:
        Path to the deer-flow backend directory
    """
    backend_path = get_deer_flow_backend_path()
    path_str = str(backend_path)

    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        logger.debug(f"Added deer-flow backend to sys.path: {path_str}")

    # Also set DEER_FLOW_CONFIG_PATH if not already set
    # (deer-flow config.yaml is expected in the parent directory)
    if "DEER_FLOW_CONFIG_PATH" not in os.environ:
        config_path = backend_path.parent / "config.yaml"
        if config_path.exists():
            os.environ["DEER_FLOW_CONFIG_PATH"] = str(config_path)
            logger.debug(f"Set DEER_FLOW_CONFIG_PATH to: {config_path}")

    return backend_path
