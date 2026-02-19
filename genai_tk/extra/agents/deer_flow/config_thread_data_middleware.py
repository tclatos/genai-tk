"""Config-based Thread Data Middleware - provides workspace paths and sandbox initialization.

This middleware replicates deer-flow's ThreadDataMiddleware functionality,
getting thread_id from ``runtime.context`` (a dict passed to ``agent.astream()``).

Enables Python-based file I/O skills (ppt-generation, chart-visualization, etc.)
and sandbox tools (read_file, write_file, bash, ls) in CLI, Streamlit, FastAPI,
or any other LangGraph application.

Also pre-initializes the local sandbox so that sandbox-based tools (read_file, write_file,
bash, ls) work without needing runtime.context for lazy acquisition.

Usage:
    ```python
    from genai_tk.extra.agents.deer_flow.config_thread_data_middleware import ConfigThreadDataMiddleware
    from genai_tk.extra.agents.deer_flow.agent import create_deer_flow_agent_simple

    # Create middleware
    thread_data_mw = ConfigThreadDataMiddleware()

    # Pass to agent creation
    agent = create_deer_flow_agent_simple(
        profile=profile,
        llm=llm,
        thread_data_middleware=thread_data_mw,
    )
    ```
"""

import os
from pathlib import Path
from typing import Any

from loguru import logger


class ConfigThreadDataMiddleware:
    """Provides workspace paths AND sandbox initialization for file I/O tools.

    Gets thread_id from ``runtime.context`` (a dict passed via
    ``agent.astream(inputs, config, context={"thread_id": ...})``).

    This middleware does TWO things:
    1. Sets ``thread_data`` in state with workspace/uploads/outputs paths
    2. Pre-initializes the local sandbox in state (``sandbox: {"sandbox_id": "local"}``)
       so that sandbox tools (read_file, write_file, bash, ls) find a ready sandbox
       and never call ``runtime.context.get("thread_id")`` (which would fail).

    Directory structure created:
        {base_dir}/threads/{thread_id}/user-data/
        ├── workspace/  # Main workspace for generated files
        ├── uploads/    # User uploaded files (future)
        └── outputs/    # Deprecated, use workspace

    This middleware is compatible with all deer-flow skills that use thread_data paths.
    """

    def __init__(self, base_dir: str | Path | None = None, lazy_init: bool = False) -> None:
        """Initialize the middleware.

        Args:
            base_dir: Base directory for thread workspaces.
                     Defaults to .deer-flow/threads in current working directory.
            lazy_init: If True, only compute paths without creating directories.
                      If False, create directories immediately.
                      Default False (create directories for reliability).
        """
        if base_dir is None:
            # Use same path structure as deer-flow native
            base_dir = Path.cwd() / ".deer-flow"
        self._base_dir = Path(base_dir)
        self._lazy_init = lazy_init

        # Create base directory
        self._base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"ConfigThreadDataMiddleware initialized with base_dir: {self._base_dir}")

    def _get_thread_paths(self, thread_id: str) -> dict[str, str]:
        """Compute paths for a thread's workspace directories.

        Args:
            thread_id: The thread/session ID

        Returns:
            Dictionary with workspace_path, uploads_path, outputs_path
        """
        thread_dir = self._base_dir / "threads" / thread_id / "user-data"

        return {
            "workspace_path": str(thread_dir / "workspace"),
            "uploads_path": str(thread_dir / "uploads"),
            "outputs_path": str(thread_dir / "outputs"),
        }

    def _create_thread_directories(self, thread_id: str) -> dict[str, str]:
        """Create workspace directories for a thread.

        Args:
            thread_id: The thread/session ID

        Returns:
            Dictionary with the created directory paths
        """
        paths = self._get_thread_paths(thread_id)

        # Create all directories
        for path_str in paths.values():
            os.makedirs(path_str, exist_ok=True)

        logger.debug(f"Created workspace directories for thread {thread_id}")
        return paths

    def before_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """Provide workspace paths and sandbox initialization to the agent.

        Gets thread_id from runtime.context (populated via ``agent.astream(inputs, config, context=...)``)
        or falls back to a default.

        Args:
            state: Current agent state
            runtime: LangGraph Runtime (context is a dict with thread_id when properly configured)

        Returns:
            Update dict with thread_data paths and sandbox state
        """
        # Try to get thread_id from runtime.context (dict passed to astream/invoke/stream)
        thread_id = None

        if hasattr(runtime, "context") and runtime.context:
            try:
                thread_id = runtime.context.get("thread_id")
            except (AttributeError, TypeError):
                pass

        # Fallback: use a default thread_id
        if not thread_id:
            thread_id = "default"
            logger.warning(
                f"Could not determine thread_id from runtime.context, using '{thread_id}'. "
                "Ensure context={{'thread_id': ...}} is passed to agent.astream()/invoke()."
            )

        # Create or compute paths
        if self._lazy_init:
            paths = self._get_thread_paths(thread_id)
            logger.debug(f"Computed workspace paths for thread {thread_id} (lazy mode)")
        else:
            paths = self._create_thread_directories(thread_id)
            logger.debug(f"Created workspace directories for thread {thread_id}")

        # Build state update with thread_data
        update: dict[str, Any] = {"thread_data": paths}

        # Pre-initialize the local sandbox so tools don't call runtime.context.get("thread_id")
        # This is the key fix: sandbox tools check state["sandbox"]["sandbox_id"] first,
        # and only fall back to runtime.context if sandbox is not in state.
        if "sandbox" not in state or state.get("sandbox") is None:
            try:
                from src.sandbox import get_sandbox_provider

                provider = get_sandbox_provider()
                sandbox_id = provider.acquire(thread_id)
                update["sandbox"] = {"sandbox_id": sandbox_id}
                logger.debug(f"Pre-initialized sandbox '{sandbox_id}' for thread {thread_id}")
            except Exception as e:
                logger.warning(f"Could not pre-initialize sandbox: {e}. File tools may not work.")

        return update

    def get_workspace_dir(self, thread_id: str) -> Path:
        """Get the workspace directory path for a thread.

        Useful for UI components or CLI tools that need to list/download files.

        Args:
            thread_id: The thread/session ID

        Returns:
            Path to the workspace directory
        """
        paths = self._get_thread_paths(thread_id)
        return Path(paths["workspace_path"])

    def list_workspace_files(self, thread_id: str) -> list[dict[str, Any]]:
        """List all files in a thread's workspace.

        Args:
            thread_id: The thread/session ID

        Returns:
            List of file info dicts with keys: name, path, size, modified, relative_path
        """
        workspace = self.get_workspace_dir(thread_id)

        if not workspace.exists():
            return []

        files = []
        for file_path in workspace.rglob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                files.append(
                    {
                        "name": file_path.name,
                        "path": str(file_path),
                        "relative_path": str(file_path.relative_to(workspace)),
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                    }
                )

        return sorted(files, key=lambda x: x["modified"], reverse=True)

    def cleanup_old_threads(self, max_age_days: int = 7) -> int:
        """Clean up workspace directories older than specified age.

        Args:
            max_age_days: Maximum age in days before cleanup

        Returns:
            Number of thread directories removed
        """
        import shutil
        import time

        threads_dir = self._base_dir / "threads"
        if not threads_dir.exists():
            return 0

        current_time = time.time()
        max_age_seconds = max_age_days * 86400
        removed_count = 0

        for thread_dir in threads_dir.iterdir():
            if not thread_dir.is_dir():
                continue

            # Check age
            dir_age = current_time - thread_dir.stat().st_mtime
            if dir_age > max_age_seconds:
                try:
                    shutil.rmtree(thread_dir)
                    removed_count += 1
                    logger.info(f"Cleaned up old thread directory: {thread_dir.name}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {thread_dir.name}: {e}")

        return removed_count
