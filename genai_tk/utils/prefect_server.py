"""Prefect server lifecycle management — start/stop/status for local development.

Provides a singleton :class:`PrefectServer` that manages a local ``prefect server start``
subprocess.  Use the ``cli prefect`` commands to control it interactively, or call
:func:`prefect_server` from Python code to get the singleton instance.

Configuration (``config/app_conf.yaml`` or any merged YAML)::

    prefect:
      api_url: "http://127.0.0.1:4200/api"
      host: "127.0.0.1"
      port: 4200
      pid_file: "${paths.data_root}/.prefect/prefect.pid"
      auto_start: true   # auto-start when workflows are executed
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
import webbrowser
from pathlib import Path

import httpx
from loguru import logger
from pydantic import BaseModel

from genai_tk.utils.singleton import once

# ---------------------------------------------------------------------------
# Pydantic config model
# ---------------------------------------------------------------------------


class PrefectConfig(BaseModel):
    """Typed configuration for the ``prefect:`` YAML section."""

    host: str = "127.0.0.1"
    port: int = 4200
    api_url: str | None = None
    pid_file: str | None = None
    auto_start: bool = True

    @property
    def resolved_api_url(self) -> str:
        if self.api_url:
            return self.api_url
        return f"http://{self.host}:{self.port}/api"


def _ensure_no_proxy(host: str, env: dict[str, str] | None = None) -> None:
    """Add *host* to NO_PROXY so it bypasses any configured HTTP proxy."""
    target = env if env is not None else os.environ
    for key in ("NO_PROXY", "no_proxy"):
        existing = target.get(key, "")
        entries = [e.strip() for e in existing.split(",") if e.strip()]
        for bypass in (host, "localhost", "127.0.0.1"):
            if bypass not in entries:
                entries.append(bypass)
        target[key] = ",".join(entries)


def _load_prefect_config() -> PrefectConfig:
    """Load the ``prefect:`` section as a typed :class:`PrefectConfig`."""
    try:
        from genai_tk.config_mgmt.config_mngr import global_config

        return global_config().section("prefect", PrefectConfig)
    except Exception:
        return PrefectConfig()


class PrefectServer:
    """Manage the lifecycle of a local Prefect server process.

    Wraps ``prefect server start`` as a detached background process.  Tracks
    the process via a PID file so it can be stopped cleanly later.

    Do not instantiate directly — use :func:`prefect_server` to get the singleton.
    """

    def __init__(self) -> None:
        self._config = _load_prefect_config()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def host(self) -> str:
        return self._config.host

    @property
    def port(self) -> int:
        return self._config.port

    @property
    def api_url(self) -> str:
        return self._config.resolved_api_url

    @property
    def ui_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def _pid_file(self) -> Path:
        if self._config.pid_file:
            pid_path = Path(self._config.pid_file)
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            return pid_path
        # Fallback: data_root/.prefect/prefect.pid
        try:
            from genai_tk.config_mgmt.config_mngr import global_config

            data_root = Path(str(global_config().paths.data_root))
        except Exception:
            data_root = Path.home() / ".cache" / "genai_tk"
        pid_dir = data_root / ".prefect"
        pid_dir.mkdir(parents=True, exist_ok=True)
        return pid_dir / "prefect.pid"

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        """Return True if the server process is alive and responding to health checks."""
        # Check PID file for a managed process
        pid = self._read_pid()
        if pid is not None:
            try:
                os.kill(pid, 0)  # signal 0: check existence only
            except (ProcessLookupError, PermissionError):
                # Process is gone — remove stale PID file
                self._pid_file.unlink(missing_ok=True)

        # HTTP health check (works for any server, managed or external)
        # Use a no-proxy client so corporate proxies don't intercept localhost traffic.
        try:
            client = httpx.Client(proxy=None, trust_env=False)
            resp = client.get(f"{self.api_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, *, foreground: bool = False) -> None:
        """Start the Prefect server.

        Args:
            foreground: When True, blocks the current process (useful for debug).
                When False (default), starts as a detached background daemon.
        """
        if self.is_running():
            logger.info("Prefect server already running at {}", self.ui_url)
            return

        cmd = [
            "prefect",
            "server",
            "start",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        if foreground:
            logger.info("Starting Prefect server in foreground at {}", self.ui_url)
            subprocess.run(cmd, check=False)
            return

        # Background daemon — detach from parent process group so it survives
        # the CLI process exiting.
        logger.info("Starting Prefect server at {} (background daemon)", self.ui_url)
        # Ensure localhost bypasses any corporate proxy in the subprocess environment
        spawn_env = dict(os.environ)
        _ensure_no_proxy(self.host, spawn_env)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=spawn_env,
        )
        self._write_pid(proc.pid)

        # Poll until ready (up to 45 seconds)
        for _ in range(90):
            time.sleep(0.5)
            if self.is_running():
                logger.info("Prefect server ready at {}", self.ui_url)
                return

        logger.warning(
            "Prefect server started but may not be fully ready yet. Check health at {}/health",
            self.api_url,
        )

    def stop(self) -> None:
        """Stop the managed Prefect server."""
        pid = self._read_pid()

        if pid is None:
            if not self.is_running():
                logger.info("Prefect server is not running")
            else:
                logger.warning(
                    "No PID file found — server may have been started outside genai-tk. "
                    "Stop it manually or with: prefect server stop"
                )
            return

        try:
            # Kill the entire process group to also stop child processes
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            logger.info("Sent SIGTERM to Prefect server process group (PID {})", pid)
        except ProcessLookupError:
            logger.info("Prefect server process {} not found (already stopped)", pid)
        except Exception as exc:
            logger.warning("Error stopping Prefect server: {}", exc)
        finally:
            self._pid_file.unlink(missing_ok=True)

    def ensure_running(self) -> None:
        """Start the server if ``auto_start`` is enabled and the server is not running.

        Called automatically by :func:`~genai_tk.workflow.executor.execute_workflow`
        when ``prefect.auto_start: true`` (the default).
        """
        auto_start = self._config.auto_start
        if not auto_start:
            return
        if not self.is_running():
            logger.info("Auto-starting Prefect server...")
            self.start()

    # ------------------------------------------------------------------
    # API URL configuration
    # ------------------------------------------------------------------

    def configure_api_url(self) -> None:
        """Set ``PREFECT_API_URL`` environment variable from config.

        Must be called before any Prefect flow is executed so the client
        connects to the correct server.  Also ensures localhost traffic
        bypasses any configured HTTP proxy.
        """
        os.environ["PREFECT_API_URL"] = self.api_url
        # Ensure localhost traffic is not routed through corporate proxies
        _ensure_no_proxy(self.host)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def open_ui(self) -> None:
        """Open the Prefect UI in the default browser."""
        webbrowser.open(self.ui_url)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_pid(self) -> int | None:
        try:
            return int(self._pid_file.read_text().strip())
        except Exception:
            return None

    def _write_pid(self, pid: int) -> None:
        self._pid_file.write_text(str(pid))


@once
def prefect_server() -> PrefectServer:
    """Return the global :class:`PrefectServer` singleton.

    Example:
        ```python
        from genai_tk.utils.prefect_server import prefect_server

        server = prefect_server()
        server.ensure_running()
        server.configure_api_url()
        ```
    """
    return PrefectServer()
