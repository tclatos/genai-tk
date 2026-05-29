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
from typing import Any

import httpx
from loguru import logger

from genai_tk.utils.singleton import once


def _ensure_no_proxy(host: str, env: dict[str, str] | None = None) -> None:
    """Add *host* to NO_PROXY so it bypasses any configured HTTP proxy.

    When *env* is None the current process environment is updated in-place.
    When *env* is provided (e.g. a subprocess env dict) it is updated instead.
    """
    target = env if env is not None else os.environ
    for key in ("NO_PROXY", "no_proxy"):
        existing = target.get(key, "")
        entries = [e.strip() for e in existing.split(",") if e.strip()]
        for bypass in (host, "localhost", "127.0.0.1"):
            if bypass not in entries:
                entries.append(bypass)
        target[key] = ",".join(entries)


def _get_prefect_config() -> dict[str, Any]:
    """Return the ``prefect:`` config section from global config, or an empty dict."""
    try:
        from omegaconf import OmegaConf

        from genai_tk.utils.config_mngr import get_raw_config

        cfg = get_raw_config()
        raw = OmegaConf.select(cfg, "prefect", default=None)
        if raw is None:
            return {}
        return dict(OmegaConf.to_container(raw, resolve=True))  # type: ignore[arg-type]
    except Exception:
        return {}


class PrefectServer:
    """Manage the lifecycle of a local Prefect server process.

    Wraps ``prefect server start`` as a detached background process.  Tracks
    the process via a PID file so it can be stopped cleanly later.

    Do not instantiate directly — use :func:`prefect_server` to get the singleton.
    """

    def __init__(self) -> None:
        self._config = _get_prefect_config()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def host(self) -> str:
        return str(self._config.get("host", "127.0.0.1"))

    @property
    def port(self) -> int:
        return int(self._config.get("port", 4200))

    @property
    def api_url(self) -> str:
        configured = self._config.get("api_url")
        if configured:
            return str(configured)
        return f"http://{self.host}:{self.port}/api"

    @property
    def ui_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def _pid_file(self) -> Path:
        configured = self._config.get("pid_file")
        if configured:
            pid_path = Path(str(configured))
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            return pid_path
        # Fallback: data_root/.prefect/prefect.pid
        try:
            from genai_tk.utils.config_mngr import global_config

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
        auto_start = bool(self._config.get("auto_start", True))
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
