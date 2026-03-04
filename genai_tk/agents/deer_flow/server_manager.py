"""Deer-flow server lifecycle management.

Starts, stops, and restarts the LangGraph server and Gateway API as subprocesses.

``restart()`` is the primary entry point for ``cli agents deerflow``: it performs
the same sequence as ``make clean && make dev`` — kill processes, stop sandbox
containers, clear logs, then start fresh.  This guarantees a freshly-written
``config.yaml`` is always loaded into a clean server.

Logs are written to ``<deer_flow_path>/logs/`` (same location as ``make dev``)
so they can be inspected with standard tools after launch.

Usage:
```python
from genai_tk.agents.deer_flow.server_manager import DeerFlowServerManager

mgr = DeerFlowServerManager(deer_flow_path="/path/to/deer-flow")
await mgr.restart()  # stop everything, clean, start fresh
await mgr.stop()  # stop if we own the processes
```
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

# Poll interval when waiting for readiness (seconds)
_POLL_INTERVAL = 2.0
# Default startup timeout — langgraph dev can be slow on first run
_DEFAULT_TIMEOUT = 90.0
# Hosts that must bypass any HTTP proxy (corporate proxies break localhost).
_NO_PROXY_HOSTS = "localhost,127.0.0.1,::1,0.0.0.0"
# Sandbox container name prefix (must match deer-flow Makefile)
_SANDBOX_PREFIX = "deer-flow-sandbox"


class DeerFlowServerManager:
    """Start, stop, and restart the LangGraph + Gateway servers.

    ``restart()`` mirrors ``make clean && make dev``:
    - kills any running LangGraph / uvicorn / nginx processes
    - stops all sandbox Docker containers
    - clears log files
    - starts LangGraph and Gateway fresh from the backend directory

    Logs go to ``<deer_flow_path>/logs/`` so they are easy to tail.
    """

    def __init__(
        self,
        deer_flow_path: str | None = None,
        langgraph_url: str = "http://localhost:2024",
        gateway_url: str = "http://localhost:8001",
        start_timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._deer_flow_path = deer_flow_path or os.environ.get("DEER_FLOW_PATH", "")
        self._lg_url = langgraph_url.rstrip("/")
        self._gw_url = gateway_url.rstrip("/")
        self._start_timeout = start_timeout
        self._lg_proc: subprocess.Popen | None = None
        self._gw_proc: subprocess.Popen | None = None
        self._lg_log: Path | None = None
        self._gw_log: Path | None = None
        self._owns_servers = False  # True when we launched them

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def is_running(self) -> bool:
        """Return True if both servers are already reachable."""
        return await self._check_url(self._lg_url + "/info") and await self._check_url(self._gw_url + "/api/models")

    async def restart(self) -> None:
        """Full clean restart — equivalent to ``make clean && make dev``.

        Steps:
        1. Kill all LangGraph / uvicorn processes (any owner, like ``make stop``)
        2. Stop all sandbox Docker containers with the deer-flow prefix
        3. Delete log files
        4. Start LangGraph and Gateway fresh
        """
        backend_path = self._resolve_backend_path()
        df_root = backend_path.parent

        # 1 — kill server processes (same targets as make stop)
        logger.info("Stopping server processes...")
        for pattern in ("langgraph dev", "uvicorn src.gateway.app:app"):
            subprocess.run(["pkill", "-f", pattern], capture_output=True)
        await asyncio.sleep(1)
        # force-kill any stragglers
        for pattern in ("langgraph dev", "uvicorn src.gateway.app:app"):
            subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)
        await asyncio.sleep(0.5)

        # 2 — stop sandbox Docker containers
        cleanup_script = df_root / "scripts" / "cleanup-containers.sh"
        if cleanup_script.exists():
            logger.info("Stopping sandbox containers...")
            subprocess.run(
                ["bash", str(cleanup_script), _SANDBOX_PREFIX],
                capture_output=True,
                cwd=str(df_root),
            )
        else:
            # Fallback: direct docker stop
            try:
                result = subprocess.run(
                    ["docker", "ps", "-q", "--filter", f"name={_SANDBOX_PREFIX}"],
                    capture_output=True,
                    text=True,
                )
                for cid in result.stdout.strip().splitlines():
                    subprocess.run(["docker", "stop", cid], capture_output=True)
            except Exception as e:
                logger.warning(f"Could not stop sandbox containers: {e}")

        # 3 — clean up log files (same as make clean)
        log_dir = df_root / "logs"
        log_dir.mkdir(exist_ok=True)
        for log_file in log_dir.glob("*.log"):
            try:
                log_file.unlink()
            except OSError:
                pass
        logger.info(f"Logs cleared: {log_dir}")

        # 4 — start fresh
        await self._start_servers(backend_path, log_dir)

    async def start(self) -> None:
        """Start servers if not already running (no cleanup).

        Prefer ``restart()`` to guarantee a clean state with fresh config.
        """
        if await self.is_running():
            logger.info("Deer-flow servers already running")
            return
        backend_path = self._resolve_backend_path()
        df_root = backend_path.parent
        log_dir = df_root / "logs"
        log_dir.mkdir(exist_ok=True)
        await self._start_servers(backend_path, log_dir)

    async def stop(self) -> None:
        """Terminate managed subprocesses if we started them."""
        if not self._owns_servers:
            return
        for proc, name in [(self._lg_proc, "LangGraph"), (self._gw_proc, "Gateway")]:
            if proc is None:
                continue
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=5)
                    logger.info(f"{name} server stopped (PID {proc.pid})")
                except Exception as e:
                    logger.warning(f"Could not stop {name} server: {e}")
                    try:
                        proc.kill()
                    except Exception:
                        pass
        self._lg_proc = None
        self._gw_proc = None
        self._owns_servers = False

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> DeerFlowServerManager:
        await self.restart()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_root_path(self) -> Path:
        """Return the resolved deer-flow root directory."""
        if not self._deer_flow_path:
            raise RuntimeError(
                "DEER_FLOW_PATH is not set and deer_flow_path was not provided. "
                "Set the DEER_FLOW_PATH environment variable to the deer-flow clone directory."
            )
        path = Path(self._deer_flow_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Deer-flow root not found at {path}")
        return path

    def _resolve_backend_path(self) -> Path:
        """Return the resolved deer-flow backend directory."""
        root = self._resolve_root_path()
        backend = (root / "backend").resolve()
        if not backend.exists():
            raise FileNotFoundError(f"Deer-flow backend not found at {backend}")
        return backend

    async def _start_servers(self, backend_path: Path, log_dir: Path) -> None:
        """Launch LangGraph and Gateway processes and wait for readiness."""
        logger.info(f"Starting Deer-flow servers from {backend_path}")

        env = {**os.environ}
        existing_no_proxy = env.get("no_proxy", env.get("NO_PROXY", ""))
        merged = ",".join(filter(None, [existing_no_proxy, _NO_PROXY_HOSTS]))
        env["no_proxy"] = merged
        env["NO_PROXY"] = merged

        self._lg_log = log_dir / "langgraph.log"
        self._gw_log = log_dir / "gateway.log"

        lg_out = open(self._lg_log, "w")  # noqa: WPS515
        logger.info(f"LangGraph log: {self._lg_log}")
        self._lg_proc = subprocess.Popen(
            ["uv", "run", "langgraph", "dev", "--no-browser", "--allow-blocking", "--no-reload", "--port", "2024"],
            cwd=str(backend_path),
            env=env,
            stdout=lg_out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        logger.debug(f"LangGraph server PID: {self._lg_proc.pid}")

        gw_out = open(self._gw_log, "w")  # noqa: WPS515
        logger.info(f"Gateway log: {self._gw_log}")
        self._gw_proc = subprocess.Popen(
            ["uv", "run", "uvicorn", "src.gateway.app:app", "--host", "0.0.0.0", "--port", "8001"],
            cwd=str(backend_path),
            env=env,
            stdout=gw_out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        logger.debug(f"Gateway server PID: {self._gw_proc.pid}")

        self._owns_servers = True
        await self._wait_for_ready()
        logger.info("Deer-flow servers are ready")

    async def _check_url(self, url: str) -> bool:
        """Return True if the URL responds with a non-5xx status."""
        try:
            transport = httpx.AsyncHTTPTransport()
            async with httpx.AsyncClient(timeout=3.0, transport=transport) as client:
                r = await client.get(url)
                return r.status_code < 500
        except Exception:
            return False

    async def _wait_for_ready(self) -> None:
        """Poll both servers until ready or timeout."""
        deadline = time.monotonic() + self._start_timeout

        while time.monotonic() < deadline:
            # Check for early process exit
            for proc, name, log, health_url in [
                (self._lg_proc, "LangGraph", self._lg_log, self._lg_url + "/info"),
                (self._gw_proc, "Gateway", self._gw_log, self._gw_url + "/api/models"),
            ]:
                if proc is None or proc.poll() is None:
                    continue  # still running or not started by us

                log_text = log.read_text(errors="replace") if (log and log.exists()) else ""

                if "address already in use" in log_text:
                    if await self._check_url(health_url):
                        logger.info(f"{name} port already in use — using existing instance")
                        if name == "LangGraph":
                            self._lg_proc = None
                        else:
                            self._gw_proc = None
                        break
                    break

                raise RuntimeError(
                    f"{name} server exited with code {proc.returncode}.\n"
                    f"Log ({log}):\n{log_text[-3000:]}\n\n"
                    "Check DEER_FLOW_PATH and that all dependencies are installed."
                )

            lg_ok = await self._check_url(self._lg_url + "/info")
            gw_ok = await self._check_url(self._gw_url + "/api/models")

            if lg_ok and gw_ok:
                return

            elapsed = time.monotonic() - (deadline - self._start_timeout)
            logger.debug(
                f"Waiting ({elapsed:.0f}s)  LangGraph={'✓' if lg_ok else '…'}  Gateway={'✓' if gw_ok else '…'}"
            )
            await asyncio.sleep(_POLL_INTERVAL)

        raise TimeoutError(
            f"Deer-flow servers did not become ready within {self._start_timeout:.0f}s.\n"
            f"Check logs:\n  LangGraph: {self._lg_log}\n  Gateway:   {self._gw_log}"
        )
