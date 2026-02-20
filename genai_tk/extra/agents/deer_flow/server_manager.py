"""Deer-flow server lifecycle management.

Starts and stops the LangGraph server and Gateway API as subprocesses.
Detects if they are already running and skips launch if so.

The LangGraph server must be started from inside the deer-flow backend
directory (``<deer_flow_path>/backend``) so it can find its ``langgraph.json``.

Usage:
```python
from genai_tk.extra.agents.deer_flow.server_manager import DeerFlowServerManager

mgr = DeerFlowServerManager(deer_flow_path="/path/to/deer-flow")
await mgr.start()             # no-op if already running
await mgr.stop()

async with DeerFlowServerManager(...) as mgr:
    ...                       # servers are up inside the block
```
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

# Poll interval when waiting for readiness (seconds)
_POLL_INTERVAL = 2.0
# Default startup timeout — langgraph dev can be slow on first run
_DEFAULT_TIMEOUT = 60.0
# Hosts that must bypass any HTTP proxy (corporate proxies break localhost).
_NO_PROXY_HOSTS = "localhost,127.0.0.1,::1,0.0.0.0"


class DeerFlowServerManager:
    """Start, poll, and stop the LangGraph + Gateway servers.

    If both servers are already reachable when ``start()`` is called, no
    subprocesses are launched and ``stop()`` is a no-op.
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

    async def start(self) -> None:
        """Ensure both servers are running.

        Checks LangGraph and Gateway independently — only launches whichever
        is not yet reachable.  If both are already up, nothing happens.
        """
        lg_ok = await self._check_url(self._lg_url + "/info")
        gw_ok = await self._check_url(self._gw_url + "/api/models")

        if lg_ok and gw_ok:
            logger.info("Deer-flow servers already running — skipping launch")
            return

        backend_path = self._resolve_backend_path()
        logger.info(f"Starting Deer-flow servers from {backend_path}")

        env = {**os.environ}  # inherit full environment
        # Ensure localhost traffic is never routed through a corporate proxy.
        existing_no_proxy = env.get("no_proxy", env.get("NO_PROXY", ""))
        merged = ",".join(filter(None, [existing_no_proxy, _NO_PROXY_HOSTS]))
        env["no_proxy"] = merged
        env["NO_PROXY"] = merged

        log_dir = Path(tempfile.gettempdir())
        self._lg_log = log_dir / "deer_flow_langgraph.log"
        self._gw_log = log_dir / "deer_flow_gateway.log"

        # --- LangGraph server (only if not already up) ---
        if not lg_ok:
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
        else:
            logger.info("LangGraph already running — skipping")

        # --- Gateway server (only if not already up) ---
        if not gw_ok:
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
        else:
            logger.info("Gateway already running — skipping")

        if self._lg_proc is not None or self._gw_proc is not None:
            self._owns_servers = True
            await self._wait_for_ready()

        logger.info("Deer-flow servers are ready")

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
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_backend_path(self) -> Path:
        """Return the resolved backend/ path inside deer-flow."""
        if not self._deer_flow_path:
            raise RuntimeError(
                "DEER_FLOW_PATH is not set and deer_flow_path was not provided. "
                "Set the DEER_FLOW_PATH environment variable to the deer-flow clone directory."
            )
        path = Path(self._deer_flow_path).expanduser().resolve() / "backend"
        if not path.exists():
            raise FileNotFoundError(f"Deer-flow backend not found at {path}")
        return path

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
                    # Port taken by an existing instance — use it if healthy
                    if await self._check_url(health_url):
                        logger.info(f"{name} port already in use — using existing instance")
                        if name == "LangGraph":
                            self._lg_proc = None
                        else:
                            self._gw_proc = None
                        break  # re-evaluate outer condition
                    # Port taken but not responding yet — keep waiting
                    break

                # Genuine failure
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
            f"Check logs:\n  LangGraph: {self._lg_log}\n  Gateway:   {self._gw_log}\n"
            "Or run `make dev` and `make gateway` in the deer-flow backend directory."
        )
