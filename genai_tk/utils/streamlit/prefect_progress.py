"""Prefect REST API polling helpers for Streamlit progress displays.

Provides lightweight wrappers around the Prefect server's HTTP API to query
flow run and task run states — without pulling in the full ``prefect`` SDK.
Uses the same ``httpx``-based proxy-bypass pattern as :mod:`genai_tk.utils.prefect_server`.

Example::

    from genai_tk.utils.streamlit.prefect_progress import PrefectPoller, FlowRunState

    poller = PrefectPoller()
    state = poller.get_flow_run(flow_run_id)
    tasks = poller.get_task_runs(flow_run_id)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx
from loguru import logger
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

TERMINAL_STATES = {"Completed", "Failed", "Crashed", "Cancelled"}


class TaskRunInfo(BaseModel):
    """Minimal info about a Prefect task run."""

    id: str
    name: str
    state_type: str  # "PENDING" | "RUNNING" | "COMPLETED" | "FAILED" | ...
    state_name: str  # Human-readable: "Pending", "Running", "Completed", ...
    start_time: str | None = None
    end_time: str | None = None


class FlowRunInfo(BaseModel):
    """Minimal info about a Prefect flow run."""

    id: str
    name: str
    state_type: str
    state_name: str
    start_time: str | None = None
    end_time: str | None = None
    task_runs: list[TaskRunInfo] = []

    @property
    def is_terminal(self) -> bool:
        return self.state_name in TERMINAL_STATES

    @property
    def is_completed(self) -> bool:
        return self.state_name == "Completed"

    @property
    def is_failed(self) -> bool:
        return self.state_name in ("Failed", "Crashed", "Cancelled")


# ---------------------------------------------------------------------------
# Poller
# ---------------------------------------------------------------------------


class PrefectPoller:
    """Thin HTTP client that polls the Prefect REST API.

    Reads ``prefect.api_url`` from genai-tk config (falls back to
    ``http://127.0.0.1:4200/api``).  Bypasses HTTP proxies for localhost.
    """

    def __init__(self, api_url: str | None = None) -> None:
        if api_url is None:
            api_url = self._resolve_api_url()
        self.api_url = api_url.rstrip("/")
        self._client = self._build_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_flow_run(self, flow_run_id: str) -> FlowRunInfo | None:
        """Fetch the current state of a flow run.

        Returns ``None`` if the run cannot be found (yet) or on any HTTP error.
        """
        try:
            resp = self._client.get(f"{self.api_url}/flow_runs/{flow_run_id}", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return self._parse_flow_run(data)
        except Exception as exc:
            logger.debug("Prefect get_flow_run error: {}", exc)
            return None

    def get_task_runs(self, flow_run_id: str) -> list[TaskRunInfo]:
        """Fetch all task runs belonging to *flow_run_id*.

        Returns an empty list on any error so callers can safely iterate.
        """
        try:
            resp = self._client.post(
                f"{self.api_url}/task_runs/filter",
                json={"flow_run_filter": {"id": {"any_": [flow_run_id]}}},
                timeout=10,
            )
            resp.raise_for_status()
            return [self._parse_task_run(t) for t in resp.json()]
        except Exception as exc:
            logger.debug("Prefect get_task_runs error: {}", exc)
            return []

    def poll_until_terminal(
        self,
        flow_run_id: str,
        *,
        interval: float = 2.0,
        timeout: float = 3600.0,
        on_update: "None | Callable[[FlowRunInfo], None]" = None,  # type: ignore[assignment]
    ) -> FlowRunInfo | None:
        """Blocking poll loop — waits until the flow run reaches a terminal state.

        Primarily for non-Streamlit callers.  In Streamlit, prefer calling
        :func:`get_flow_run` / :func:`get_task_runs` in your own loop so you
        can render UI updates between polls.

        Args:
            flow_run_id: The Prefect flow run UUID.
            interval: Seconds between polls.
            timeout: Maximum seconds to wait.
            on_update: Optional callback invoked with the latest FlowRunInfo after each poll.

        Returns:
            The final FlowRunInfo, or ``None`` if the run never appeared / timed out.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            info = self.get_flow_run(flow_run_id)
            if info is not None:
                info.task_runs = self.get_task_runs(flow_run_id)
                if on_update:
                    on_update(info)
                if info.is_terminal:
                    return info
            time.sleep(interval)
        logger.warning("Prefect poller timed out after {:.0f}s for flow run {}", timeout, flow_run_id)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_api_url() -> str:
        try:
            from genai_tk.config_mgmt.config_mngr import global_config
            from genai_tk.utils.prefect_server import PrefectConfig

            cfg: PrefectConfig = global_config().section("prefect", PrefectConfig)
            return cfg.resolved_api_url
        except Exception:
            return "http://127.0.0.1:4200/api"

    @staticmethod
    def _build_client() -> httpx.Client:
        """Build an httpx client that bypasses HTTP proxies for localhost."""
        import os

        # Ensure localhost is in NO_PROXY
        for key in ("NO_PROXY", "no_proxy"):
            existing = os.environ.get(key, "")
            entries = [e.strip() for e in existing.split(",") if e.strip()]
            for bypass in ("localhost", "127.0.0.1"):
                if bypass not in entries:
                    entries.append(bypass)
            os.environ[key] = ",".join(entries)

        return httpx.Client(
            headers={"Content-Type": "application/json"},
            timeout=15,
            follow_redirects=True,
        )

    @staticmethod
    def _parse_flow_run(data: dict[str, Any]) -> FlowRunInfo:
        state: dict = data.get("state") or {}
        return FlowRunInfo(
            id=data["id"],
            name=data.get("name", data["id"]),
            state_type=state.get("type", "UNKNOWN"),
            state_name=state.get("name", "Unknown"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
        )

    @staticmethod
    def _parse_task_run(data: dict[str, Any]) -> TaskRunInfo:
        state: dict = data.get("state") or {}
        return TaskRunInfo(
            id=data["id"],
            name=data.get("name", data["id"]),
            state_type=state.get("type", "UNKNOWN"),
            state_name=state.get("name", "Unknown"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
        )
