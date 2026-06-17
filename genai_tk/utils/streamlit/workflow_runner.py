"""Reusable 'run a workflow and show live Prefect progress' Streamlit component.

Runs a genai-tk YAML workflow in a background thread (so the Streamlit main thread
remains free to re-render) while polling the Prefect API for task states and
updating ``st.status`` / ``st.progress`` widgets on every refresh.

Usage in any Streamlit page::

    from genai_tk.utils.streamlit.workflow_runner import WorkflowRunner

    runner = WorkflowRunner(key="my_workflow")
    if runner.idle:
        if st.button("Run workflow"):
            runner.start("my_workflow_name", values={"source_dir": "/some/path"})
            st.rerun()

    runner.render_progress()

    if runner.completed:
        results = runner.results
        st.success("Done!")
    elif runner.failed:
        st.error(runner.error_message)
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any

from loguru import logger

import streamlit as st
from genai_tk.utils.streamlit.prefect_progress import FlowRunInfo, PrefectPoller, TaskRunInfo

# ---------------------------------------------------------------------------
# Module-level thread-safe store
# ---------------------------------------------------------------------------
# The background thread NEVER writes to st.session_state (no ScriptRunContext
# in non-main threads).  Instead it writes to this plain dict.  The main
# Streamlit thread calls _sync_from_thread_store() on every rerun to pull the
# latest data into session_state before rendering.
# ---------------------------------------------------------------------------
_thread_store: dict[str, dict[str, Any]] = {}

# State machine states
_IDLE = "idle"
_RUNNING = "running"
_COMPLETED = "completed"
_FAILED = "failed"

# Icons for task states
_STATE_ICONS: dict[str, str] = {
    "Pending": "⏳",
    "Running": "🔄",
    "Completed": "✅",
    "Failed": "❌",
    "Crashed": "💥",
    "Cancelled": "⛔",
    "Cached": "⚡",
    "Skipped": "⏭️",
}
_DEFAULT_ICON = "❓"

_POLL_INTERVAL = 2.0  # seconds between Prefect REST polls


class WorkflowRunner:
    """Stateful Streamlit component that runs a genai-tk workflow with live progress.

    State is stored in ``st.session_state`` under a namespaced key so multiple
    instances can coexist on the same page.  The background thread writes to a
    module-level ``_thread_store`` (never to ``session_state`` directly) to
    avoid the Streamlit ``ScriptRunContext`` thread safety warning.

    Args:
        key: Unique string key scoped to this runner instance.
        poll_interval: Seconds between Prefect API polls (default: 2.0).
    """

    def __init__(self, key: str = "workflow_runner", poll_interval: float = _POLL_INTERVAL) -> None:
        self._key = key
        self._poll_interval = poll_interval
        self._ensure_state()

    # ------------------------------------------------------------------
    # State accessors (read from session_state — always on the main thread)
    # ------------------------------------------------------------------

    @property
    def _state(self) -> dict[str, Any]:
        return st.session_state[self._key]

    @property
    def idle(self) -> bool:
        return self._state["status"] == _IDLE

    @property
    def running(self) -> bool:
        return self._state["status"] == _RUNNING

    @property
    def completed(self) -> bool:
        return self._state["status"] == _COMPLETED

    @property
    def failed(self) -> bool:
        return self._state["status"] == _FAILED

    @property
    def results(self) -> dict[str, Any] | None:
        """Results dict (step_id → result) when completed, else None."""
        return self._state.get("results")

    @property
    def error_message(self) -> str | None:
        return self._state.get("error")

    @property
    def flow_run_id(self) -> str | None:
        return self._state.get("flow_run_id")

    @property
    def task_runs(self) -> list[TaskRunInfo]:
        return self._state.get("task_runs") or []

    @property
    def flow_info(self) -> FlowRunInfo | None:
        return self._state.get("flow_info")

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(
        self,
        workflow_or_profile: str,
        *,
        values: dict[str, Any] | None = None,
        max_workers: int = 4,
    ) -> None:
        """Launch a named YAML workflow in a background thread.

        Args:
            workflow_or_profile: Workflow name or ``workflow/profile`` string.
            values: Optional parameter overrides.
            max_workers: Prefect thread-pool size.
        """
        if not self.idle and not self.failed:
            logger.warning("WorkflowRunner[{}]: already running, ignoring start()", self._key)
            return

        self._reset(status=_RUNNING)
        _thread_store[self._key] = {"status": _RUNNING}

        thread = threading.Thread(
            target=self._run_in_thread,
            args=(workflow_or_profile,),
            kwargs={"values": values or {}, "max_workers": max_workers},
            daemon=True,
            name=f"wf-runner-{self._key}",
        )
        self._state["thread"] = thread
        thread.start()

    def start_flow(self, flow_fn: "Any", *, flow_kwargs: "dict[str, Any] | None" = None) -> None:
        """Launch a pre-built Prefect ``@flow`` function in a background thread.

        Use this when you have a Prefect flow object directly (e.g. a demo or
        test flow) and don't need to go through the YAML workflow engine.

        Args:
            flow_fn: A Prefect ``@flow``-decorated callable.
            flow_kwargs: Keyword arguments forwarded to the flow when called.
        """
        if not self.idle and not self.failed:
            logger.warning("WorkflowRunner[{}]: already running, ignoring start_flow()", self._key)
            return

        self._reset(status=_RUNNING)
        _thread_store[self._key] = {"status": _RUNNING}

        thread = threading.Thread(
            target=self._run_flow_in_thread,
            args=(flow_fn,),
            kwargs={"flow_kwargs": flow_kwargs or {}},
            daemon=True,
            name=f"wf-runner-{self._key}",
        )
        self._state["thread"] = thread
        thread.start()

    def reset(self) -> None:
        """Reset to idle state so a new workflow can be started."""
        _thread_store.pop(self._key, None)
        self._reset(status=_IDLE)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_progress(self, *, auto_rerun: bool = True, rerun_interval: float = _POLL_INTERVAL) -> None:
        """Render live progress widgets.

        Syncs from the thread store first, then renders ``st.status`` with
        per-task progress.  When running, polls Prefect and schedules a page
        rerun every *rerun_interval* seconds so the display stays live.

        Args:
            auto_rerun: If True, calls ``st.rerun()`` while the workflow runs.
            rerun_interval: Seconds to wait between page reruns during execution.
        """
        # Always pull the latest data written by the background thread
        self._sync_from_thread_store()

        if self.idle:
            return

        status = self._state["status"]
        task_runs = self.task_runs
        flow_info = self.flow_info

        # Determine st.status label and state
        if status == _RUNNING:
            # Show the current running task in the label for better UX
            running_tasks = [t for t in task_runs if t.state_name == "Running"]
            if running_tasks:
                # Get the most recent running task
                running_tasks.sort(key=lambda t: t.start_time or "", reverse=True)
                current_task = running_tasks[0]
                # Strip Prefect hash suffix for cleaner display
                clean_name = re.sub(r"-[0-9a-f]{3,6}$", "", current_task.name)
                label = f"⏳ {clean_name}…"
            else:
                label = "⏳ Processing…"
            expanded = True
            state_arg = "running"
        elif status == _COMPLETED:
            label = "✅ Completed"
            expanded = False
            state_arg = "complete"
        else:
            label = "❌ Failed"
            expanded = True
            state_arg = "error"

        with st.status(label, expanded=expanded, state=state_arg):  # type: ignore[arg-type]
            if flow_info:
                try:
                    from genai_tk.utils.prefect_server import prefect_server

                    ui_base = prefect_server().ui_url
                except Exception:
                    ui_base = "http://127.0.0.1:4200"
                run_link = f"[{flow_info.name}]({ui_base}/runs/flow-run/{flow_info.id})"
                st.caption(f"Flow run: {run_link} — {flow_info.state_name}")
            elif status == _RUNNING:
                st.caption("Starting flow run…")

            # Accumulated task trace.
            # Deduplicate by logical task name (strip the trailing Prefect hash suffix
            # e.g. "fetch-data-4e4" -> "fetch-data") so fan-out tasks show as one row.
            # Show the worst state seen for that logical name.
            started = [t for t in task_runs if t.state_name not in ("Pending", "Scheduled")]
            started.sort(key=lambda t: (t.start_time or "", t.name))

            # Merge by logical name: keep most informative state (Running > Failed > Completed)
            _STATE_PRIORITY = {
                "Running": 0,
                "Failed": 1,
                "Crashed": 1,
                "Cancelled": 1,
                "Completed": 2,
                "Cached": 2,
                "Skipped": 3,
            }
            merged: dict[str, str] = {}  # logical_name -> state_name (best)
            seen_order: list[str] = []
            for t in started:
                # Strip trailing Prefect hash suffix: "fetch-data-4e4" -> "fetch-data"
                logical = re.sub(r"-[0-9a-f]{3,6}$", "", t.name)
                if logical not in merged:
                    merged[logical] = t.state_name
                    seen_order.append(logical)
                else:
                    # Keep the more interesting state
                    current_prio = _STATE_PRIORITY.get(merged[logical], 9)
                    new_prio = _STATE_PRIORITY.get(t.state_name, 9)
                    if new_prio < current_prio:
                        merged[logical] = t.state_name

            if merged:
                lines = "  \n".join(f"{_STATE_ICONS.get(merged[name], _DEFAULT_ICON)} `{name}`" for name in seen_order)
                st.markdown(lines)
            elif status == _RUNNING:
                st.write("Waiting for tasks to start…")

            if status == _FAILED and self.error_message:
                st.error(self.error_message)

        # Poll & schedule rerun while running
        if status == _RUNNING:
            self._poll_prefect()
            if auto_rerun:
                time.sleep(rerun_interval)
                st.rerun()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _ensure_state(self) -> None:
        if self._key not in st.session_state:
            self._reset(status=_IDLE)

    def _reset(self, *, status: str) -> None:
        st.session_state[self._key] = {
            "status": status,
            "results": None,
            "error": None,
            "flow_run_id": None,
            "flow_info": None,
            "task_runs": [],
            "thread": None,
        }

    def sync(self) -> None:
        """Sync background-thread state into session_state.

        Call this **once at the top of the page** (before any widget renders)
        so that all subsequent ``runner.running`` / ``runner.completed`` checks
        reflect the latest state written by the background thread.
        """
        store = _thread_store.get(self._key)
        if not store:
            return
        state = st.session_state.get(self._key)
        if state is None:
            return
        for field in ("status", "results", "error", "flow_run_id"):
            if field in store:
                state[field] = store[field]

    # Alias kept for internal callers
    _sync_from_thread_store = sync

    def _run_in_thread(self, workflow_or_profile: str, *, values: dict[str, Any], max_workers: int) -> None:
        """Target function executed in the background thread.

        All state updates go to ``_thread_store`` — never to ``st.session_state``.
        """
        store = _thread_store.setdefault(self._key, {})
        try:
            from genai_tk.utils.prefect_server import prefect_server
            from genai_tk.workflow.prefect.flow_factory import PrefectFlowFactory

            server = prefect_server()
            server.ensure_running()
            server.configure_api_url()

            factory = PrefectFlowFactory.from_profile(workflow_or_profile, values=values, max_workers=max_workers)

            # Capture the flow run ID via Prefect's on_running hook.
            # This hook fires INSIDE the flow context so flow_run.id is available.
            def _on_running(flow: Any, flow_run: Any, state: Any) -> None:  # noqa: ANN401
                try:
                    store["flow_run_id"] = str(flow_run.id)
                    logger.debug("WorkflowRunner[{}] captured flow_run_id={}", self._key, flow_run.id)
                except Exception as exc:
                    logger.debug("WorkflowRunner[{}] on_running hook error: {}", self._key, exc)

            flow_fn = factory.get().with_options(on_running=[_on_running])
            results = flow_fn()

            store["status"] = _COMPLETED
            store["results"] = results

        except Exception as exc:
            logger.exception("WorkflowRunner[{}] failed: {}", self._key, exc)
            store["status"] = _FAILED
            store["error"] = str(exc)

    def _run_flow_in_thread(self, flow_fn: "Any", *, flow_kwargs: "dict[str, Any] | None" = None) -> None:
        """Target for start_flow() — runs a bare Prefect @flow function.

        All state updates go to ``_thread_store``.
        """
        store = _thread_store.setdefault(self._key, {})
        try:
            from genai_tk.utils.prefect_server import prefect_server

            server = prefect_server()
            server.ensure_running()
            server.configure_api_url()

            def _on_running(flow: Any, flow_run: Any, state: Any) -> None:  # noqa: ANN401
                try:
                    store["flow_run_id"] = str(flow_run.id)
                    logger.debug("WorkflowRunner[{}] captured flow_run_id={}", self._key, flow_run.id)
                except Exception as exc:
                    logger.debug("WorkflowRunner[{}] on_running hook error: {}", self._key, exc)

            patched = flow_fn.with_options(on_running=[_on_running])
            results = patched(**(flow_kwargs or {}))

            store["status"] = _COMPLETED
            store["results"] = results if isinstance(results, dict) else {"result": results}

        except Exception as exc:
            logger.exception("WorkflowRunner[{}] failed: {}", self._key, exc)
            store["status"] = _FAILED
            store["error"] = str(exc)

    def _poll_prefect(self) -> None:
        """Query Prefect REST API and update session_state with latest task states."""
        flow_run_id = self.flow_run_id
        if not flow_run_id:
            return

        try:
            poller = PrefectPoller()
            flow_info = poller.get_flow_run(flow_run_id)
            task_runs = poller.get_task_runs(flow_run_id)

            state = st.session_state.get(self._key)
            if state is not None:
                if flow_info:
                    state["flow_info"] = flow_info
                state["task_runs"] = task_runs
        except Exception as exc:
            logger.debug("WorkflowRunner[{}] poll error: {}", self._key, exc)


# _prefect_ui_url is no longer used directly — callers now use prefect_server().ui_url
