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

import threading
import time
from typing import Any

from loguru import logger

import streamlit as st
from genai_tk.utils.streamlit.prefect_progress import FlowRunInfo, PrefectPoller, TaskRunInfo

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
    instances can coexist on the same page.

    Args:
        key: Unique string key scoped to this runner instance.
        poll_interval: Seconds between Prefect API polls (default: 2.0).
    """

    def __init__(self, key: str = "workflow_runner", poll_interval: float = _POLL_INTERVAL) -> None:
        self._key = key
        self._poll_interval = poll_interval
        self._ensure_state()

    # ------------------------------------------------------------------
    # State accessors
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
        """Launch the workflow in a background thread.

        Args:
            workflow_or_profile: Workflow name or ``workflow/profile`` string.
            values: Optional parameter overrides.
            max_workers: Prefect thread-pool size.
        """
        if not self.idle and not self.failed:
            logger.warning("WorkflowRunner[{}]: already running, ignoring start()", self._key)
            return

        self._reset(status=_RUNNING)
        thread = threading.Thread(
            target=self._run_in_thread,
            args=(workflow_or_profile,),
            kwargs={"values": values or {}, "max_workers": max_workers},
            daemon=True,
            name=f"wf-runner-{self._key}",
        )
        self._state["thread"] = thread
        thread.start()

    def reset(self) -> None:
        """Reset to idle state so a new workflow can be started."""
        self._reset(status=_IDLE)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_progress(self, *, auto_rerun: bool = True, rerun_interval: float = _POLL_INTERVAL) -> None:
        """Render live progress widgets.

        When the workflow is running, displays a ``st.status`` container with
        per-task progress and schedules a page rerun every *rerun_interval* seconds.

        Args:
            auto_rerun: If True, calls ``st.rerun()`` while the workflow runs.
            rerun_interval: Seconds to wait between page reruns during execution.
        """
        if self.idle:
            return

        status = self._state["status"]
        task_runs = self.task_runs
        flow_info = self.flow_info

        # Determine st.status label and state
        if status == _RUNNING:
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
                st.caption(f"Flow run: `{flow_info.name}` — {flow_info.state_name}")

            if task_runs:
                for task in sorted(task_runs, key=lambda t: (t.start_time or "", t.name)):
                    icon = _STATE_ICONS.get(task.state_name, _DEFAULT_ICON)
                    st.write(f"{icon} **{task.name}** — {task.state_name}")
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

    def _run_in_thread(self, workflow_or_profile: str, *, values: dict[str, Any], max_workers: int) -> None:
        """Target function executed in the background thread."""
        try:
            # Import here to avoid circular imports at module level
            from genai_tk.utils.prefect_server import prefect_server
            from genai_tk.workflow.prefect.flow_factory import PrefectFlowFactory

            server = prefect_server()
            server.ensure_running()
            server.configure_api_url()

            factory = PrefectFlowFactory.from_profile(workflow_or_profile, values=values, max_workers=max_workers)

            # Intercept the flow run ID by patching the Prefect context
            flow_run_id: str | None = None

            def _capture_flow_run_id(flow_run_id_: str) -> None:
                nonlocal flow_run_id
                flow_run_id = flow_run_id_
                if self._key in st.session_state:
                    st.session_state[self._key]["flow_run_id"] = flow_run_id_

            # Attach a Prefect event hook (Prefect fires this after the run is created)
            flow_fn = factory.get()

            # Monkey-patch: after creating the flow run, Prefect stores run context.
            # We use a wrapper to capture the flow run id from FlowRunContext.
            original_call = flow_fn.__call__  # type: ignore[attr-defined]

            def _patched_call(*args: Any, **kwargs: Any) -> Any:
                result = original_call(*args, **kwargs)
                # Try to read the last flow run from context
                try:
                    from prefect.context import FlowRunContext

                    ctx = FlowRunContext.get()
                    if ctx and ctx.flow_run:
                        _capture_flow_run_id(str(ctx.flow_run.id))
                except Exception:
                    pass
                return result

            results = _patched_call()

            # Success
            if self._key in st.session_state:
                st.session_state[self._key]["status"] = _COMPLETED
                st.session_state[self._key]["results"] = results
        except Exception as exc:
            logger.exception("WorkflowRunner[{}] failed: {}", self._key, exc)
            if self._key in st.session_state:
                st.session_state[self._key]["status"] = _FAILED
                st.session_state[self._key]["error"] = str(exc)

    def _poll_prefect(self) -> None:
        """Query Prefect REST API and update session state with latest task states."""
        flow_run_id = self.flow_run_id
        if not flow_run_id:
            # Thread may not have written the ID yet
            return

        try:
            poller = PrefectPoller()
            flow_info = poller.get_flow_run(flow_run_id)
            task_runs = poller.get_task_runs(flow_run_id)

            if self._key in st.session_state:
                if flow_info:
                    st.session_state[self._key]["flow_info"] = flow_info
                st.session_state[self._key]["task_runs"] = task_runs
        except Exception as exc:
            logger.debug("WorkflowRunner[{}] poll error: {}", self._key, exc)
