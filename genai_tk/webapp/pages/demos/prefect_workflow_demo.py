"""Prefect Workflow Demo — live progress display in Streamlit.

A self-contained demo that launches a simple Prefect flow with three tasks
(two run in parallel, one waits for them) and shows real-time execution
progress via the ``WorkflowRunner`` component.

Purpose: validate that the Prefect polling + ``st.status`` rendering works
correctly before wiring up more complex workflows.
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st
from prefect import flow, task
from prefect.task_runners import ThreadPoolTaskRunner

from genai_tk.utils.streamlit.workflow_runner import WorkflowRunner

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Prefect Workflow Demo", page_icon="⚙️", layout="wide")
st.title("⚙️ Prefect Workflow Demo")
st.caption(
    "Launches a simple 3-task Prefect flow and renders live progress via Prefect REST API polling. "
    "Use this page to validate the ``WorkflowRunner`` component."
)

# ── Demo flow definition ─────────────────────────────────────────────────────


@task(name="fetch-data")
def fetch_data(delay: float = 8.0) -> str:
    """Simulate fetching data from an external source."""
    time.sleep(delay)
    return "dataset-v1"


@task(name="process-records")
def process_records(delay: float = 10.0) -> int:
    """Simulate CPU-bound record processing."""
    time.sleep(delay)
    return 42


@task(name="generate-report")
def generate_report(source: str, count: int) -> dict[str, Any]:
    """Combine results and produce a report (depends on both upstream tasks)."""
    time.sleep(4.0)
    return {"source": source, "records_processed": count, "status": "ok"}


@flow(
    name="demo-workflow",
    task_runner=ThreadPoolTaskRunner(max_workers=4),
    log_prints=True,
)
def demo_flow() -> dict[str, Any]:
    """Simple demo: fetch + process run in parallel, then report is generated."""
    # Submit both independent tasks concurrently
    fetch_future = fetch_data.submit()
    process_future = process_records.submit()

    # Wait for both, then produce the report
    report = generate_report(
        source=fetch_future.result(),
        count=process_future.result(),
    )
    return {"fetch": fetch_future.result(), "process": process_future.result(), "report": report}


# ── Session state ────────────────────────────────────────────────────────────
runner = WorkflowRunner(key="prefect_demo_runner")

# Sync background-thread state FIRST, before any widget reads runner.running etc.
runner.sync()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Demo Controls")

    if runner.running:
        active = [t.name for t in runner.task_runs if t.state_name == "Running"]
        if active:
            st.info(f"🔄 {', '.join(active)}")
        else:
            st.info("Workflow is running…")
    elif runner.completed:
        if st.button("🔄 Run again", use_container_width=True):
            runner.reset()
            st.rerun()
    elif runner.failed:
        if st.button("🔄 Retry", use_container_width=True, type="primary"):
            runner.reset()
            st.rerun()

    st.divider()
    st.subheader("Flow run")
    flow_run_id = runner.flow_run_id
    if flow_run_id:
        try:
            from genai_tk.utils.prefect_server import prefect_server as _ps

            prefect_url = _ps().ui_url
        except Exception:
            prefect_url = "http://127.0.0.1:4200"
        st.markdown(f"[Open in Prefect UI ↗]({prefect_url}/runs/flow-run/{flow_run_id})")
        st.code(flow_run_id, language=None)
    else:
        st.caption("No run started yet.")

    if runner.running:
        active = [t.name for t in runner.task_runs if t.state_name == "Running"]
        if active:
            st.caption(f"Running: {', '.join(active)}")

# ── Main area ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Execution progress")

    if runner.idle:
        st.info(
            "This demo runs a 3-task Prefect flow:\n\n"
            "- 🔵 **fetch-data** (8 s) — runs in parallel with process-records\n"
            "- 🔵 **process-records** (10 s) — runs in parallel with fetch-data\n"
            "- 🟢 **generate-report** (4 s) — waits for both to finish\n\n"
            "Total wall-clock time ≈ 14 s."
        )
        if st.button("▶️ Launch workflow", type="primary"):
            runner.start_flow(demo_flow)
            st.rerun()
    else:
        # render_progress handles: polling, st.status rendering, and auto-rerun
        runner.render_progress()

with col_right:
    st.subheader("Results")

    if runner.completed:
        results = runner.results or {}
        st.success("Workflow completed successfully.")
        st.json(results)

    elif runner.failed:
        st.error(f"Workflow failed: {runner.error_message}")

    elif runner.running:
        st.info("Results will appear here once the workflow completes.")

    else:
        st.caption("No results yet.")
