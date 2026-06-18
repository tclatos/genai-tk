"""Prefect Workflow Demo — live progress display in Streamlit.

A self-contained demo that launches a multi-stage Prefect flow with 11 tasks
(parallel fetching, validation, parallel batch processing, aggregation, report)
and shows real-time execution progress via the ``WorkflowRunner`` component.

The label of the ``st.status`` widget reflects the currently executing task,
so users can track progress at a glance.
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner  # type: ignore[attr-defined]

from genai_tk.utils.streamlit.workflow_runner import WorkflowRunner

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Prefect Workflow Demo", page_icon="⚙️", layout="wide")
st.title("⚙️ Prefect Workflow Demo")
st.caption(
    "Launches a realistic 5-stage Prefect flow and renders live progress via Prefect REST API polling. "
    "The status label tracks the **currently running task** in real time."
)


# ── Demo flow definition ──────────────────────────────────────────────────────
# Stage 1: parallel fetches
@task(name="fetch-source-a")
def fetch_source_a() -> dict[str, Any]:
    """Simulate fetching from source A."""
    time.sleep(2.5)
    return {"source": "A", "records": 500}


@task(name="fetch-source-b")
def fetch_source_b() -> dict[str, Any]:
    """Simulate fetching from source B (slower)."""
    time.sleep(3.5)
    return {"source": "B", "records": 750}


@task(name="fetch-source-c")
def fetch_source_c() -> dict[str, Any]:
    """Simulate fetching from source C."""
    time.sleep(2.0)
    return {"source": "C", "records": 300}


# Stage 2: validate
@task(name="validate-sources")
def validate_sources(results: list[dict[str, Any]]) -> int:
    """Cross-check all fetched sources and return total record count."""
    time.sleep(1.5)
    return sum(r["records"] for r in results)


# Stage 3: parallel batch processing
@task(name="process-batch-1")
def process_batch_1(total: int) -> dict[str, Any]:
    """Process first third of records."""
    time.sleep(3.5)
    return {"batch": 1, "processed": total // 3}


@task(name="process-batch-2")
def process_batch_2(total: int) -> dict[str, Any]:
    """Process second third of records."""
    time.sleep(3.2)
    return {"batch": 2, "processed": total // 3}


@task(name="process-batch-3")
def process_batch_3(total: int) -> dict[str, Any]:
    """Process remaining records."""
    time.sleep(2.8)
    return {"batch": 3, "processed": total - 2 * (total // 3)}


# Stage 4: aggregate
@task(name="aggregate-results")
def aggregate_results(batches: list[dict[str, Any]], total: int) -> dict[str, Any]:
    """Merge batch outputs into a single summary."""
    time.sleep(1.0)
    return {"total_input": total, "total_processed": sum(b["processed"] for b in batches)}


# Stage 5: report
@task(name="generate-report")
def generate_report(summary: dict[str, Any]) -> dict[str, Any]:
    """Produce the final structured report."""
    time.sleep(1.5)
    return {
        **summary,
        "status": "ok",
        "coverage_pct": round(summary["total_processed"] / summary["total_input"] * 100, 1),
    }


@flow(
    name="demo-workflow",
    task_runner=ConcurrentTaskRunner(),  # type: ignore[call-arg]
    log_prints=True,
)
def demo_flow() -> dict[str, Any]:
    """5-stage demo: parallel fetch → validate → parallel batches → aggregate → report."""
    # Stage 1 — parallel fetches
    fa = fetch_source_a.submit()
    fb = fetch_source_b.submit()
    fc = fetch_source_c.submit()
    fetch_results = [fa.result(), fb.result(), fc.result()]

    # Stage 2 — validate
    total = validate_sources.submit(fetch_results).result()

    # Stage 3 — parallel batch processing
    b1 = process_batch_1.submit(total)
    b2 = process_batch_2.submit(total)
    b3 = process_batch_3.submit(total)
    batch_results = [b1.result(), b2.result(), b3.result()]

    # Stage 4 — aggregate
    summary = aggregate_results.submit(batch_results, total).result()

    # Stage 5 — report
    report = generate_report.submit(summary).result()

    return {"fetch_results": fetch_results, "total_records": total, "report": report}


# ── WorkflowRunner ────────────────────────────────────────────────────────────
runner = WorkflowRunner(key="prefect_demo_runner")

# Sync background-thread state FIRST, before any widget reads runner.running etc.
runner.sync()

# ── Sidebar ───────────────────────────────────────────────────────────────────
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
            "This demo runs a **5-stage Prefect flow** with 11 tasks:\n\n"
            "**Stage 1 — Fetch (parallel, ~3.5 s)**\n"
            "- 🔵 `fetch-source-a` (2.5 s)\n"
            "- 🔵 `fetch-source-b` (3.5 s)\n"
            "- 🔵 `fetch-source-c` (2.0 s)\n\n"
            "**Stage 2 — Validate (1.5 s)**\n"
            "- 🟡 `validate-sources`\n\n"
            "**Stage 3 — Process (parallel, ~3.5 s)**\n"
            "- 🔵 `process-batch-1` (3.5 s)\n"
            "- 🔵 `process-batch-2` (3.2 s)\n"
            "- 🔵 `process-batch-3` (2.8 s)\n\n"
            "**Stage 4 — Aggregate (1.0 s)**\n"
            "- 🟡 `aggregate-results`\n\n"
            "**Stage 5 — Report (1.5 s)**\n"
            "- 🟢 `generate-report`\n\n"
            "Total wall-clock time ≈ 12 s."
        )
        if st.button("▶️ Launch workflow", type="primary"):
            runner.start_flow(demo_flow)
            st.rerun()
    else:
        # render_progress handles: polling, st.status label with current task, auto-rerun
        runner.render_progress()

with col_right:
    st.subheader("Results")

    if runner.completed:
        results = runner.results or {}
        st.success("Workflow completed successfully.")

        report = results.get("report") or {}
        if report:
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Total records", results.get("total_records", "—"))
            mcol2.metric("Processed", report.get("total_processed", "—"))
            mcol3.metric("Coverage", f"{report.get('coverage_pct', '—')} %")

        with st.expander("Full results JSON", expanded=False):
            st.json(results)

    elif runner.failed:
        st.error(f"Workflow failed: {runner.error_message}")

    elif runner.running:
        # Show a live task summary table while running
        task_runs = runner.task_runs
        if task_runs:
            import re

            from genai_tk.utils.streamlit.workflow_runner import _DEFAULT_ICON, _STATE_ICONS

            rows = {}
            for t in task_runs:
                logical = re.sub(r"-[0-9a-f]{3,6}$", "", t.name)
                rows[logical] = t.state_name

            table_md = "| Task | State |\n|---|---|\n"
            table_md += "\n".join(
                f"| `{name}` | {_STATE_ICONS.get(state, _DEFAULT_ICON)} {state} |" for name, state in rows.items()
            )
            st.markdown(table_md)
        else:
            st.info("Results will appear here once the workflow completes.")

    else:
        st.caption("No results yet.")
