# Streamlit Workflow Runner — Live Prefect Execution UI

**Problem:** You want to run a Prefect workflow from a Streamlit page and show live progress without writing thread/polling logic.

**Solution:** Use the `WorkflowRunner` component to execute workflows in a background thread and display task progress via Prefect REST API polling.

---

## Quick Example

```python
from genai_tk.utils.streamlit.workflow_runner import WorkflowRunner
import streamlit as st

st.title("Run My Workflow")

runner = WorkflowRunner(key="my_runner")
runner.sync()  # sync background state at page top

if st.button("Launch Workflow"):
    runner.start("my_workflow_name")
    st.rerun()

if not runner.idle:
    runner.render_progress()  # shows live trace + auto-rerun

if runner.completed:
    st.success(f"Done! {runner.results}")
elif runner.failed:
    st.error(f"Failed: {runner.error_message}")
```

---

## API Reference

### WorkflowRunner

```python
runner = WorkflowRunner(key="unique_key", poll_interval=2.0)
```

#### State Properties

- `runner.idle` — No workflow is running.
- `runner.running` — Workflow is currently executing.
- `runner.completed` — Workflow finished successfully.
- `runner.failed` — Workflow encountered an error.
- `runner.flow_run_id` — UUID of the current Prefect flow run.
- `runner.flow_info` — FlowRunInfo object (name, state, start/end time).
- `runner.task_runs` — List of TaskRunInfo objects for started tasks.
- `runner.results` — Dict of workflow step results (when completed).
- `runner.error_message` — Error text (when failed).

#### Methods

**`sync()`** — Pull background-thread state into session_state.  
Call **once at the top** of your page (before rendering widgets that read runner state) so all subsequent state checks reflect the latest data from the background thread.

```python
runner = WorkflowRunner(key="my_runner")
runner.sync()  # IMPORTANT: always call this at page top
if runner.running:
    st.info("Running…")
```

**`start(workflow_name, values=None, max_workers=4)`** — Launch a named YAML workflow.

```python
runner.start("markdownize", values={"source_dir": "/data/docs"})
st.rerun()  # trigger a rerun so render_progress can poll
```

**`start_flow(flow_fn)`** — Launch a Prefect `@flow` function directly (no YAML config).

```python
from prefect import flow, task

@task
def process():
    return "result"

@flow
def my_flow():
    return process()

runner.start_flow(my_flow)
st.rerun()
```

**`reset()`** — Reset to idle state so a new workflow can be started.

```python
if st.button("Start over"):
    runner.reset()
    st.rerun()
```

**`render_progress(auto_rerun=True, rerun_interval=2.0)`** — Display live progress in `st.status` container.

When running, automatically:
- Polls Prefect API for task/flow state every `rerun_interval` seconds
- Calls `st.rerun()` to refresh the display
- Shows accumulated task trace (Running + Completed tasks; Pending tasks hidden)
- Deduplicates fan-out task names (e.g., all `process_item-*` variants show as one row)

```python
if not runner.idle:
    runner.render_progress(auto_rerun=True, rerun_interval=2.0)
```

---

## Common Patterns

### Sidebar Status + Run Button

```python
with st.sidebar:
    if runner.running:
        st.info("🔄 Workflow running…")
    elif runner.completed:
        if st.button("🔄 Run again"):
            runner.reset()
            st.rerun()
    else:
        if st.button("▶️ Start", type="primary"):
            runner.start("my_workflow")
            st.rerun()

    if runner.flow_run_id:
        from genai_tk.utils.prefect_server import prefect_server
        ui_url = prefect_server().ui_url
        st.markdown(f"[Open in Prefect UI]({ui_url}/runs/flow-run/{runner.flow_run_id})")
```

### Two-Column Layout (Progress + Results)

```python
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Progress")
    if not runner.idle:
        runner.render_progress()

with col_right:
    st.subheader("Results")
    if runner.completed:
        st.json(runner.results)
    elif runner.failed:
        st.error(runner.error_message)
```

### Uploading Files + Running Workflow

```python
# Step 1: Upload
uploaded = st.file_uploader("Choose ZIP")
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.getbuffer())
        zip_path = tmp.name
    st.success("Uploaded!")

# Step 2: Run workflow
if st.button("Process"):
    runner.start(
        "my_workflow",
        values={"zip_file": zip_path}
    )
    st.rerun()

# Step 3: Show progress
if not runner.idle:
    runner.render_progress()

# Step 4: Display results
if runner.completed:
    st.download_button("Download result", data=runner.results["output"])
```

---

## Configuration

The WorkflowRunner reads Prefect settings from `config/app_conf.yaml`:

```yaml
prefect:
  host: "127.0.0.1"      # Prefect server host
  port: 4200             # Prefect server port
  api_url: null          # Full API URL (auto-resolved when null)
  pid_file: null         # Where to store Prefect server PID
  auto_start: true       # Auto-start server if not running
```

All defaults are suitable for local development. The `WorkflowRunner` automatically:
1. Ensures the Prefect server is running (`server.ensure_running()`)
2. Configures the API URL environment variable
3. Handles HTTP proxy bypass for localhost

To use a remote Prefect server, set:
```yaml
prefect:
  api_url: "https://api.prefect.mycompany.com/api"
  auto_start: false
```

---

## Demo Page

See `genai_tk/webapp/pages/demos/prefect_workflow_demo.py` for a self-contained example that:
- Defines a simple 3-task flow
- Launches it via `runner.start_flow()`
- Shows accumulated task trace with deduplication
- Links to Prefect UI

Run it:
```bash
just webapp
# Open sidebar → Workflow section → Prefect Workflow Demo
```

---

## Thread Safety

The `WorkflowRunner` is safe to use in Streamlit:

- Background thread writes to a module-level store (never `st.session_state`)
- Main thread calls `runner.sync()` at page top to pull state from the store into `session_state`
- All state reads/writes happen on the main thread

This avoids the `ScriptRunContext` warning that occurs when non-main threads access Streamlit objects.

---

## Under the Hood

1. **`workflow_runner.py`** — Stateful component managing background thread and progress UI.
2. **`prefect_progress.py`** — Low-level Prefect REST API poller (queries `/flow_runs/{id}` and `/task_runs/filter`).
3. **`prefect_server.py`** — Singleton for Prefect server lifecycle (start/stop, health check, config).

If you need fine-grained polling control (e.g., custom retry logic), use `PrefectPoller` directly:

```python
from genai_tk.utils.streamlit.prefect_progress import PrefectPoller

poller = PrefectPoller()
flow_info = poller.get_flow_run(flow_run_id)
tasks = poller.get_task_runs(flow_run_id)
```
