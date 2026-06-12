# Web Interface (`just webapp`)

genai-tk ships a Streamlit webapp so you can test agents interactively
without writing any UI code.  Three built-in demo pages are included out
of the box:

| Page | Section | Description |
|------|---------|-------------|
| 🦌 **DeerFlow Agent** | Agents | Full 2-panel UI — execution trace + chat, streaming, artifact viewer |
| 🤖 **ReAct Agent** | Agents | Two-panel chat + trace, tool-call display, MCP support, slash commands |
| 🤖 **SmolAgents** | Agents | SmolAgents step-by-step display |
| ⚙️ **Prefect Workflow Demo** | Workflow | Live Prefect flow execution trace with real-time task progress polling |

Downstream projects (e.g. genai-blueprint) can embed these pages alongside
their own pages using the `genai_tk://` reference prefix described below,
so the agent UIs stay maintained in one place.

---

## Installation

Streamlit is a **required dependency** of genai-tk, so it is installed
automatically — no separate `uv add streamlit` step needed.

### Starting from scratch

```bash
mkdir my-project && cd my-project
uv init

# 1. Install genai-tk (pulls in streamlit, pillow, and all other deps)
uv add git+https://github.com/tclatos/genai-tk@main

# 2. Initialise config and a project justfile
uv run cli init                # copies default config/ tree + writes justfile
uv run cli init --name "My Project"  # optionally set the app title

# 3. (Optional) Also install the Deer-flow backend for the DeerFlow demo page
uv run cli init --deer-flow            # clones deer-flow into ~/deer-flow
uv run cli init --deer-flow --path ./ext/deer-flow  # or to a custom path
```

`cli init` is idempotent — re-running it skips files that already exist unless
you pass `--force`.

### Adding to an existing project

```bash
# In your existing uv-managed project
uv add git+https://github.com/tclatos/genai-tk@main
uv run cli init
```

---

## Quick start

```bash
# From your project (or from genai-tk itself)
just webapp
```

This runs:
```
uv run python -m streamlit run <genai_tk-package>/webapp/main/streamlit.py
```

Open <http://localhost:8501> in your browser.

> **Prerequisites**: `DEER_FLOW_PATH` must be set for the DeerFlow page.
> Run `cli init --deer-flow` or set the env var pointing to your Deer-flow clone.

---

## Configuration

The webapp reads two optional config keys:

```yaml
# config/webapp.yaml (created by cli init)
ui:
  app_name: My Project     # browser tab title (default: current directory name)
  logo: path/to/logo.png   # relative to CWD or absolute — omit for no logo
```

`cli init` copies `webapp.yaml` to your `config/` directory and sets `app_name`
to the project directory name.  Run with a custom name:

```bash
cli init --name "My AI Project"
```

---

## Running from a new project

See [Installation](#installation) above for how to bootstrap a new project.
Once `cli init` has run:

```bash
just webapp
```

The generated `justfile` has a `webapp` recipe that resolves the genai-tk
entry point automatically via Python — no path override needed.

---

## Adding your own pages

To extend the built-in demo with project-specific pages, configure navigation
in your `config/webapp.yaml`:

```yaml
ui:
  app_name: My Project
  pages_dir: myapp/webapp/pages   # path to your pages directory
  navigation:
    demos:
      - demos/my_custom_agent.py
      - demos/deer_flow_agent.py   # include the built-in ones too if you want
    settings:
      - settings/configuration.py
```

When `ui.navigation` and `ui.pages_dir` are both set, the webapp switches to
full project-mode navigation and only shows the pages you list.

---

## Referencing genai-tk pages from a downstream project

Any project that includes genai-tk can embed its built-in agent pages alongside
their own pages using the `genai_tk://` prefix in navigation entries.  The
webapp entry point resolves this prefix to the installed package path via
`importlib.resources`, so it works whether genai-tk is installed from GitHub,
as an editable local path, or declared as a workspace dependency.

```yaml
# config/app_conf.yaml (or webapp.yaml) in your downstream project
ui:
  pages_dir: ${paths.src}/webapp/pages   # your own pages directory
  navigation:
    agents:
      - genai_tk://demos/deer_flow_agent.py   # ← installed genai-tk page
      - genai_tk://demos/reAct_agent.py       # ← installed genai-tk page
    demos:
      - demos/my_custom_agent.py              # ← your own page (relative to pages_dir)
    settings:
      - settings/configuration.py
```

Three path formats are supported for each navigation entry:

| Prefix | Resolution |
|--------|------------|
| `genai_tk://path/page.py` | `genai_tk/webapp/pages/path/page.py` inside the installed package |
| `/absolute/path/page.py` | Used as-is |
| `relative/path/page.py` | Joined with `ui.pages_dir` |

Page titles are set automatically.  Built-in genai-tk pages receive pretty
names with emojis (`🦌 DeerFlow Agent`, `🤖 ReAct Agent`, etc.).  Your own
pages are named from their filename with standard title-case conversion.

> **Important:** Streamlit requires each page to have a unique URL pathname.
> Do not list the same physical file in more than one navigation section.

---

## Module layout

```
genai_tk/webapp/
├── main/
│   └── streamlit.py           ← config-driven entry point; handles genai_tk:// resolution
├── pages/
│   └── demos/
│       ├── deer_flow_agent.py ← DeerFlow 2-panel demo
│       └── reAct_agent.py     ← ReAct agent demo
└── ui_components/
    ├── agent_layout.py        ← sidebar helpers shared by both pages
    ├── config_editor.py       ← YAML editor dialog (requires streamlit-monaco)
    ├── llm_selector.py        ← LLM dropdown widget
    ├── message_renderer.py    ← Mermaid diagram + markdown renderer
    ├── smolagents_streamlit.py ← SmolAgents step display helpers
    └── streamlit_chat.py      ← chat message display + callback handler

genai_tk/utils/streamlit/
    ├── auto_scroll.py
    ├── capturing_callback_handler.py
    ├── prefect_progress.py       ← Prefect REST API polling helpers
    ├── workflow_runner.py        ← stateful workflow execution + progress UI
    └── thread_issue_fix.py
```

---

## Extending via `ui_components`

All UI helpers are importable from genai-tk directly:

```python
from genai_tk.webapp.ui_components.agent_layout import render_agent_sidebar
from genai_tk.webapp.ui_components.message_renderer import render_message_with_mermaid
from genai_tk.webapp.ui_components.streamlit_chat import StreamlitStatusCallbackHandler
from genai_tk.utils.streamlit.thread_issue_fix import get_streamlit_cb
```

---

## Optional: YAML config editor

The `edit_config_dialog` widget needs `streamlit-monaco`:

```bash
uv add streamlit-monaco
```

Without it the Edit Config button shows the raw YAML as read-only text.

---

## Live Workflow Execution in Streamlit (`WorkflowRunner`)

The `WorkflowRunner` component executes Prefect workflows in a background thread and displays live progress in a Streamlit `st.status` container.  Use it in custom pages to add workflow execution UI without writing polling/threading boilerplate.

### Basic Usage

```python
from genai_tk.utils.streamlit.workflow_runner import WorkflowRunner
import streamlit as st

runner = WorkflowRunner(key="my_runner")

# Always sync at page top before rendering widgets that read runner state
runner.sync()

with st.sidebar:
    if runner.running:
        st.info("Workflow is running…")
    elif runner.completed:
        if st.button("Run again"):
            runner.reset()
            st.rerun()

    if runner.flow_run_id:
        st.markdown(f"[Prefect UI](.../{runner.flow_run_id})")

if st.button("Launch"):
    runner.start("workflow_name")
    st.rerun()

# Renders live progress: flow name, task status, auto-updates every 2s
runner.render_progress()

if runner.completed:
    st.success(f"Results: {runner.results}")
```

### Two Ways to Run Workflows

**1. YAML workflow via `start()`:**
```python
runner.start(
    "markdownize_and_merge",  # workflow name from config/workflows/
    values={"source_dir": "/path/to/docs"}  # override defaults
)
```

**2. Prefect `@flow` function via `start_flow()`:**
```python
from prefect import flow, task
import time

@task
def fetch_data():
    time.sleep(3)
    return "data"

@flow
def my_flow():
    return fetch_data()

runner.start_flow(my_flow)  # no YAML config needed
```

### Progress Display

The `render_progress()` method shows:
- **Flow run link** — clickable `[name]` to Prefect UI
- **Task trace** — accumulated list of tasks with icons:
  - `🔄` Running
  - `✅` Completed
  - `❌` Failed  
  - etc.
- **Deduplication** — fan-out tasks (`process_item-0`, `process_item-1`, …) appear as a single `process_item` line

The component automatically:
- Polls Prefect REST API every 2 seconds (configurable: `rerun_interval=`)
- Calls `st.rerun()` to refresh the display
- Syncs background-thread state into `session_state` before each poll

### Configuration

The `WorkflowRunner` reads the `prefect:` section of your config for server details:

```yaml
prefect:
  host: "127.0.0.1"
  port: 4200
  api_url: null  # auto-resolved as http://{host}:{port}/api when null
  auto_start: true
```

All defaults are sensible for local development; override in your `config/app_conf.yaml` or environment.

### See Also

- [workflows.md](workflows.md) — YAML-driven workflow definitions
- [prefect.md](prefect.md) — `@flow` and `@task` authoring
- `genai_tk/utils/streamlit/prefect_progress.py` — low-level REST API poller (if you need fine-grained polling control)
