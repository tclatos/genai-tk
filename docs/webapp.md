# Web Interface (`make webapp`)

genai-tk ships a minimal Streamlit webapp so you can test agents interactively
without writing any UI code.  Two built-in demo pages are included:

| Page | Description |
|------|-------------|
| 🦌 **DeerFlow Agent** | Full 2-panel UI — execution trace + chat, streaming, artifact viewer |
| 🤖 **ReAct Agent** | Two-panel chat + trace, tool-call display, MCP support, slash commands |

---

## Quick start

```bash
# From your project (or from genai-tk itself)
make webapp
```

This runs:
```
uv run streamlit run genai_tk/webapp/main/streamlit.py
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

After `uv add genai-tk` and `cli init`:

```bash
mkdir my-project && cd my-project
uv init && uv add genai-tk
cli init --name "My Project"
make webapp
```

The generated `Makefile` has a `webapp` target pointing to genai-tk's built-in
entry point.  Override `STREAMLIT_ENTRY` to use a custom file:

```bash
make webapp STREAMLIT_ENTRY=myapp/main/streamlit.py
```

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

## Running genai-blueprint's full UI

genai-blueprint extends the built-in pages with blueprint-specific demos
(Graph RAG, Maintenance Agent, etc.):

```bash
cd my-blueprint-project
make webapp    # uses blueprint's Makefile → genai_blueprint/main/streamlit.py
```

The blueprint's `deer_flow_agent.py` and `reAct_agent.py` pages are thin wrappers
that delegate to `genai_tk.webapp.pages.demos.*`.  Changes to the genai-tk
implementations are picked up automatically.

---

## Module layout

```
genai_tk/webapp/
├── main/
│   └── streamlit.py           ← minimal entry point (config-driven)
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
