---
name: kg-explorer
description: Work on the genai-graph Streamlit KG Explorer and webapp — the interactive Cypher query UI, Text-to-Cypher (natural language) panel, D3 graph visualization, tabular results, page assembly, and shared UI components. Use when editing genai_graph/webapp or genai_graph/main/streamlit.py, adding a KG Explorer page, or debugging the Streamlit KG UI.
---

# GenAI Graph KG Explorer (Streamlit)

## Read First

- `docs/kg_explorer.md` — KG Explorer features and usage
- `genai_graph/main/streamlit.py` — app entry point, page assembly
- `genai_graph/webapp/pages/` — pages (`settings/`, `demos/`)
- `genai_graph/webapp/ui_components/` — shared UI components
- `genai_graph/webapp/cli_commands.py` — CLI-from-the-webapp helpers
- `genai_graph/kg/query/` — the query primitives the Explorer wraps (see `kg-query`)
- `config/app_conf.yaml` — webapp/page registration

## What the KG Explorer is

A Streamlit page for interactively exploring and querying a Knowledge Graph:

- **Cypher query interface** — predefined query examples dropdown + editable query input;
  results as graph, table, or both.
- **Text-to-Cypher (natural language)** — ask in plain English; AI generates Cypher
  (review before execute); choose subgraph/LLM. Backed by `text2cypher_chain` / `query_kg`.
- **Visualization** — interactive D3.js graph (zoom/pan/drag, hover tooltips, color-coded by
  node type) via `cypher_graph_display`; tabular results with CSV export.

It reuses the same query primitives as the CLI and agents (`create_kg_cypher_tool`,
`text2cypher_chain`, `query_kg`, `build_kg_agent_system_prompt`) — see `kg-query`.

## App structure

```
genai_graph/main/streamlit.py        # entry point; assembles st.Page list, sidebar nav
genai_graph/webapp/
  pages/
    settings/   # configuration, MCP_servers, welcome
    demos/      # kg_query (KG Explorer), kg_schema_visualization, kg_visualization,
                # kg_tables, kg_lineage, kg_warnings, reAct_agent, external_web_page
  ui_components/  # kg_config_selector, llm_selector, cypher_graph_display,
                  # streamlit_chat, trace_middleware, config_editor, smolagents_streamlit
  cli_commands.py
```

## Run it

```bash
just webapp
# or:
uv run streamlit run genai_graph/main/streamlit.py
```

Prerequisite: a built KG (`cli kg create <name>`), since the Explorer reads the schema JSON
and connects to the Ladybug DB via `create_backend_from_config`.

## Add / modify a page

1. Add a page module under `webapp/pages/...` (Streamlit scripts) and register it in
   `config/app_conf.yaml` webapp page list / `main/streamlit.py` page assembly.
2. Reuse `ui_components/` — `kg_config_selector` (pick a KG profile), `llm_selector`,
   `cypher_graph_display` (render Cypher results as a D3 graph), `streamlit_chat`,
   `trace_middleware` — instead of rebuilding them.
3. Use stable Streamlit session-state keys prefixed by the page feature.
4. Keep page imports light so app startup stays responsive; lazy-import heavy/optional deps
   inside the page.
5. For any new query surface, reuse `kg/query` primitives so the CLI, agents, and Explorer
   stay consistent.

## Change Workflow

1. Decide whether the change is a new page, a UI component, or query behavior.
2. New shared UI → `ui_components/`; new self-contained page → `pages/`.
3. Query/visualization behavior changes should flow through `kg/query` +
   `cypher_graph_display` so all surfaces benefit.
4. Verify with `just webapp` against a real KG.

## Commands

```bash
just webapp
uv run streamlit run genai_graph/main/streamlit.py
cli kg create <name>      # prerequisite: a built KG to explore
uv run just test
```

## Avoid

- Do not store credentials in Streamlit session state.
- Do not block app import with live LLM calls or browser startup.
- Do not duplicate UI components when one exists under `ui_components/`.
- Do not hand-roll Cypher execution in a page — reuse `kg/query` primitives.

## Complements

- `genai-tk/webapp` — the genai-tk Streamlit webapp foundation and UI components.
- `genai-tk/streamlit-workflow-runner` — `WorkflowRunner` for running KG build flows in the UI.
- `kg-query` — the Cypher / Text-to-Cypher / agent-tool primitives the Explorer wraps.
- `kg-export` — `cypher_graph_display` and the D3 schema diagram share rendering roots.
