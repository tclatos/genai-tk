---
name: kg-repo-map
description: Navigate the genai-graph repository by mapping work areas to docs, code, config, and the matching kg-* skill. Use before making cross-cutting changes to genai_graph/ or when deciding which genai-graph domain skill to load. Load this first whenever a task touches Knowledge Graph construction, the Document Graph, Cypher agents, Neo4j import, KG workflows, or the KG Explorer.
---

# GenAI Graph Repository Map

## Start Here

genai-graph is a Knowledge Graph / GraphRAG library built on genai-tk. It ingests
heterogeneous sources into a [Ladybug](https://github.com/LadybugDB/ladybug) (Kuzu-compatible)
graph database and exposes them through a CLI, a Streamlit webapp, and Cypher-aware agents.

Read the closest doc first, then inspect the matching implementation paths. Do not infer
behavior from filenames alone — the factory/schema layer is configuration-driven and the
docs are the intended navigation layer.

## Relationship to genai-tk

genai-graph extends genai-tk's three domains. When a task is about genai-tk itself
(workflow DSL, RAG retrievers, LLM/embeddings factories, agent harness, MCP), load the
matching `genai-tk/*` skill (see `genai-tk/skills/README.md`). When it is about graph
schema, factories, the Ladybug backend, the Document Graph, Cypher query agents, Neo4j
import, KG workflows, export/visualization, or the KG Explorer, load the `kg-*` skill
below. The two bundles are designed to be loaded together — see
[../README.md](../README.md).

## Domain Map

| Work area | Read first | Then inspect | Load skill |
|---|---|---|---|
| Orientation (this skill) | `docs/graph-definition-guide.md` | `genai_graph/kg/`, `genai_graph/orchestration/` | `kg-repo-map` |
| Graph schema definition | `docs/graph-definition-guide.md`, `docs/schema-compilation.md` | `genai_graph/kg/schema/` | `kg-schema` |
| Factories (data → graph) | `docs/graph-authoring-patterns.md`, `docs/graph_construction.md` | `genai_graph/kg/factories/` | `kg-factories` |
| Ingestion / Ladybug backend | `docs/graph_construction.md`, `docs/cache_management.md` | `genai_graph/kg/ingest/`, `genai_graph/kg/backend.py`, `genai_graph/kg/embeddings_handler.py` | `kg-ingest` |
| Document Graph | `docs/document-graph.md` | `genai_graph/kg/document_graph/`, `genai_graph/kg/factories/document_graph_factory.py` | `kg-document-graph` |
| Querying / Cypher agents | `docs/kg_explorer.md`, `docs/document-graph.md` | `genai_graph/kg/query/` | `kg-query` |
| Neo4j import | `docs/graph-authoring-patterns.md` (Pattern 3) | `genai_graph/neo4j_import/` | `kg-neo4j-import` |
| KG workflows | `docs/workflows.md`, `docs/prefect_dag_pipeline.md` | `genai_graph/orchestration/` | `kg-workflows` |
| Export / visualization | `docs/graph-definition-guide.md` (Visualise), `docs/graph_construction.md` | `genai_graph/kg/export/` | `kg-export` |
| CLI (`cli kg` / `cli docgraph` / `cli neo4j`) | `docs/workflows.md` (CLI Reference), `docs/document-graph.md` | `genai_graph/core/commands_ekg.py`, `genai_graph/core/commands_docgraph.py`, `genai_graph/neo4j_import/commands.py` | `kg-cli` |
| Streamlit KG Explorer | `docs/kg_explorer.md` | `genai_graph/webapp/`, `genai_graph/main/streamlit.py` | `kg-explorer` |
| Schema maintenance procedures | `Agents_Skills.md`, `docs/schema-compilation.md` | `genai_graph/kg/schema/`, `genai_graph/kg/factories/` | `kg-schema-maintenance` |

## Architecture (one screen)

```
Sources (Neo4j JSONL / Excel-CSV / PDF-PPTX-MD)
  → Factory layer (kg/factories/) → DataFrames of typed Pydantic nodes
  → GraphSchema compile (kg/schema/) → MERGE statements
  → Ladybug graph DB (kg/backend.py, kg/ingest/)
  → CLI (core/commands_*) | Streamlit webapp (webapp/) | Cypher agents (kg/query/)
```

genai-tk provides the Workflow Engine (`kg_build_step`/`docgraph_build_step` are
`@workflow`-decorated steps) and the BAML/LLM/embeddings factories that genai-graph builds on.

## Implementation Rules

- Use absolute imports from `genai_graph.*` (and `genai_tk.*` for toolkit APIs).
- Use Pydantic v2 for node models, config, DTOs, and results — see `Agents.md`.
- Use modern type hints (`str | None`, `list[Person]`); avoid `Optional`/`Union`.
- Use `model_dump_json()` / `model_validate` for Pydantic serialization, not `json.dumps`.
- Keep configuration in YAML (`config/workflows/*.yaml`, `config/app_conf.yaml`) where docs describe YAML-driven behavior.
- Run commands and tests with `uv run ...` (the project uses `just` recipes that wrap `uv`).
- Favor removing backward-compatibility code over keeping it (per `Agents.md`).

## Verification Shortcuts

```bash
uv run just test          # full suite
uv run just lint          # ruff
uv run just fmt           # ruff format
uv run just webapp        # Streamlit KG Explorer
uv run cli kg info        # current KG stats
uv run cli kg schema      # node/relationship schema
```

## Complements

- `genai-tk/repo-map` — orientation for the genai-tk repo itself. Load this skill for
  genai-graph, and `genai-tk/repo-map` for genai-tk.
