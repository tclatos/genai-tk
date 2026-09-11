---
name: kg-export
description: Export genai-graph Knowledge Graph artifacts — interactive HTML/D3 schema diagrams and graph visualizations (generate_html, export_schema_html), DAG HTML, the consolidated Markdown warnings report (export_warnings), info/schema/JSON/parquet artifacts, and ResolvedSchema rendering. Use when editing genai_graph/kg/export, adding a viz/info/warnings output, or debugging a cli kg create export step.
---

# GenAI Graph Export & Visualization

## Read First

- `docs/graph-definition-guide.md` — "Visualise the schema" section
- `docs/graph_construction.md` — artifact locations and "Warnings and How to Handle Them"
- `genai_graph/kg/export/__init__.py` — public exports
- `genai_graph/kg/export/artifacts.py` — `export_html`, `export_schema`, `export_info`, `export_warnings`, fingerprints, parquet manifest
- `genai_graph/kg/export/html.py`, `html_template.py` — graph HTML viz
- `genai_graph/kg/export/dag_html.py`, `dag_html_template.py` — pipeline DAG viz
- `genai_graph/kg/schema/schema_html.py`, `schema_html_template.py`, `schema_d3.py` — schema D3 diagram (in `kg/schema/`)
- `genai_graph/kg/export/warnings_report.py` — warning categorization + Markdown report
- `genai_graph/kg/schema/resolved.py` — `ResolvedSchema` render methods

## Public API

```python
from genai_graph.kg.export import (
    generate_html,  # graph visualization HTML
    generate_dag_html,  # pipeline DAG HTML
    export_html,  # full HTML export for a KG profile
    export_schema,  # schema markdown
    export_schema_json,  # canonical schema JSON (used by agents/text2cypher)
    export_schema_html,  # interactive D3 schema diagram
    export_info,  # info.md (stats + links to schema/warnings)
    export_warnings,  # consolidated Markdown warnings report
    HtmlExportResult,
    ParquetExportResult,
    ParquetManifest,
    CacheFingerprints,
    compute_fingerprints_for_config,
    validate_parquet_cache,
)
from genai_graph.kg.schema import ResolvedSchema
```

## Schema rendering (ResolvedSchema)

`ResolvedSchema` is the canonical enriched representation used for rendering and LLM prompts.

```python
from genai_graph.kg.schema import ResolvedSchema

resolved = ResolvedSchema.from_graph_schema(schema)
print(resolved.to_markdown())  # table summary (node/rel/vector sections)
print(resolved.to_vector_section_markdown())  # '### Vector-Indexed Fields' for agent prompts
resolved.to_html_file("schema.html")  # interactive D3 diagram
json_str = resolved.to_json_str()  # D3 JSON for tools/prompts
resolved.to_json_file("schema.json")  # canonical schema JSON
```

The canonical schema JSON (`export_schema_json`) is what `build_kg_agent_system_prompt` and
`_ensure_vector_indexes` load at query time (see `kg-query`) — it carries structured
`vector_indexes` info. Inject BAML field/class descriptions via
`ResolvedSchema.from_graph_schema(schema, descriptions=_parse_baml_descriptions(file_map))`.

## Graph & DAG HTML

- `generate_html(...)` / `export_html(config_name, ...)` — interactive HTML visualization of
  the ingested graph (nodes/relationships), opened by `cli kg view`.
- `generate_dag_html(...)` — renders the KG build pipeline as a DAG (Prefect task graph).

## Warnings report

`export_warnings(config_name, warnings)` analyzes and categorizes schema warnings into a
single Markdown report (`{kg_outputs}/{profile}/{profile}-{tag}-warnings.md`), grouped by
category with explanations and suggestions:

- 🔄 Duplicate relationships — multiple rel types between the same node pair (e.g.
  `HAS_CUSTOMER` and `FOR_CUSTOMER` between `Opportunity` and `Customer`).
- ⚠️ Missing node configurations — referenced nodes without a `GraphNode`.
- 🔗 Orphaned nodes — nodes not reachable from the root model.
- ❌ Schema creation failures — subgraph schema errors.
- ℹ️ Other warnings.

The report is linked from the info file and is generated automatically at the end of
`create_kg_flow` (via `summarize_warnings` + `export_warnings`). See `kg-schema-maintenance`
for how to act on each category.

## Info & cache artifacts

- `export_info(config_name, ...)` — `info.md` with DB stats and links to the schema and
  warnings reports.
- `compute_fingerprints_for_config` / `CacheFingerprints` / `validate_parquet_cache` —
  parquet cache fingerprinting and validation (see `kg-ingest` for the force-stage story).
- `ParquetManifest` / `ParquetExportResult` — parquet data export for KG transfer.

## Artifact locations

```
{kg_outputs}/{profile}/
  {profile}-{tag}-schema.md       # export_schema
  {profile}-{tag}-schema.json     # export_schema_json (canonical, used by agents)
  {profile}-{tag}-schema.html     # export_schema_html (D3)
  {profile}-{tag}-info.md         # export_info
  {profile}-{tag}-warnings.log    # plain text log
  {profile}-{tag}-warnings.md     # export_warnings (consolidated report)
  {profile}-{tag}.html            # export_html (graph viz)
```

## Change Workflow

1. Rendering changes go through `ResolvedSchema` (markdown/JSON/HTML) so agents, docs, and
   the UI stay consistent — don't render schema from raw `GraphSchema` in a new place.
2. New warning categories go in `warnings_report.py` (`WarningCategory` + pattern matching);
   add a test in `tests/unit_tests/test_warnings_report.py`.
3. New artifacts get an `export_*` function in `artifacts.py`, an export in `__init__.py`,
   and a link from `export_info` so users can discover them.
4. HTML/D3 assets live under `kg/export/` and `kg/schema/` (`genai_graph/kg/_d3_bundle.py`,
   `d3.v5.min.js`, `d3-dag.iife.min.js`, templates) — keep them bundled, not fetched from a CDN.

## Commands

```bash
cli kg view                       # open the graph HTML visualization
cli kg schema                     # print schema (regenerates with --regen)
cli kg info                       # DB stats + artifact links
cli kg create <name>              # generates schema/info/warnings/html artifacts
uv run just test
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/test_warnings_report.py -q
```

## Avoid

- Do not hand-render schema for agent prompts — use `ResolvedSchema.to_markdown()` /
  `to_vector_section_markdown()` so vector-index info is included.
- Do not drop the warnings report when changing `create_kg_flow` — call `export_warnings`
  after `summarize_warnings`.
- Do not fetch D3/viz assets from a CDN; bundle them in `kg/export/`.

## Complements

- `kg-schema` — `ResolvedSchema` and the warnings emitted at schema construction.
- `kg-schema-maintenance` — how to fix each warnings-report category.
- `kg-query` — consumes the canonical schema JSON this exports.
- `kg-cli` — `cli kg view` / `cli kg schema` / `cli kg info` over these exporters.
