---
name: kg-workflows
description: Orchestrate genai-graph Knowledge Graph pipelines with the genai-tk workflow engine — kg_create_step/kg_build_step/docgraph_build_step, force stages (unzip/pdf/md/parquet/graph/embed/all), multi-factory pipelines, cli kg create vs cli docgraph run, and sub-workflow composition. Use when editing genai_graph/orchestration or config/workflows/*.yaml, or debugging a KG build pipeline.
---

# GenAI Graph Workflows

## Read First

- `docs/workflows.md` — DSL, force stages, `cli kg create` / `cli docgraph` / `cli neo4j` reference
- `docs/prefect_dag_pipeline.md` — Prefect DAG internals, concurrency model
- `docs/document-graph.md` — `docgraph_build_step` end-to-end
- `genai_graph/orchestration/workflow_steps.py` — `kg_create_step`, `kg_build_step`, `docgraph_build_step`
- `genai_graph/orchestration/flows.py` — `create_kg_flow`
- `genai_graph/orchestration/tasks.py` — per-stage Prefect tasks
- `config/workflows/generic_workflows.yaml`, `config/workflows/data_injection.yaml`

## Relationship to genai-tk

genai-graph does **not** reimplement the workflow engine — it uses genai-tk's
(`genai_tk.workflow`). The three genai-graph step functions are `@workflow`-decorated
adapters referenced by `run:` in a project's `config/workflows/*.yaml`. For the engine
itself (DSL, resolver, `flow_from_yaml`, Prefect server), load `genai-tk/workflow-engine`.

## Step functions

| Step | `run:` dotted path | Purpose |
|---|---|---|
| `kg_create` | `genai_graph.orchestration.workflow_steps.kg_create_step` | Build a predefined KG profile (`config_name`). |
| `kg_build` (hidden) | `genai_graph.orchestration.workflow_steps.kg_build_step` | Build a KG from a single inline `graph` factory config + `kg_name`. The building block for multi-factory pipelines. |
| `docgraph_build` (hidden) | `genai_graph.orchestration.workflow_steps.docgraph_build_step` | Markdownize → entity factories → Document Graph, into one DB. |

All three clear factory caches before running (prevents cross-contamination) and return a
summary dict `{config_name, total_processed, total_failed, warnings_count, db_path}`.

## Force stages

A single ordered `--force <stage>` replaces ad-hoc booleans. Forcing a stage re-runs it and
everything downstream (downstream caches derive from upstream outputs). Defined in
`genai_tk.workflow.force.ForceStage`:

| Stage | Effect | Commands |
|---|---|---|
| `unzip` | Re-extract `.zip` archives | `markdownize_documents`, `docgraph build` |
| `pdf` | Re-run Office → PDF | `markdownize_documents`, `docgraph build` |
| `md` | Re-run document → Markdown | `markdownize_documents`, `docgraph build` |
| `parquet` | Rebuild JSON → parquet import caches | `kg create`, `kg_build` |
| `graph` | Re-ingest (drops destination store) | `docgraph build`, `kg create`, `kg_build` |
| `embed` | Recompute embeddings | `docgraph build`, `kg create` |
| `all` | Force every stage, drop destination store | all |

`--delete-first` is a separate DB-lifecycle flag (drop DB without forcing upstream caches).

## Adding a KG config (single factory)

```yaml
# config/workflows/my_graph.yaml
workflows:
  my_new_kg:
    description: "My new knowledge graph"
    run: genai_graph.orchestration.workflow_steps.kg_create_step
    defaults:
      config_name: my_new_kg
      delete_first: false
      export_html: true
```

```bash
uv run cli workflow run my_new_kg --dry-run
cli kg create my_new_kg
```

## Multi-factory pipeline (one DB)

Chain several `kg_build` steps targeting the same `kg_name` with `after:` — every factory
MERGEs into one database:

```yaml
workflows:
  rainbow_add_crm:
    description: "Rainbow reviews + architecture docs + CRM export"
    pipeline:
      - id: rainbow
        run: kg_build
        with:
          kg_name: rainbow_add_crm
          graph:
            factory: myapp.schema.rainbow_review.ReviewedOpportunityGraph
            md_root: '${paths.rainbow_md}'
            json_cache_root: '${paths.rainbow_json}'
      - id: arch_doc
        run: kg_build
        after: [rainbow]
        with:
          kg_name: rainbow_add_crm
          delete_first: false
          graph: {factory: myapp.schema.architecture_doc.ArchitectureDocumentGraph, data_root: '${paths.add_json}'}
      - id: crm
        run: kg_build
        after: [arch_doc]
        with:
          kg_name: rainbow_add_crm
          delete_first: false
          graph: {factory: myapp.schema.crm_export.CrmExtractGraph, files: ['${paths.ekg_data}/crm_export/report.xlsx']}
```

## Document-driven KG (markdownize + entities + document graph)

```yaml
workflows:
  docgraph_build:
    run: genai_graph.orchestration.workflow_steps.docgraph_build_step
    hidden: true
    defaults:
      markdownize_profile: fast
      build_document_graph: true
      delete_first: false
      export_html: true
    params:
      kg_name: {required: true}
      sources: {required: true}
      md_output_dir: {required: true}
      factories: {required: true}

  rainbow_extract:
    description: "PPT/PDF/Markdown -> opportunity entities + document graph"
    pipeline:
      - id: build
        run: docgraph_build
        with:
          kg_name: rainbow_extract
          sources: '${values.sources}'
          md_output_dir: '${paths.rainbow_md}'
          factories:
            - factory: myapp.schema.rainbow_review.ReviewedOpportunityGraph
              md_root: '${paths.rainbow_md}'
              json_cache_root: '${paths.rainbow_json}'
    defaults:
      sources: '${paths.rainbow_ppt}'
```

`docgraph_build_step` markdownizes (if `markdownize_profile` set), runs entity `factories`
into `kg_name`, then ingests the `Folder → Document → MarkdownSection` graph into the same
DB (`build_document_graph`, default true) — `Document` nodes MERGE by content hash. See
`kg-document-graph`.

## Sub-workflow composition

A pipeline step's `run:` can reference another workflow name — it expands in place with
steps prefixed `{step_id}.`:

```yaml
workflows:
  full_kg_pipeline:
    pipeline:
      - id: markdownize
        run: markdownize_documents
        with: {...}
      - id: create_kg
        run: rainbow_add_crm      # expands to create_kg.rainbow, create_kg.arch_doc, …
        after: [markdownize]
```

## CLI

```bash
# Low-level (full control)
uv run cli workflow run document_to_kg --dry-run
uv run cli workflow run document_to_kg --force md

# High-level shorthands
cli kg create one_rainbow                       # build/update a named KG profile
cli kg create one_rainbow --force parquet       # rebuild import caches
cli kg create one_rainbow --force all           # full clean rebuild
cli kg create one_rainbow --delete-first
cli kg create --all                             # every kg_* profile

cli docgraph run --workflow rainbow_extract                # configured default sources
cli docgraph run -w rainbow_extract -s ./some_file.pptx    # ad-hoc source override
cli docgraph run -w rainbow_extract --force md --dry-run

cli docgraph build ./docs --db ./data/kg/tree.db           # document graph only
```

`cli kg create <name>` and `cli docgraph run --workflow <name>` both resolve through
`resolve_workflow_invocation` + `execute_workflow`. Use `kg create` for a named, predefined
document set; `docgraph run` when pointing at a file/folder ad hoc.

## Change Workflow

1. genai-graph ships only domain-agnostic workflows (`kg_build`, `document_nodes`,
   `document_graph`, `office2pdf_documents`, `markdownize_documents`, `document_to_kg`).
   Domain-specific workflows (concrete factories, data paths) belong in the downstream
   project's `config/workflows/*.yaml`.
2. A downstream project that depends on genai-graph as a package must define its own full
   workflow entries (`run:`/`defaults:`/`params:`) for `office2pdf_documents` /
   `markdownize_documents`, not just presets.
3. New step functions get `@workflow(name=..., description=...)` and are referenced by
   dotted path in YAML; clear factory caches at the start.
4. After changing a workflow, run `uv run cli workflow validate` and `--dry-run`.

## Commands

```bash
uv run cli workflow list
uv run cli workflow run <name>[/<preset>] --dry-run
uv run cli kg create <name> --force graph
uv run cli workflow validate
uv run just test
```

## Avoid

- Do not reimplement workflow DSL/resolver logic in genai-graph — use `genai_tk.workflow`.
- Do not use the removed `--set delete_first=true` / `--set force_rebuild=true`; use
  `--delete-first` and `--force <stage>`.
- Do not split a multi-factory build across different `kg_name` values expecting one graph —
  chain steps on the same `kg_name` with `after:`.
- Do not ship domain-specific workflow YAML in genai-graph itself.

## Complements

- `genai-tk/workflow-engine` — the engine, DSL, resolver, `flow_from_yaml`, Prefect server.
- `kg-factories` / `kg-document-graph` — the factories these steps run.
- `kg-ingest` — `create_kg_flow` and the vector-index drop/recreate ordering.
- `kg-cli` — the `cli kg` / `cli docgraph` shorthands over these steps.
