# Workflows: YAML-Driven Task Orchestration

The **Workflow Engine** provides a YAML-based abstraction layer over Prefect that makes it simple to:

- Define **multi-step pipelines** without writing Python
- Make Prefect flows **composable** and **reusable**
- Chain pre-processing (ppt2pdf → markdownize) with domain logic (KG creation, RAG ingestion)
- Use **CLI `--set` flags** to override any parameter inline
- Run workflows with `--dry-run` to see the full execution plan before committing

---

## Quick Start

### List Available Workflows & Profiles

```bash
# Show all configured workflows and profiles
uv run cli workflow list

# Show just workflows or just profiles
uv run cli workflow list workflows
uv run cli workflow list profiles
```

### Resolve & Execute a Workflow

```bash
# Dry-run: resolve the workflow, show the plan, don't execute
uv run cli workflow run markdownize_docs --dry-run

# Execute: resolve and run the full workflow
uv run cli workflow run markdownize_docs

# Override values at the CLI
uv run cli workflow run markdownize_docs --set batch_size=10 --set converter=mistral
```

---

## DSL Reference

### Step Fields

| Field | Purpose | Example |
|-------|---------|---------|
| `id` | Unique step identifier | `ppt_to_pdf` |
| `invoke` | How to invoke the step — `kind` + `target` | see below |
| `with` | All arguments passed to the callable (flat dict) | `with: {base_dir: /path, batch_size: 5}` |
| `wait_for` | List of step IDs this step depends on | `wait_for: [ppt_to_pdf]` |
| `ref` | Reference a step template defined in `step_templates:` | `ref: markdownize_step` |
| `execution` | Retry / failure policy | `execution: {on_failure: abort, retries: 2}` |
| `foreach` | Fan-out: run this step once per item | `foreach: {from: items, as: item}` |

### `invoke` Block

```yaml
invoke:
  kind: flow        # flow | callable | workflow | task | deployment
  target: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
```

| `kind` | When to use |
|--------|-------------|
| `flow` | A `@flow`-decorated Prefect function |
| `callable` | Any plain Python function |
| `workflow` | Inline another workflow (sub-workflow expansion) |
| `task` | A Prefect `@task` |

### `with` Block

All arguments (formerly split across `inputs:` and `params:`) are now a single flat dict:

```yaml
with:
  base_dir: "${values.base_dir}"
  output_dir: "${values.output_dir}"
  batch_size: "${values.batch_size}"
  converter: mistral
```

Supports `${values.*}` interpolation from profile values and workflow defaults.

### `execution` Block

```yaml
execution:
  on_failure: abort     # abort (default) | skip | continue
  retries: 2
  retry_delay_seconds: 30.0
  tags: [slow, gpu]
```

### Value Placeholders

Use `${values.key}` in `with:` values. Resolution priority (highest first):

```
CLI --set  >  profile values  >  workflow defaults
```

---

## Concepts

### Workflow (`workflow:` in YAML)

A **workflow** is a DAG of **steps**.  Declare `defaults:` as fallback values for `${values.*}` placeholders:

```yaml
workflows:
  convert_and_ingest:
    description: "Convert PDFs to markdown, then ingest into RAG"
    defaults:
      batch_size: 5
      converter: mistral
    steps:
      - id: to_markdown
        invoke:
          kind: flow
          target: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
        with:
          base_dir: "${values.pdf_dir}"
          output_dir: "${values.md_dir}"
          pathspecs: "${values.pathspecs}"
          batch_size: "${values.batch_size}"
          converter: "${values.converter}"

      - id: ingest
        invoke:
          kind: flow
          target: genai_tk.workflow.prefect.flows.rag_flow.rag_file_ingestion_flow
        wait_for: [to_markdown]
        with:
          base_dir: "${values.md_dir}"
          retriever_name: "${values.retriever}"
          pathspecs: "${values.pathspecs}"
```

### Profile (`workflow_profiles:` in YAML)

A **profile** binds a workflow to concrete values:

```yaml
workflow_profiles:
  marketing_docs:
    workflow: convert_and_ingest
    values:
      pdf_dir: "${paths.data_root}/marketing/pdfs"
      md_dir: "${paths.data_root}/marketing/markdown"
      batch_size: 5
      retriever: marketing_embeddings
      pathspecs:
        - "**/*.pdf"
```

### Step Templates (`step_templates:` in YAML)

A **step template** is a reusable step definition. Reference it in a workflow step with `ref:`.
Step-level fields override template fields; the `with` dict is merged key-by-key.

```yaml
step_templates:
  ingest_step:
    invoke:
      kind: flow
      target: genai_tk.workflow.prefect.flows.rag_flow.rag_file_ingestion_flow
    with:
      base_dir: "${values.base_dir}"
      retriever_name: "${values.retriever_name}"
      pathspecs: "${values.pathspecs}"
      batch_size: "${values.batch_size}"
      max_chunk_tokens: "${values.chunk_size}"
      chunker_name: "${values.chunker}"

workflows:
  rag_add_files:
    steps:
      - id: ingest
        ref: ingest_step

  anonymize_and_ingest:
    steps:
      - id: anonymize
        ref: anonymize_step
        with:
          output_dir: "${values.anon_dir}"   # Override template's output_dir

      - id: ingest
        ref: ingest_step
        wait_for: [anonymize]
        with:
          base_dir: "${values.anon_dir}"     # Chain output from prior step
```

### Sub-Workflows (`invoke: {kind: workflow}`)

A step can **inline another workflow** by setting `invoke.kind: workflow` and `invoke.target`
to the workflow name.  The referenced workflow's steps are expanded in place, prefixed with
`{step_id}.` to avoid ID collisions.

**How DAG wiring works:**

- Root steps of the sub-workflow (those with no internal `wait_for`) inherit the parent step's `wait_for`.
- Internal dependencies within the sub-workflow are preserved and prefixed.
- When a step's `wait_for` references a parent step that was expanded via `kind: workflow`,
  the reference is automatically rewritten to the **terminal steps** (leaves) of that sub-workflow.

**Example:**

```yaml
workflows:
  ingest_pdfs:
    steps:
      - id: convert
        invoke:
          kind: flow
          target: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
        with:
          base_dir: "${values.pdf_dir}"
          output_dir: "${values.md_dir}"
          pathspecs: ["**/*.pdf"]

      - id: index
        invoke:
          kind: flow
          target: genai_tk.workflow.prefect.flows.rag_flow.rag_file_ingestion_flow
        wait_for: [convert]
        with:
          base_dir: "${values.md_dir}"
          retriever_name: "${values.retriever}"

  full_knowledge_pipeline:
    steps:
      - id: pdfs
        invoke:
          kind: workflow
          target: ingest_pdfs        # Expands to: pdfs.convert → pdfs.index

      - id: analyze
        invoke:
          kind: callable
          target: myproject.steps.run_analysis
        wait_for: [pdfs]             # Resolved to pdfs.index (terminal step)
```

`cli workflow run full_knowledge_pipeline --dry-run` shows:

```
┌──────────────┬────────────────────────────┬──────────────┐
│ Id           │ Invoke                     │ Wait For     │
├──────────────┼────────────────────────────┼──────────────┤
│ pdfs.convert │ ...markdownize_flow        │ -            │
│ pdfs.index   │ ...rag_file_ingestion_flow │ pdfs.convert │
│ analyze      │ ...run_analysis            │ pdfs.index   │
└──────────────┴────────────────────────────┴──────────────┘
```

Sub-workflows are recursive.  Cycles are detected and reported as errors at load time.

---

## Configuration

### File Location

```
config/
  app_conf.yaml       # Main config (lists which workflow files to include)
  workflows.yaml      # Workflow + profile definitions (imported via :merge:)
```

### Including Workflow Files

In `config/app_conf.yaml`, add the workflow file to the `:merge:` list:

```yaml
:merge:
  - ${paths.config}/baseline.yaml
  - ${paths.config}/workflows.yaml    # ← Add this line
```

### Workflow File Format

```yaml
# Optional: reusable step building blocks
step_templates:
  my_step:
    invoke:
      kind: flow
      target: module.path.to_flow
    with:
      arg1: "${values.arg1}"

# Workflow definitions
workflows:
  my_workflow:
    description: "Multi-step workflow"
    defaults:
      arg1: default_value
    steps:
      - id: step1
        ref: my_step

      - id: step2
        invoke:
          kind: flow
          target: module.path.to_other_flow
        wait_for: [step1]
        with:
          input: "${values.input}"

# Profiles — named, concrete invocations
workflow_profiles:
  my_profile:
    workflow: my_workflow
    values:
      arg1: /path/to/data
      input: /path/to/other/data
```

---

## Common Patterns

### Single-Step Workflow

```yaml
workflows:
  markdownize:
    defaults:
      converter: mistral
      batch_size: 5
    steps:
      - id: convert
        ref: markdownize_step

workflow_profiles:
  marketing_pdfs:
    workflow: markdownize
    values:
      base_dir: /data/marketing/pdfs
      output_dir: /data/marketing/markdown
      pathspecs: ["**/*.pdf"]

  engineering_docs:
    workflow: markdownize
    values:
      base_dir: /data/engineering/docs
      output_dir: /data/engineering/markdown
      pathspecs: ["**/*.docx", "**/*.txt"]
      converter: markitdown           # Override default
```

### Multi-Step Pipeline

```yaml
workflows:
  full_pipeline:
    defaults:
      batch_size: 5
    steps:
      - id: ppt_to_pdf
        invoke:
          kind: flow
          target: genai_tk.workflow.prefect.flows.ppt2pdf_flow.ppt2pdf_flow
        with:
          base_dir: "${values.ppt_dir}"
          output_dir: "${values.pdf_dir}"
          batch_size: "${values.batch_size}"

      - id: pdf_to_markdown
        invoke:
          kind: flow
          target: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
        wait_for: [ppt_to_pdf]
        with:
          base_dir: "${values.pdf_dir}"
          output_dir: "${values.md_dir}"
          pathspecs: ["**/*.pdf"]
          batch_size: "${values.batch_size}"

      - id: ingest_to_rag
        invoke:
          kind: flow
          target: genai_tk.workflow.prefect.flows.rag_flow.rag_file_ingestion_flow
        wait_for: [pdf_to_markdown]
        with:
          base_dir: "${values.md_dir}"
          retriever_name: "${values.retriever}"
          pathspecs: ["**/*.md"]

workflow_profiles:
  production:
    workflow: full_pipeline
    values:
      ppt_dir: /production/ppts
      pdf_dir: /production/pdfs
      md_dir: /production/markdown
      batch_size: 10
      retriever: production_search
```

### Conditional Execution with `execution.on_failure`

```yaml
workflows:
  resilient_pipeline:
    steps:
      - id: try_mistral_ocr
        invoke:
          kind: flow
          target: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
        with:
          base_dir: "${values.pdf_dir}"
          output_dir: "${values.md_dir}"
          pathspecs: ["**/*.pdf"]
          converter: mistral
        execution:
          on_failure: skip      # If Mistral API fails, skip and continue

      - id: fallback_ocr
        invoke:
          kind: flow
          target: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
        with:
          base_dir: "${values.pdf_dir}"
          output_dir: "${values.md_dir}"
          pathspecs: ["**/*.pdf"]
          converter: markitdown
        execution:
          on_failure: abort     # If markitdown fails, stop the whole workflow
```

---

## CLI Reference

### `cli workflow list [KIND]`

```bash
uv run cli workflow list              # Show all workflows and profiles
uv run cli workflow list workflows    # Show only workflows
uv run cli workflow list profiles     # Show only profiles
```

### `cli workflow run NAME [--set KEY=VAL ...] [--force] [--dry-run]`

| Option | Purpose |
|--------|---------|
| `--set KEY=VALUE` | Override a value; can repeat |
| `--force` | Force recomputation (passed to steps as `force=True`) |
| `--dry-run` | Resolve the workflow and show the plan; don't execute |

```bash
# Show the execution plan for a profile
uv run cli workflow run marketing_docs --dry-run

# Execute with value overrides
uv run cli workflow run my_pipeline --set batch_size=20 --set converter=mistral

# Force-rebuild (bypass caches)
uv run cli workflow run my_pipeline --force
```

---

## Step Implementation Guide

### Plain Python Step

```python
# myproject/steps.py

def my_transform_step(
    input_dir: str,
    output_dir: str,
    batch_size: int = 10,
) -> dict:
    """Transform documents from input to output directory."""
    # ... do work ...
    return {"processed": 42, "output_dir": output_dir}
```

```yaml
workflows:
  my_pipeline:
    steps:
      - id: transform
        invoke:
          kind: callable
          target: myproject.steps.my_transform_step
        with:
          input_dir: "${values.input}"
          output_dir: "${values.output}"
          batch_size: "${values.batch_size}"
```

### Prefect Flow Step

```python
# myproject/flows.py
from prefect import flow, task

@task
def load_documents(root_dir: str):
    return [...]

@flow(name="my_transform_flow")
def my_transform_flow(
    input_dir: str,
    output_dir: str,
    batch_size: int = 10,
) -> dict:
    docs = load_documents(input_dir)
    # ... process and save to output_dir ...
    return {"processed": len(docs), "output_dir": output_dir}
```

```yaml
workflows:
  my_pipeline:
    steps:
      - id: transform
        invoke:
          kind: flow
          target: myproject.flows.my_transform_flow
        with:
          input_dir: "${values.input}"
          output_dir: "${values.output}"
          batch_size: "${values.batch_size}"
```

---

## Examples in the Repository

### genai-tk Examples

- **Markdownize profile:** [config/workflows.yaml](../config/workflows.yaml) — `markdownize_docs` profile
- **RAG ingestion profile:** [config/workflows.yaml](../config/workflows.yaml) — `rag_ingest_docs` profile
- **Anonymize + ingest:** [config/workflows.yaml](../config/workflows.yaml) — `anonymize_and_ingest_docs` profile

### genai-graph Examples

- **KG creation profiles:** [config/ekg_workflows.yaml](../../genai-graph/config/ekg_workflows.yaml) — `kg_one_rainbow`, `kg_stratnav_subset_rainbow_crm`, `kg_learned`
- **Sub-workflow composition:** [config/ekg_workflows.yaml](../../genai-graph/config/ekg_workflows.yaml) — `stratnav_subset_rainbow_crm` (two `invoke: {kind: workflow}` steps)
- **Full pipeline:** [config/workflows.yaml](../../genai-graph/config/workflows.yaml) — `full_rainbow_pipeline` (ppt2pdf → markdownize → KG creation)
- **Step wrapper:** [genai_graph/orchestration/workflow_steps.py](../../genai-graph/genai_graph/orchestration/workflow_steps.py) — `kg_build_step()` accepts inline graph configs

---

## Advanced Topics

### Interpolation & Configuration

All values support OmegaConf **interpolation**:

```yaml
workflow_profiles:
  my_profile:
    values:
      root_dir: "${paths.data_root}"        # From global config paths.*
      output_dir: "${paths.data_root}/out"  # Combine with static text
      batch_size: "${oc.env:BATCH_SIZE,5}"  # From environment, default 5
```

### Topological Sorting

The workflow engine automatically topologically sorts steps based on `wait_for:` dependencies.
Circular dependencies are detected and reported as errors.

### Error Handling

Steps can be configured to fail fast (`on_failure: abort`), skip on error (`skip`), or continue
anyway (`continue`).  The overall workflow result includes per-step failure info for debugging.

---

## Troubleshooting

### "Workflow not found"

Ensure the workflow file is merged into `app_conf.yaml`:

```yaml
:merge:
  - ${paths.config}/workflows.yaml  # ← Ensure this is listed
```

### "Profile points to unknown workflow"

Check that `workflow:` in the profile matches a defined workflow name.

### "Interpolation key 'values.X' not found"

Ensure `x` is defined in the profile's `values:` or the workflow's `defaults:`:

```yaml
workflow_profiles:
  my_profile:
    workflow: my_workflow
    values:
      x: some_value  # ← Required if no workflow default exists
```

### "Cannot import step module"

Verify the `invoke.target` dotted path is correct and the module is importable:

```bash
uv run python -c "from genai_tk.workflow.prefect.flows.markdownize_flow import markdownize_flow"
```

---

## See Also

- [CLI Reference](cli.md) — Full CLI command reference
- [Prefect Documentation](prefect.md) — Detailed Prefect integration guide
- [Configuration](configuration.md) — Config file format and interpolation
- [AGENTS.md](../AGENTS.md) — Copilot agent coding guidelines
