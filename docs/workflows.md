# Workflows: YAML-Driven Task Orchestration

The **Workflow Engine** provides a YAML DSL that wraps Prefect flows into composable,
parameterised pipelines.  It sits on top of Prefect — every step is ultimately a Prefect
`@flow` invocation — but adds a declarative layer for chaining, parameterisation, and
caching without writing boilerplate Python.

**Core ideas:**

- Define **multi-step pipelines** in YAML without writing Python glue code
- Make Prefect flows **composable** and **reusable** across projects
- Bundle concrete parameter sets in named **presets** inside each workflow definition
- Chain steps with explicit **dependencies** (`after:`); independent steps run in parallel
- Use **named presets** to switch between data sources without duplicating workflow code
- Run with `--dry-run` to see the full plan (DAG, caches, values) before executing

> **See also:** [prefect.md](prefect.md) — how to write `@flow` functions, run them directly,
> and manage the local Prefect server.

---

## How the DSL Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│  YAML definition  (config/workflows/*.yaml)                             │
│                                                                         │
│  workflows:                                                             │
│    my_pipeline:                                                         │
│      pipeline:                                                          │
│        - id: step_a                                                     │
│          run: myapp.flows.flow_a      ← any @flow or plain callable     │
│          with:                                                          │
│            input: "${values.base_dir}"                                  │
│        - id: step_b                                                     │
│          run: myapp.flows.flow_b                                        │
│          after: [step_a]             ← dependency → parallel by default │
└─────────────────────────────────────────────────────────────────────────┘
         │
         │  cli workflow run my_pipeline/my_preset
         ▼
┌───────────────────┐   ┌───────────────────┐   ┌────────────────────┐
│  Resolver         │   │  Compiler          │   │  PrefectFlowFactory│
│  (merge values,   │──►│  (validate,        │──►│  (build @flow,     │
│   pick preset)    │   │   topological sort)│   │   submit tasks)    │
└───────────────────┘   └───────────────────┘   └────────────────────┘
```

**Execution path:**

1. `resolver.py` — parses the `workflow_name[/preset_name]` string, merges `defaults → preset → CLI --set`, validates required params.
2. `compiler.py` — compiles the `WorkflowDefV2` into a flat `CompiledWorkflow` (expands sub-workflows, resolves `${values.*}` references, topological sort).
3. `executor.py` — pre-flight checks keyword signatures, then calls `PrefectFlowFactory`.
4. `PrefectFlowFactory` — dynamically builds a Prefect `@flow` that submits each step as a `@task` with the correct `wait_for` dependencies, runs manifests checks for caching.

---

## Quick Start

```bash
# List all workflows and their presets
uv run cli workflow list

# Inspect a specific workflow (DAG, parameters, presets)
uv run cli workflow show markdownize

# Dry-run: resolve and print the plan, do not execute
uv run cli workflow run markdownize/docs --dry-run

# Execute a workflow preset
uv run cli workflow run markdownize/docs

# Execute with ad-hoc overrides on top of a preset
uv run cli workflow run markdownize/docs --set batch_size=10 --to /tmp/output

# Execute without a preset (uses defaults + your overrides)
uv run cli workflow run markdownize \
    --base-dir /data/pdfs --to /data/md --pathspec '**/*.pdf'

# Validate all workflow definitions
uv run cli workflow validate
```

---

## Built-in Workflows

The following workflows ship with genai-tk and are defined in `config/workflows.yaml`.
Run `cli workflow list` to see all of them with their presets.

| Workflow | Description | Key Defaults |
|----------|-------------|--------------|
| `markdownize` | Convert PDF/DOCX/PPTX/ODP → Markdown | `batch_size: 5`, `pdf_converter: mistral` |
| `ppt2pdf` | Convert PPT/PPTX/ODP → PDF via LibreOffice | `pathspecs: ["**/*.pptx", "**/*.ppt", "**/*.odp"]` |
| `rag_ingest` | Chunk, embed, and upsert documents into vector store | `force: false` |
| `baml_extract` | Structured extraction from Markdown via BAML | `function_name: required` |
| `anonymize` | PII anonymization with Presidio | `base_dir: required` |
| `baml_to_table` | BAML extraction + flatten to CSV/XLSX | `output_file: required` |

### Running a built-in workflow

```bash
# See the full list
uv run cli workflow list

# Inspect a workflow's DAG, params, and presets
uv run cli workflow show markdownize

# Run with a preset
uv run cli workflow run markdownize/docs

# Run with ad-hoc values (no preset needed)
uv run cli workflow run markdownize \
    --base-dir /data/pdfs --to /data/md --pathspec '**/*.pdf'
```

---

## Creating Your First Workflow

Follow these steps to add a new workflow to your project.

### Step 1: Write (or reuse) a `@flow` function

Any Python callable works.  If you're using a Prefect `@flow`:

```python
# myproject/flows/my_step.py
from prefect import flow, task
from pathlib import Path

@task
def process(src: Path, dest: Path) -> str:
    dest.write_text(src.read_text().upper())
    return str(dest)

@flow(name="uppercase-files")
def uppercase_flow(base_dir: str, output_dir: str) -> list[str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    futures = [process.submit(f, out / f.name) for f in Path(base_dir).glob("*.txt")]
    return [f.result() for f in futures]
```

### Step 2: Add a YAML workflow definition

Create or edit any YAML file under `config/` (e.g. `config/workflows/my_workflows.yaml`):

```yaml
workflows:
  uppercase_files:
    description: "Convert text files to uppercase"
    run: myproject.flows.my_step.uppercase_flow
    defaults:
      base_dir: "${paths.data_root}/in"
      output_dir: "${paths.data_root}/out"
    params:
      base_dir:  {required: true}
      output_dir: {required: true}
    presets:
      demo:
        base_dir: /tmp/demo_in
        output_dir: /tmp/demo_out
```

### Step 3: Validate and run

```bash
# Check the definition is correct (imports resolve, DAG is acyclic)
uv run cli workflow validate

# Dry-run: see what would happen
uv run cli workflow run uppercase_files/demo --dry-run

# Execute
uv run cli workflow run uppercase_files/demo
```

### Step 4 (optional): Use programmatically

```python
from genai_tk.workflow import PrefectFlowFactory

factory = PrefectFlowFactory.from_profile("uppercase_files/demo")
results = factory.run()
print(results)
```

Or inline with `flow_from_yaml` (no config directory needed):

```python
from genai_tk.workflow import flow_from_yaml

flow = flow_from_yaml("""
workflows:
  uppercase_files:
    run: myproject.flows.my_step.uppercase_flow
    defaults:
      base_dir: /tmp/in
      output_dir: /tmp/out
""")
flow()
```

---

## DSL Reference

### Workflow Definition

A workflow lives inside the top-level `workflows:` key in any YAML file auto-scanned under
`config/`.  There are two forms:

#### Single-step shorthand (`run:`)

```yaml
workflows:
  markdownize:
    description: "Convert documents to Markdown"
    run: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
    defaults:
      pdf_converter: mistral
      batch_size: 5
      base_dir: "${paths.data_root}"
      output_dir: "${paths.data_root}/markdown"
      pathspecs: null
    params:
      base_dir: {required: true}
      output_dir: {required: true}
    presets:
      docs:
        base_dir: "${paths.data_root}/documents"
        output_dir: "${paths.data_root}/markdown"
        pathspecs: ["**/*.pdf", "**/*.docx", "**/*.pptx"]
```

- `run` is a **dotted Python path** to a callable **or** the name of another workflow.
- `defaults` supplies fallback values for every parameter.  All keys are automatically
  passed as keyword arguments to the callable — no explicit `with:` mapping needed.
- `params` documents parameters (optional); used for validation and `cli workflow show`.
  Set `required: true` for params that must be provided via a preset or `--set`.
- `presets` are named bundles of concrete values that override `defaults`.

#### Multi-step pipeline (`pipeline:`)

```yaml
workflows:
  full_pipeline:
    description: "PPT → PDF → Markdown → KG"
    defaults:
      batch_size: 5
    pipeline:
      - id: ppt_to_pdf
        run: ppt2pdf              # Name of another workflow (sub-workflow)
        with:
          base_dir: "${paths.data_root}/ppts"
          output_dir: "${paths.data_root}/pdfs"

      - id: markdownize
        run: markdownize          # Sub-workflow
        after: [ppt_to_pdf]
        with:
          base_dir: "${paths.data_root}/pdfs"
          output_dir: "${paths.data_root}/md"
          pathspecs: ["**/*.pdf"]

      - id: create_kg
        run: myproject.steps.kg_create_step   # Direct callable
        after: [markdownize]
        with:
          source_dir: "${paths.data_root}/md"
```

### Step Fields

| Field | Alias | Purpose |
|-------|-------|---------|
| `id` | | Unique step identifier within this pipeline |
| `run` | | Dotted Python path **or** workflow name |
| `after` | `wait_for` | List of step IDs this step depends on |
| `with` | | Arguments passed to the callable (merged on top of sub-workflow defaults) |
| `cache` | | Cache policy: `none` (default), `manifest`, `hybrid` |
| `execution` | | Retry / failure policy |
| `foreach` | | Fan-out: run this step once per item in a list — see [`foreach:` Block](#foreach-block--fan-out--map) |

### `with:` Values

Arguments passed to the callable.  Supports `${values.*}` interpolation:

```yaml
with:
  base_dir: "${values.base_dir}"        # Resolved from workflow defaults/presets/CLI
  batch_size: 5                          # Literal value
  output_dir: "${paths.data_root}/out"  # Global config interpolation
```

For **single-step workflows**, `with:` is auto-wired from `defaults` — you do not need to
write it manually.  For **pipeline steps** referencing sub-workflows, provide explicit values.

### `cache:` Block

```yaml
cache: manifest          # shorthand (backend only)

cache:                   # full form
  backend: manifest      # none | manifest | hybrid
  key_include: [base_dir, pathspecs]   # restrict fingerprint to these keys
```

### `execution:` Block

```yaml
execution:
  on_failure: abort     # abort (default) | skip | continue
  retries: 2
  retry_delay_seconds: 30.0
  tags: [slow, gpu]
```

### `foreach:` Block — Fan-Out / Map

Run a step **once per item** in a collection.  All iterations are submitted concurrently
as Prefect tasks (bounded by the flow's `max_workers`).

```yaml
foreach:
  from: "${steps.prev_step.result.}"  # expression that resolves to a list
  as: item                             # name of the loop variable (default: item)
  concurrency_limit: 4                 # optional: cap parallel iterations
```

Reference the current iteration value in `with:` using `${item}`:

```yaml
- id: process_each
  run: my_module.process_item
  after: [produce]
  foreach:
    from: "${steps.produce.result.}"
    as: item
  with:
    value: "${item}"
```

**How it works:**

1. `from:` is resolved against already-collected step results.  The predecessor step is
   awaited eagerly before fan-out starts so the list is available immediately.
2. One Prefect task is submitted per item.  Tasks are independent and run concurrently.
3. The step's result in subsequent `${steps.<id>.result.}` references is a **list** — one
   entry per iteration, in submission order.

**Full example — produce a list then process each item:**

```yaml
workflows:
  fan_out_demo:
    description: "Produce a list, then process each item in parallel"
    pipeline:
      - id: produce
        run: my_module.produce_items    # returns ["item-0", "item-1", ...]
        with:
          count: 10

      - id: process_each
        run: my_module.process_item     # called once per item
        after: [produce]
        foreach:
          from: "${steps.produce.result.}"
          as: item
        with:
          value: "${item}"

      - id: summarise
        run: my_module.summarise        # receives list of all processed results
        after: [process_each]
        with:
          items: "${steps.process_each.result.}"
```

**Rules and constraints:**

- `from:` must resolve to a `list` or `tuple` at runtime — a `WorkflowExecutionError` is
  raised otherwise.
- `${item}` is only valid inside the `with:` block of the step that declares `foreach:`.
- Manifest caching is **not** applied to fan-out steps (the fingerprint is per-step, not
  per-item).
- `foreach:` and `pipeline:` (multi-step) are fully compatible; single-step shorthand
  (`run:`) does not support `foreach:`.

### `params:` Block

Documents parameter metadata used by `cli workflow show` and for early validation:

```yaml
params:
  base_dir: {required: true, description: "Root directory to scan"}
  output_dir: {required: true}
  batch_size: {}          # optional — has a default
```

### `presets:` Block

Named value bundles inside the workflow.  Select with `workflow_name/preset_name`:

```yaml
presets:
  rainbow:
    base_dir: "${paths.rainbow_pdf}"
    output_dir: "${paths.rainbow_md}"
    pathspecs: ["**/*.pdf"]
  rfq:
    base_dir: "${paths.rfq_pdf}"
    output_dir: "${paths.rfq_md}"
```

---

## Concepts

### Value Resolution Order

Values are merged from lowest to highest priority:

```
workflow defaults  →  preset values  →  CLI --set overrides
```

`${values.key}` in `with:` blocks is resolved against the merged result.

### Sub-Workflow Composition

A pipeline step can reference another workflow by name via `run:`.  The referenced workflow's
steps are **expanded in-place** and prefixed with `{step_id}.` to avoid ID collisions.

- Root steps of the sub-workflow (no internal `after:`) inherit the parent step's `after:`.
- When a later step lists an expanded step ID in its `after:`, the reference is automatically
  rewritten to the **terminal steps** (leaves) of that sub-workflow.
- Parent's `with:` is merged on top of sub-workflow's auto-wired defaults — explicit values win.
- Sub-workflows are recursive; cycles are detected and reported as errors.

**Example:**

```yaml
workflows:
  kg_build:
    run: myproject.steps.kg_build_step
    cache: manifest
    defaults:
      delete_first: false
      export_html: true
    params:
      graphs: {required: true}

  one_rainbow:
    pipeline:
      - id: build
        run: kg_build       # expands to build.run
        with:
          kg_name: one_rainbow
          graphs: [{source: "${paths.rainbow_json}"}]
```

`cli workflow run one_rainbow --dry-run` shows:

```
┌───────────┬──────────────────┬──────────┐
│ Id        │ Invoke           │ Wait For │
├───────────┼──────────────────┼──────────┤
│ build.run │ ...kg_build_step │ -        │
└───────────┴──────────────────┴──────────┘
```

### Python `@workflow` Decorator

Register a Python callable as a named workflow without YAML:

```python
from genai_tk.workflow import workflow

@workflow(
    name="my_step",
    description="My Python step",
    defaults={"batch_size": 5},
)
def my_step(input_dir: str, output_dir: str, batch_size: int = 5) -> dict:
    ...
```

Decorated callables appear in `cli workflow list` and can be used as `run:` targets by name
in YAML pipeline steps.

---

## CLI Reference

### `cli workflow list`

Show a table of all workflows, their presets, step count, and description.

```bash
uv run cli workflow list
```

### `cli workflow show NAME`

Inspect a single workflow: pipeline steps, defaults, and presets.

```bash
uv run cli workflow show markdownize
uv run cli workflow show full_pipeline
```

### `cli workflow run NAME[/PRESET] [OPTIONS]`

```bash
# Run with a named preset
uv run cli workflow run markdownize/docs

# Run without a preset (required params must be given via flags)
uv run cli workflow run markdownize --base-dir /data/pdfs --to /data/md

# Override individual values
uv run cli workflow run markdownize/docs --set batch_size=10

# Force recomputation (bypass caches)
uv run cli workflow run markdownize/docs --force

# Dry-run — resolve and print plan only
uv run cli workflow run markdownize/docs --dry-run
```

| Option | Purpose |
|--------|---------|
| `--set KEY=VALUE` | Override a value (repeatable) |
| `--pathspec / -p` | Gitwildmatch pattern → `values.pathspecs` (repeatable) |
| `--to` | Output directory → `values.output_dir` |
| `--base-dir` | Base directory → `values.base_dir` |
| `--force` | Set `force=True` and `force_rebuild=True` in values |
| `--dry-run` | Resolve and show the plan; do not execute |

### `cli workflow serve NAME[/PRESET] [OPTIONS]`

Register a workflow as a long-running Prefect **deployment** that polls for scheduled or
manually triggered runs.  Blocks the terminal until stopped with Ctrl-C.

```bash
# Serve a workflow (the Prefect server must be running)
uv run cli workflow serve markdownize/docs

# Give the deployment a custom name
uv run cli workflow serve markdownize/docs --name nightly-markdownize

# Schedule via cron
uv run cli workflow serve markdownize/docs --cron "0 2 * * *"

# Schedule at a fixed interval (seconds)
uv run cli workflow serve markdownize/docs --interval 3600
```

| Option | Purpose |
|--------|---------|
| `--name` | Deployment name shown in the Prefect UI |
| `--cron` | Cron schedule (e.g. `"0 2 * * *"`) |
| `--interval` | Interval schedule in seconds |

### `cli prefect`

Control the Prefect server daemon.  See [prefect.md](prefect.md) for full details.

```bash
cli prefect start       # start background daemon (auto-starts on workflow run)
cli prefect stop        # stop the daemon
cli prefect status      # show running state + URLs
cli prefect ui          # open the dashboard in a browser
```

### `cli workflow validate`

Validate every workflow definition: check Python targets are importable, DAGs are acyclic,
and all referenced presets resolve correctly.

```bash
uv run cli workflow validate
```

---

## Configuration

### File Location

Workflow definitions live in any YAML file under `config/` (auto-scanned on startup).
No manual registration needed:

```
config/
  workflows.yaml                    # genai-tk built-in workflows
  workflows/
    graph_construction.yaml         # project-specific workflow files
    data_injection.yaml
```

### Format

```yaml
# Optional: define path shortcuts used by workflows
paths:
  rainbow_md: "${paths.ekg_data}/rainbow/md/"
  rainbow_pdf: "${paths.ekg_data}/rainbow/pdf/"

workflows:

  markdownize:
    description: "Convert documents to Markdown"
    run: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
    defaults:
      pdf_converter: mistral
      batch_size: 5
      base_dir: "${paths.data_root}"
      output_dir: "${paths.data_root}/markdown"
      pathspecs: null
    params:
      base_dir: {required: true}
      output_dir: {required: true}
    presets:
      docs:
        base_dir: "${paths.data_root}/documents"
        output_dir: "${paths.data_root}/markdown"
        pathspecs: ["**/*.pdf", "**/*.docx"]

  full_pipeline:
    description: "End-to-end document pipeline"
    pipeline:
      - id: ppt_to_pdf
        run: ppt2pdf
        with:
          base_dir: "${paths.data_root}/ppts"
          output_dir: "${paths.data_root}/pdfs"
      - id: markdownize
        run: markdownize
        after: [ppt_to_pdf]
        with:
          base_dir: "${paths.data_root}/pdfs"
          output_dir: "${paths.data_root}/md"
    presets:
      rainbow:
        base_dir: "${paths.rainbow_ppt}"
        output_dir: "${paths.rainbow_md}"
```

---

## Common Patterns

### Single-Step with Presets

```yaml
workflows:
  ppt2pdf:
    description: "Convert PowerPoint files to PDF"
    run: genai_tk.workflow.prefect.flows.ppt2pdf_flow.ppt2pdf_flow
    defaults:
      batch_size: 5
    params:
      base_dir: {required: true}
      output_dir: {required: true}
    presets:
      marketing:
        base_dir: "${paths.data_root}/marketing/ppts"
        output_dir: "${paths.data_root}/marketing/pdfs"
      engineering:
        base_dir: "${paths.data_root}/engineering/ppts"
        output_dir: "${paths.data_root}/engineering/pdfs"
```

```bash
uv run cli workflow run ppt2pdf/marketing
uv run cli workflow run ppt2pdf/engineering --force
```

### Library Workflow (Parameterized Sub-Step)

```yaml
workflows:
  kg_build:                         # No presets — designed to be called via run:
    run: myproject.kg_build_step
    cache: manifest
    defaults:
      delete_first: false
      export_html: true
    params:
      graphs: {required: true}      # Must be provided by the pipeline step's with:

  one_rainbow:
    description: "Build rainbow KG"
    pipeline:
      - id: build
        run: kg_build
        with:
          kg_name: one_rainbow
          graphs: [{source: "${paths.rainbow_json}"}]
```

`cli workflow validate` shows `kg_build` as **"parameterized"** (requires: graphs) — valid
but not directly runnable without `--set graphs=...`.

### Multi-Level Composition

```yaml
  rainbow_and_crm:
    pipeline:
      - id: rainbow
        run: one_rainbow
      - id: crm
        run: crm_export             # Runs in parallel (no after:)
      - id: merge
        run: myproject.merge_step
        after: [rainbow, crm]       # after: is rewritten to leaf steps automatically
```

### Failure Handling

```yaml
workflows:
  resilient_pipeline:
    pipeline:
      - id: try_ocr
        run: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
        with:
          pdf_converter: mistral
        execution:
          on_failure: skip          # Mistral API down? Skip and continue

      - id: fallback_ocr
        run: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
        with:
          pdf_converter: markitdown
        execution:
          on_failure: abort         # Local converter failed? Abort
```

### `@workflow` Decorator

```python
# myproject/steps.py
from genai_tk.workflow import workflow

@workflow(
    name="kg_build",
    description="Build a knowledge graph from inline configs",
    defaults={"delete_first": False, "export_html": True},
)
def kg_build_step(
    graphs: list[dict],
    kg_name: str = "inline",
    delete_first: bool = False,
    export_html: bool = True,
    force_rebuild: bool = False,
) -> dict:
    ...
```

The decorated callable is registered globally and can be used as `run: kg_build` in YAML.

---

## Step Implementation Guide

### Plain Python Callable

Any Python function works as a `run:` target.  The engine passes resolved values as
keyword arguments:

```python
def my_transform_step(
    input_dir: str,
    output_dir: str,
    batch_size: int = 10,
) -> dict:
    # ... do work ...
    return {"processed": 42, "output_dir": output_dir}
```

```yaml
workflows:
  transform:
    run: myproject.steps.my_transform_step
    defaults:
      batch_size: 10
    params:
      input_dir: {required: true}
      output_dir: {required: true}
```

### Prefect Flow

`@flow`-decorated Prefect functions work identically as `run:` targets:

```python
from prefect import flow, task

@task
def process_file(path: str) -> str:
    return path.upper()

@flow
def my_flow(input_dir: str, output_dir: str, batch_size: int = 10) -> dict:
    futures = [process_file.submit(p) for p in Path(input_dir).iterdir()]
    return {"processed": len([f.result() for f in futures])}
```

---

## Troubleshooting

### "Workflow not found"

Check that the workflow name exists in `cli workflow list`.  The YAML file must be under
`config/` with a `workflows:` key.

### "Missing required parameter(s)"

The workflow has a `params` entry with `required: true` and no matching `defaults` key.
Supply it via a preset or CLI:

```bash
uv run cli workflow run kg_build --set graphs='[{source: /data/json}]'
uv run cli workflow run kg_build/my_preset
```

Run `cli workflow show <name>` to see which params are required.

### "Unexpected argument(s)"

A key in `defaults` or `with:` does not match the function's parameter names.  The pre-flight
check reports this before execution with the accepted parameter list and a fix hint.

### "Interpolation key 'paths.X' not found"

The path key is not defined in the merged config.  Define it under `paths:` in the workflow
YAML or in `config/app_conf.yaml`.

### "Cannot import step module"

Verify the dotted path is correct:

```bash
uv run python -c "from genai_tk.workflow.prefect.flows.markdownize_flow import markdownize_flow"
```

---

## Examples in the Repository

### genai-tk Built-in Workflows

`config/workflows.yaml` (auto-loaded):
- `markdownize` — PDF/DOCX/PPTX → Markdown; preset `docs`
- `ppt2pdf` — PPT → PDF; preset `docs`
- `rag_ingest` — ingest files into vector store; preset `docs`
- `anonymize` — PII removal; preset `docs`
- `anonymize_and_ingest` — pipeline: anonymize → RAG ingest
- `baml_extract` — extract structured data from files (Markdown/PDF); preset `default`
- `baml_run` — extract structured data from single text input; preset `default`
- `json_to_table` — convert JSON files in a directory to a single CSV/Excel table; preset `default`
- `baml_to_table` — 2-step pipeline: BAML extraction → tabularize to CSV/Excel; preset `default`

#### BAML Workflows

The `baml_extract` and `baml_run` workflows enable **composable, cacheable BAML extraction**
within the workflow engine:

```bash
# Extract structured data from a directory
uv run cli workflow run baml_extract/default --set base_dir=./docs --to ./output

# Extract from a single text input
uv run cli workflow run baml_run/default --set input_text="John Smith; 10 yrs Python"

# Compose in a multi-step pipeline
uv run cli workflow run full_pipeline/default  # if pipeline includes baml_extract step
```

Both workflows maintain manifests for incremental processing — unchanged files/inputs
are skipped on subsequent runs.  Use `--force` to re-process everything.

For full BAML documentation, see [docs/baml.md](baml.md).  CLI commands `cli baml run`
and `cli baml extract` delegate to these workflows.

#### JSON to Table Workflows

The `json_to_table` workflow converts a directory of JSON files (e.g. BAML extraction outputs)
to a single flat CSV or Excel file using pandas:

```bash
# Convert JSON outputs to CSV
uv run cli workflow run json_to_table \
    --set input_dir=./data/structured/Resume \
    --set output_file=./data/results.csv \
    --set keys='["name","skills","years_experience"]'

# Excel output (auto-detected by .xlsx extension)
uv run cli workflow run json_to_table/default \
    --set output_file=./data/results.xlsx

# With Pydantic model for validation/coercion
uv run cli workflow run json_to_table \
    --set input_dir=./data/structured/Resume \
    --set output_file=./data/results.csv \
    --set model=myapp.baml_client.types.Resume
```

The `baml_to_table` pipeline combines both steps — extract structured data then export to table:

```bash
# Full pipeline: Markdown → BAML extraction → CSV
uv run cli workflow run baml_to_table \
    --set base_dir=./docs \
    --set structured_dir=./data/structured \
    --set output_file=./data/results.csv \
    --set function_name=ExtractResume \
    --set keys='["name","skills","years_experience"]'

# Using the default preset with ad-hoc overrides
uv run cli workflow run baml_to_table/default \
    --set function_name=ExtractResume \
    --set output_file=./data/results.xlsx
```

**Key `json_to_table` parameters:**

| Parameter | Required | Description |
|---|---|---|
| `input_dir` | yes | Directory of JSON files to read |
| `output_file` | yes | Destination `.csv` or `.xlsx` path |
| `model` | no | Dotted Python path to a Pydantic model for validation |
| `keys` | no | List of columns to include (all keys used when omitted) |
| `pathspecs` | no | File filter patterns (default `["**/*.json"]`) |
| `sheet_name` | no | Excel sheet name (default `Sheet1`) |

### genai-graph Examples

- `config/workflows/graph_construction.yaml` — `kg_build` (library), `one_rainbow`, `learned` (3-level composition)
- `config/workflows/data_injection.yaml` — `ppt2pdf_documents`, `markdownize_documents`, `full_kg_pipeline`
- `genai_graph/orchestration/workflow_steps.py` — `@workflow`-decorated `kg_build_step`

---

## See Also

- [prefect.md](prefect.md) — Running `@flow` functions directly, managing the Prefect server, `flow_from_yaml`
- [configuration.md](configuration.md) — Config file format and OmegaConf interpolation
- [cli.md](cli.md) — Full CLI command reference
- [core.md](core.md) — Vector store / embeddings configuration (used by `rag_ingest`)
- [baml.md](baml.md) — BAML structured extraction (used by `baml_extract`)
- [AGENTS.md](../AGENTS.md) — Copilot agent coding guidelines
