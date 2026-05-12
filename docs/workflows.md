# Workflows: YAML-Driven Task Orchestration

The **Workflow Engine** provides a YAML-based abstraction layer over Prefect that makes it simple to:

- Define **multi-step pipelines** without writing Python
- Make Prefect flows **composable** and **reusable**
- Chain pre-processing (ppt2pdf → markdownize) with domain logic (KG creation, RAG ingestion)
- Use **CLI `--config`/`--profile` flags** to invoke pipelines from plain English names
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

# Use --profile to specify a profile for a named workflow
uv run cli workflow run markdownize_documents --profile my_docs
```

### Use Workflows via Command Shorthands

Instead of `cli workflow run`, convenience commands map directly to profiles:

```bash
# These are equivalent:
uv run cli workflow run markdownize_docs --dry-run
uv run cli tools markdownize --config markdownize_docs --dry-run

uv run cli workflow run rag_ingest_docs
uv run cli rag add-files --config rag_ingest_docs
```

---

## Concepts

### Workflow (`workflow:` in YAML)

A **workflow** is a DAG (directed acyclic graph) of **steps**.  Each step has:

| Field | Purpose | Example |
|-------|---------|---------|
| `id` | Unique step identifier | `ppt_to_pdf` |
| `uses` | Dotted Python path to a flow or function | `genai_tk.workflow.prefect.flows.ppt2pdf_flow.ppt2pdf_flow` |
| `inputs` | Static inputs passed to the flow | `{"base_dir": "/path/to/ppts"}` |
| `params` | Parameters (CLI flags, options) | `{"batch_size": 5, "force": false}` |
| `needs` | List of step IDs this depends on | `[ppt_to_pdf]` (execute after ppt_to_pdf) |
| `concurrency` | `serial` or `parallel` | `serial` (default: `auto`) |
| `on_failure` | `abort` (fail fast), `skip`, or `continue` | `abort` (default) |

**Workflow Example:**

```yaml
workflows:
  convert_and_ingest:
    description: "Convert PDFs to markdown, then ingest into RAG"
    steps:
      - id: to_markdown
        uses: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
        inputs:
          base_dir: "${profile.pdf_dir}"
          output_dir: "${profile.md_dir}"
        params:
          pathspecs: "${profile.pathspecs}"
          batch_size: "${profile.batch_size}"
        concurrency: serial

      - id: ingest
        uses: genai_tk.workflow.prefect.flows.rag_flow.rag_file_ingestion_flow
        needs: [to_markdown]  # Run after 'to_markdown'
        inputs:
          base_dir: "${profile.md_dir}"
        params:
          retriever_name: "${profile.retriever}"
          pathspecs: "${profile.pathspecs}"
        concurrency: serial
```

### Profile (`workflow_profiles:` in YAML)

A **profile** binds a **workflow** to concrete **values**.  Profiles allow:

- Parameterizing workflows (e.g., different data directories for different teams)
- Shorthand CLI invocation (name instead of workflow + args)
- Configuration interpolation (e.g., `${paths.data_root}`)

**Profile Example:**

```yaml
workflow_profiles:
  marketing_docs:
    workflow: convert_and_ingest      # Points to 'convert_and_ingest' workflow
    values:
      pdf_dir: "${paths.data_root}/marketing/pdfs"
      md_dir: "${paths.data_root}/marketing/markdown"
      batch_size: 5
      retriever: marketing_embeddings
```

### Step Templates (`step_templates:` in YAML)

A **step template** is a reusable step definition shared across multiple workflows. Define
templates once in `step_templates:`, then reference them in workflow steps with `ref:`.

Step-level fields **override** template fields. For dict fields (`inputs`, `params`, `outputs`)
the merge happens at key level — the step adds or replaces individual keys while keeping the rest
from the template.

**Step Template Example:**

```yaml
step_templates:
  ingest_step:
    uses: genai_tk.workflow.prefect.flows.rag_flow.rag_file_ingestion_flow
    inputs:
      base_dir: "${profile.base_dir}"
    params:
      retriever_name: "${profile.retriever_name}"
      pathspecs: "${profile.pathspecs}"
      batch_size: "${profile.batch_size}"
      max_chunk_tokens: "${profile.chunk_size}"
      chunker_name: "${profile.chunker}"
    concurrency: serial

workflows:
  # Simple workflow: use the template as-is
  rag_add_files:
    steps:
      - id: ingest
        ref: ingest_step

  # Composed workflow: same template with an input override
  anonymize_and_ingest:
    steps:
      - id: anonymize
        ref: anonymize_step
        inputs:
          output_dir: "${profile.anon_dir}"   # Override template's output_dir

      - id: ingest
        ref: ingest_step
        needs: [anonymize]
        inputs:
          base_dir: "${profile.anon_dir}"     # Chain output from prior step
```

### Workflow Defaults (`defaults:` in workflows)

A workflow can declare **default values** for any `${profile.*}` placeholder via `defaults:`.
These are the lowest-priority values — overridden by profile values, which are in turn overridden
by CLI `--set` flags:

```
priority (highest first):  CLI --set  >  profile values  >  workflow defaults
```

```yaml
workflows:
  rag_add_files:
    defaults:
      batch_size: 10
      chunk_size: 512
      chunker: auto
    steps:
      - id: ingest
        ref: ingest_step

workflow_profiles:
  rag_ingest_docs:
    workflow: rag_add_files
    values:
      base_dir: "${paths.data_root}/markdown"
      retriever_name: default
      pathspecs:
        - "**/*.md"
      # batch_size, chunk_size, chunker come from workflow defaults — no need to repeat them
```

### Step Inputs & Params

**Inputs** are passed directly to the flow as `**kwargs`:

```yaml
inputs:
  base_dir: /path/to/docs       # becomes base_dir=/path/to/docs
  output_dir: /path/to/output   # becomes output_dir=/path/to/output
```

**Params** are also passed as `**kwargs`:

```yaml
params:
  batch_size: 10           # becomes batch_size=10
  converter: markitdown    # becomes converter=markitdown
```

Both support **placeholder substitution** via `${profile.*}`:

```yaml
inputs:
  root_dir: "${profile.data_root}"      # Resolved from profile values
```

---

## Configuration

### File Location

Workflows are defined in **one or more YAML files** under `config/`:

```
config/
  app_conf.yaml           # Main config (lists which workflow files to include)
  workflows.yaml          # Workflow + profile definitions (imported via :merge:)
  baseline.yaml
  overrides.yaml
  ...
```

### Including Workflow Files

In `config/app_conf.yaml`, add the workflow file to the `:merge:` list:

```yaml
:merge:
  - ${paths.config}/baseline.yaml
  - ${paths.config}/workflows.yaml    # ← Add this line
```

Or reference it from a section that's already merged.

### Workflow File Format

```yaml
# Workflow definitions
workflows:
  my_workflow:
    description: "Multi-step workflow"
    steps:
      - id: step1
        uses: module.path.to_flow
        inputs: {...}
        params: {...}

      - id: step2
        uses: module.path.to_other_flow
        needs: [step1]
        inputs: {...}

# Profile definitions
workflow_profiles:
  my_profile:
    workflow: my_workflow              # Reference the workflow
    values:
      # Values that get substituted into ${profile.*} placeholders
      data_root: /path/to/data
      batch_size: 5
```

---

## Common Patterns

### Single-Step Workflow (Convenience Pattern)

Define a workflow with just one step, then create profiles for different configs:

```yaml
workflows:
  markdownize:
    steps:
      - id: convert
        uses: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
        inputs:
          base_dir: "${profile.base_dir}"
          output_dir: "${profile.output_dir}"
        params:
          pathspecs: "${profile.pathspecs}"
          converter: "${profile.converter}"

workflow_profiles:
  marketing_pdfs:
    workflow: markdownize
    values:
      base_dir: /data/marketing/pdfs
      output_dir: /data/marketing/markdown
      pathspecs:
        - "**/*.pdf"
      converter: mistral

  engineering_docs:
    workflow: markdownize
    values:
      base_dir: /data/engineering/docs
      output_dir: /data/engineering/markdown
      pathspecs:
        - "**/*.docx"
        - "**/*.txt"
      converter: markitdown
```

Usage:
```bash
uv run cli workflow run marketing_pdfs --dry-run
uv run cli workflow run engineering_docs
```

### Multi-Step Pipeline (Chained Inputs/Outputs)

Chain steps where output from one feeds into the next via step dependencies and shared values:

```yaml
workflows:
  full_pipeline:
    steps:
      - id: ppt_to_pdf
        uses: genai_tk.workflow.prefect.flows.ppt2pdf_flow.ppt2pdf_flow
        inputs:
          base_dir: "${profile.ppt_dir}"
          output_dir: "${profile.pdf_dir}"
        params:
          batch_size: "${profile.batch_size}"

      - id: pdf_to_markdown
        uses: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
        needs: [ppt_to_pdf]               # Run after ppt_to_pdf
        inputs:
          base_dir: "${profile.pdf_dir}"  # Output dir from previous step
          output_dir: "${profile.md_dir}"
        params:
          pathspecs:
            - "**/*.pdf"
          batch_size: "${profile.batch_size}"

      - id: ingest_to_rag
        uses: genai_tk.workflow.prefect.flows.rag_flow.rag_file_ingestion_flow
        needs: [pdf_to_markdown]
        inputs:
          base_dir: "${profile.md_dir}"
        params:
          retriever_name: "${profile.retriever}"
          pathspecs:
            - "**/*.md"

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

Usage:
```bash
# See the full 3-step plan
uv run cli workflow run production --dry-run

# Execute all 3 steps in order
uv run cli workflow run production
```

### Conditional Execution with `on_failure`

Steps can gracefully degrade on failure:

```yaml
workflows:
  resilient_pipeline:
    steps:
      - id: try_mistral_ocr
        uses: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
        inputs:
          base_dir: "${profile.pdf_dir}"
          output_dir: "${profile.md_dir}"
        params:
          pathspecs: ["**/*.pdf"]
          converter: mistral
        on_failure: skip        # If Mistral API fails, skip and continue

      - id: fallback_ocr
        uses: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
        inputs:
          base_dir: "${profile.pdf_dir}"
          output_dir: "${profile.md_dir}"
        params:
          pathspecs: ["**/*.pdf"]
          converter: markitdown  # Use markitdown as fallback
        on_failure: abort        # If markitdown fails, stop the whole workflow
```

---

## CLI Reference

### `cli workflow list [KIND]`

List all configured workflows and/or profiles.

```bash
uv run cli workflow list              # Show all workflows and profiles
uv run cli workflow list workflows    # Show only workflows
uv run cli workflow list profiles     # Show only profiles
```

Output:

```
        Workflows         
┏━━━━━━━━━━━━━━━━━━━━━━┓
│ my_workflow           │
│ convert_and_ingest    │
└───────────────────────┘

       Profiles          
┏━━━━━━━━━━━━━━━━━━━━┓
│ marketing_docs      │
│ engineering_docs    │
└─────────────────────┘
```

### `cli workflow run NAME [--profile PROFILE] [--set KEY=VAL ...] [--force] [--dry-run]`

Resolve and execute (or dry-run) a workflow or profile.

**Arguments:**

- `NAME` — Workflow name or profile name

**Options:**

| Option | Purpose |
|--------|---------|
| `--profile PROFILE` | Use a profile when NAME is a workflow (disambiguates when both exist) |
| `--set KEY=VALUE` | Override a value (e.g., `--set batch_size=20`); can repeat |
| `--force` | Force recomputation (passed to steps as `force=True`) |
| `--dry-run` | Resolve the workflow and show the plan; don't execute |

**Examples:**

```bash
# Resolve a profile, show the full execution plan
uv run cli workflow run marketing_docs --dry-run

# Execute a workflow with a specific profile
uv run cli workflow run markdownize --profile engineering_docs

# Override a value at the command line
uv run cli workflow run my_pipeline --set batch_size=20 --set converter=mistral

# Use --force to bypass caches and re-run all steps
uv run cli workflow run my_pipeline --force
```

---

## Integration with Command Shorthands

**Convenience commands** (like `cli tools markdownize`, `cli rag add-files`) can accept `--config` and `--dry-run` to integrate with the workflow system:

```bash
# Old style: direct arguments
uv run cli tools markdownize /input /output --converter mistral

# New style: profile + workflow resolution
uv run cli tools markdownize --config marketing_pdfs --dry-run

# CLI overrides are merged with profile values
uv run cli tools markdownize --config marketing_pdfs --set batch_size=20
```

The command resolves the profile, merges CLI overrides, and falls back to the original Prefect flow if no `--config` is provided.

---

## Step Implementation Guide

### Creating a Workflow Step

A workflow step is any function or Prefect flow that can be invoked with keyword arguments.

**Minimal Example:**

```python
# myproject/steps.py

def my_transform_step(
    input_dir: str,
    output_dir: str,
    batch_size: int = 10,
) -> dict:
    """Transform documents from input to output directory."""
    print(f"Processing {input_dir} → {output_dir} (batch={batch_size})")
    # ... do work ...
    return {
        "processed": 42,
        "output_dir": output_dir,
    }
```

**Register it in a workflow:**

```yaml
workflows:
  my_pipeline:
    steps:
      - id: transform
        uses: myproject.steps.my_transform_step
        inputs:
          input_dir: "${profile.input}"
          output_dir: "${profile.output}"
        params:
          batch_size: "${profile.batch_size}"

workflow_profiles:
  default:
    workflow: my_pipeline
    values:
      input: /data/input
      output: /data/output
      batch_size: 5
```

### Creating a Prefect Flow Step

For observability and retryability, wrap your step as a Prefect flow:

```python
# myproject/flows.py

from prefect import flow, task

@task
def load_documents(root_dir: str):
    """Load documents from directory."""
    return [...]

@task
def process_batch(docs, batch_size):
    """Process documents in batches."""
    return [...]

@flow(name="my_transform_flow")
def my_transform_flow(
    input_dir: str,
    output_dir: str,
    batch_size: int = 10,
) -> dict:
    """Multi-task flow."""
    docs = load_documents(input_dir)
    results = process_batch(docs, batch_size)
    # ... save results to output_dir ...
    return {"processed": len(results), "output_dir": output_dir}
```

Register the same way:

```yaml
workflows:
  my_pipeline:
    steps:
      - id: transform
        uses: myproject.flows.my_transform_flow
        inputs:
          input_dir: "${profile.input}"
          output_dir: "${profile.output}"
        params:
          batch_size: "${profile.batch_size}"
```

---

## Examples in the Repository

### genai-tk Examples

- **Markdownize profile:** [config/workflows.yaml](../config/workflows.yaml) — `markdownize_docs` and `markdownize_rfq` profiles
- **RAG ingestion profile:** [config/workflows.yaml](../config/workflows.yaml) — `rag_ingest_docs` profile
- **Multi-step pipeline:** [config/workflows.yaml](../config/workflows.yaml) — `full_kg_pipeline` (ppt2pdf → markdownize → kg create)

### genai-graph Examples

- **KG creation profiles:** [config/workflows.yaml](../config/workflows.yaml) — `kg_one_rainbow`, `kg_stratnav_subset_rainbow_crm`
- **Full pipeline:** [config/workflows.yaml](../config/workflows.yaml) — `full_rainbow_pipeline` (3-step chained workflow)
- **Step wrapper:** [genai_graph/orchestration/workflow_steps.py](../genai_graph/orchestration/workflow_steps.py) — `kg_create_step()` integrates KG creation into workflow engine

---

## Advanced Topics

### Interpolation & Configuration

All values in workflows support OmegaConf **interpolation**:

```yaml
workflow_profiles:
  my_profile:
    values:
      root_dir: "${paths.data_root}"        # From global config paths.*
      output_dir: "${paths.data_root}/out"  # Combine with static text
      batch_size: "${oc.env:BATCH_SIZE,5}"  # From environment, default 5
```

### Topological Sorting

The workflow engine automatically **topologically sorts steps** based on `needs:` dependencies.  Circular dependencies are detected and reported as errors.

### Error Handling

Steps can be configured to fail fast (`on_failure: abort`), skip on error (`skip`), or continue anyway (`continue`).  The overall workflow result includes per-step failure info for debugging.

---

## Troubleshooting

### "Workflow not found"

Make sure the workflow is defined in a file that's merged into `app_conf.yaml`:

```yaml
# app_conf.yaml
:merge:
  - ${paths.config}/workflows.yaml  # ← Ensure this is listed
```

### "Profile points to unknown workflow"

Check that the `workflow:` key in the profile matches a defined workflow name:

```yaml
workflow_profiles:
  my_profile:
    workflow: my_workflow  # ← Must exist in workflows: section
```

### "Interpolation key 'profile.X' not found"

When a step uses `${profile.x}`, ensure `x` is defined in the profile's `values:`:

```yaml
workflow_profiles:
  my_profile:
    workflow: my_workflow
    values:
      x: some_value  # ← Make sure this is here
```

### "Cannot import step module"

Verify the `uses:` dotted path is correct and the module is importable:

```bash
# Test the import
uv run python -c "from genai_tk.workflow.prefect.flows.markdownize_flow import markdownize_flow"
```

---

## See Also

- [CLI Reference](cli.md) — Full CLI command reference
- [Prefect Documentation](prefect.md) — Detailed Prefect integration guide
- [Configuration](configuration.md) — Config file format and interpolation
- [AGENTS.md](../AGENTS.md) — Copilot agent coding guidelines
