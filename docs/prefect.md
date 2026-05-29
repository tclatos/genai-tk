# Prefect Workflow Orchestration

GenAI Toolkit uses [Prefect](https://docs.prefect.io/) for orchestrating multi-step, long-running
tasks — document conversion, RAG ingestion, structured extraction — so that each step is
observable, retryable, and optionally parallelised.

Flows connect to a **locally-managed Prefect server** that is started and stopped explicitly
via `cli prefect`.  This avoids the lock contention and stale-database issues of the old
in-process ephemeral approach.

---

## Quick Overview

| Flow | CLI trigger | Description |
|------|-------------|-------------|
| `markdownize_flow` | `cli tools markdownize` | Convert PDF / DOCX / PPTX → Markdown |
| `ppt2pdf_flow` | `cli tools ppt2pdf` | Convert PPT / PPTX / ODP → PDF via LibreOffice |
| `rag_file_ingestion_flow` | `cli rag add-files` | Chunk, embed, and upsert documents into vector store |
| `baml_extraction_flow` | `cli baml extract` | Run BAML structured extraction on Markdown files |

---

## Managing the Prefect Server

GenAI Toolkit manages the Prefect server as an explicit background daemon.  Use the
`cli prefect` command group to control it:

```bash
# Start as a background daemon (auto-starts when workflows run if auto_start: true)
cli prefect start

# Start in foreground — useful for debugging; Ctrl-C to stop
cli prefect start --foreground

# Check running state + URLs
cli prefect status

# Stop the background daemon
cli prefect stop

# Open the Prefect UI in a browser
cli prefect ui
```

When `prefect.auto_start: true` (the default), the server is started automatically the first
time a `cli workflow run` or `cli workflow serve` is executed — you do not need to start it
manually.

### Configuration

```yaml
# config/app_conf.yaml (or any auto-scanned YAML)
prefect:
  host: "127.0.0.1"
  port: 4200
  auto_start: true   # start automatically before workflow runs
```

The PID file is stored at `<paths.data_root>/.prefect/prefect.pid` (or
`~/.cache/genai_tk/.prefect/prefect.pid` if `paths.data_root` is not configured).  The
`PREFECT_API_URL` environment variable is set automatically when the server is started.

---

## Available Flows

### Document-to-Markdown (`markdownize_flow`)

Converts documents in a directory to Markdown using `markitdown` or Mistral OCR.

```bash
# Basic conversion (PDF, DOCX, PPTX, …)
uv run cli tools markdownize ./docs ./output

# Recursive, force re-conversion
uv run cli tools markdownize ./docs ./output --recursive --force

# Use Mistral OCR for better PDF quality (requires MISTRAL_API_KEY)
uv run cli tools markdownize ./pdfs ./output --mistral-ocr --batch-size 5

# Filter by file pattern
uv run cli tools markdownize ./docs ./output \
    --include '*.pdf' --include '*.docx' --exclude '*_draft*'
```

**Supports:** PDF, DOCX, PPTX, ODP, XLSX, HTML, images (PNG/JPEG/WEBP).

**Key options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--recursive` | false | Recurse into sub-directories |
| `--force` | false | Re-convert already-converted files |
| `--mistral-ocr` | false | Use Mistral batch OCR API instead of markitdown |
| `--batch-size N` | 10 | Files per Mistral OCR batch job |
| `--include PATTERN` | all | Glob patterns to include (repeatable) |
| `--exclude PATTERN` | none | Glob patterns to exclude (repeatable) |

A **manifest file** (`manifest.json`) in the output directory tracks which source files have
been converted and their content hashes, enabling **incremental processing** — only new or
changed files are re-processed on subsequent runs.

---

### PPT → PDF (`ppt2pdf_flow`)

Converts PowerPoint/Impress files to PDF via LibreOffice headless.

```bash
uv run cli tools ppt2pdf ./slides ./pdfs
uv run cli tools ppt2pdf ./slides ./pdfs --recursive --force
```

Requires LibreOffice: `sudo apt-get install libreoffice` or equivalent.

Uses a `ThreadPoolTaskRunner` to convert multiple files in parallel.

---

### RAG Ingestion (`rag_file_ingestion_flow`)

Processes documents into the configured vector store: load → chunk → embed → upsert.

```bash
uv run cli rag add-files ./documents
uv run cli rag add-files ./documents --retriever persistent --force
```

**Deduplication:** each file is hashed before ingestion; unchanged files are skipped on
re-runs unless `--force` is provided.

**Chunking:** markdown-aware splitting that respects heading hierarchy (configurable chunk
size and overlap).

See [core.md](core.md) for vector store configuration.

---

### BAML Structured Extraction (`baml_extraction_flow`)

Runs a BAML function over a directory of Markdown files and writes JSON results.

```bash
uv run cli baml extract ./docs ./output \
    --recursive --function ExtractRainbow

uv run cli baml extract ./reports ./output \
    --function ExtractSummary --batch-size 10 --force
```

See [baml.md](baml.md) for BAML setup and configuration.

**Manifest tracking:** a per-function manifest records processed files (source hash,
output path, schema fingerprint).  The schema fingerprint changes when the BAML function
definition changes, triggering automatic re-extraction.

---

## Programmatic Usage

Flows can be called directly from Python.  The server must be running (or `auto_start: true`);
call `ensure_running()` to start it if needed:

```python
from genai_tk.utils.prefect_server import prefect_server
from genai_tk.workflow.prefect.flows.markdownize_flow import markdownize_flow
from genai_tk.workflow.prefect.flows.rag_flow import rag_file_ingestion_flow

# Ensure server is up (no-op if already running)
server = prefect_server()
server.ensure_running()
server.configure_api_url()  # sets PREFECT_API_URL in the current process

# Convert documents
markdownize_flow(
    source_dir="./docs",
    output_dir="./output",
    recursive=True,
)

# Ingest into vector store
rag_file_ingestion_flow(
    source_dir="./output",
    force=False,
)
```

To register a flow as a long-running Prefect deployment (receives runs from the server queue):

```python
from genai_tk.workflow.prefect.flow_factory import PrefectFlowFactory
from genai_tk.workflow.resolver import resolve_workflow_invocation

compiled = resolve_workflow_invocation("markdownize/docs")
PrefectFlowFactory(compiled=compiled).serve(name="markdownize-docs")
```

---

## Writing a New Flow

A minimal Prefect flow integrated with the toolkit:

```python
# myapp/my_flow.py
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
from genai_tk.workflow.prefect.run import run_flow_ephemeral


@task
def process_item(item: str) -> str:
    return item.upper()


@flow(task_runner=ConcurrentTaskRunner())
def my_flow(items: list[str]) -> list[str]:
    futures = [process_item.submit(item) for item in items]
    return [f.result() for f in futures]


# Invoke via the runtime helper (handles ephemeral vs server mode automatically)
if __name__ == "__main__":
    results = run_flow_ephemeral(my_flow, items=["a", "b", "c"])
    print(results)
```

### Manifest-Based Incremental Processing

For flows that process files, implement a manifest to skip unchanged inputs:

```python
from pathlib import Path
from pydantic import BaseModel, Field
from genai_tk.utils.hashing import buffer_digest


class ManifestEntry(BaseModel):
    source_hash: str
    output_path: str


class Manifest(BaseModel):
    entries: dict[str, ManifestEntry] = Field(default_factory=dict)


def load_manifest(path: Path) -> Manifest:
    if path.exists():
        return Manifest.model_validate_json(path.read_text())
    return Manifest()


def save_manifest(manifest: Manifest, path: Path) -> None:
    path.write_text(manifest.model_dump_json(indent=2))


@flow
def my_file_flow(input_dir: str, output_dir: str, force: bool = False) -> None:
    out = Path(output_dir)
    manifest_path = out / "manifest.json"
    manifest = load_manifest(manifest_path)

    for src in Path(input_dir).glob("**/*.md"):
        content = src.read_bytes()
        file_hash = buffer_digest(content)
        key = str(src)

        entry = manifest.entries.get(key)
        if not force and entry and entry.source_hash == file_hash:
            continue  # skip unchanged

        # … process src …
        dest = out / src.name.replace(".md", ".json")
        manifest.entries[key] = ManifestEntry(
            source_hash=file_hash, output_path=str(dest)
        )

    save_manifest(manifest, manifest_path)
```

---

## Deployment (Prefect Server)

For production workloads — scheduling, retries, alerting — deploy a Prefect server and
configure the toolkit to connect to it.

```bash
# Start server (create ~/.prefect/ profile first if needed)
prefect server start

# Point the toolkit at the server
export GENAI_PREFECT_API_URL=http://127.0.0.1:4200/api

# All CLI flows now register runs in the dashboard
uv run cli tools markdownize ./docs ./output
```

Prefect also provides a `prefect.yaml` for deploying flows as scheduled deployments — see
the [Prefect deployment docs](https://docs.prefect.io/latest/deploy/) for details.

---

## Workflow Engine (Composable Pipelines)

For **complex multi-step pipelines** that chain together multiple flows, the toolkit provides
a higher-level **Workflow Engine** that is YAML-driven and composable.

### Overview

The Workflow Engine lets you:

- Define **workflows** as DAGs of steps using YAML
- Bundle concrete parameter sets in named **presets** inside the workflow definition
- Chain **multiple Prefect flows** together with dependency tracking
- **Compose** workflows as sub-steps of other workflows
- Invoke workflows via `cli workflow run` with `--dry-run` support

**Example:** Chain PDF-to-Markdown conversion with RAG ingestion:

```yaml
workflows:
  full_ingest_pipeline:
    description: "PPT → PDF → Markdown → RAG ingestion"
    pipeline:
      - id: ppt_to_pdf
        run: genai_tk.workflow.prefect.flows.ppt2pdf_flow.ppt2pdf_flow
        with:
          base_dir: "${paths.data_root}/ppts"
          output_dir: "${paths.data_root}/pdfs"

      - id: to_markdown
        run: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
        after: [ppt_to_pdf]
        with:
          base_dir: "${paths.data_root}/pdfs"
          output_dir: "${paths.data_root}/md"
          pathspecs: ["**/*.pdf"]

      - id: ingest
        run: genai_tk.workflow.prefect.flows.rag_flow.rag_file_ingestion_flow
        after: [to_markdown]
        with:
          base_dir: "${paths.data_root}/md"
    presets:
      marketing:
        base_dir: "${paths.data_root}/marketing"
```

Invoke with:

```bash
# See the full 3-step plan
uv run cli workflow run full_ingest_pipeline/marketing --dry-run

# Execute all steps in order
uv run cli workflow run full_ingest_pipeline/marketing
```

### How It Works

1. **Workflow Definition** — Steps specify a callable target (via dotted Python path or
   workflow name), arguments (`with:`), and dependencies (`after:`).

2. **Resolution** — Preset values and CLI `--set` overrides are merged with workflow
   `defaults` to produce concrete parameter values.  `${values.*}` references in `with:`
   are resolved against this merged dict.

3. **Pre-flight Validation** — Before Prefect starts, the engine validates each step's
   keyword arguments against its function signature, giving a clean error for typos or
   renamed parameters instead of a verbose Prefect traceback.

4. **Topological Sort** — Steps are ordered based on `after:` dependencies; circular
   dependencies are detected and reported.

5. **Execution** — Each step invokes its target via the Prefect task runner.  If a step
   fails, the `on_failure:` policy determines whether to abort, skip, or continue.

### See Also

For the complete Workflow Engine guide, including:
- Full DSL syntax (`run:`, `pipeline:`, `presets:`, `params:`, `cache:`)
- Sub-workflow composition and the `@workflow` decorator
- CLI reference (`list`, `show`, `run`, `validate`)
- Error handling modes (`abort`, `skip`, `continue`)
- Step implementation guide

See [workflows.md](workflows.md).

