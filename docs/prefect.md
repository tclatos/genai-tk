# Prefect Workflow Orchestration

GenAI Toolkit integrates [Prefect](https://docs.prefect.io/) as its execution engine for
long-running, multi-step tasks — document conversion, RAG ingestion, structured extraction.
Every Prefect flow benefits from observability, retries, and optional parallelism.

**Two complementary ways to run Prefect flows:**

| Approach | When to use |
|----------|-------------|
| **`@flow`-decorated function** (standard Prefect) | Existing flows, custom logic, one-off scripts |
| **YAML Workflow definition** | Multi-step pipelines, reusable parameterised workflows, scheduled runs |

Both approaches share the same **local Prefect server** managed by `cli prefect`.

> **See also:** [workflows.md](workflows.md) — full DSL reference and how to build YAML pipelines.

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

## Running Any `@flow` Function

The toolkit's Prefect server can execute **any** Prefect `@flow` — your own flows, third-party
flows, or the built-in ones. Use the server singleton to start the server and set `PREFECT_API_URL`:

```python
from genai_tk.utils.prefect_server import prefect_server
from prefect import flow, task

# Ensure server is up (no-op if already running, auto-starts if auto_start: true)
server = prefect_server()
server.ensure_running()
server.configure_api_url()   # sets PREFECT_API_URL in the current process

@task
def process_item(item: str) -> str:
    return item.upper()

@flow(name="my-custom-flow")
def my_flow(items: list[str]) -> list[str]:
    futures = [process_item.submit(item) for item in items]
    return [f.result() for f in futures]

# Execute — run is visible in the Prefect UI (http://localhost:4200)
results = my_flow(items=["hello", "world"])
```

### Calling Built-in Flows Directly

```python
from genai_tk.utils.prefect_server import prefect_server
from genai_tk.workflow.prefect.flows.markdownize_flow import markdownize_flow
from genai_tk.workflow.prefect.flows.rag_flow import rag_file_ingestion_flow

server = prefect_server()
server.ensure_running()
server.configure_api_url()

markdownize_flow(source_dir="./docs", output_dir="./output", recursive=True)
rag_file_ingestion_flow(source_dir="./output", force=False)
```

### Serving a `@flow` as a Deployment

Register any `@flow` as a long-running deployment that polls for scheduled or manually
triggered runs from the UI or API:

```python
server = prefect_server()
server.ensure_running()
server.configure_api_url()

# Blocks the process; Ctrl-C to stop
my_flow.serve(name="my-flow-daily", cron="0 9 * * 1-5")  # Weekdays at 9 AM
```

---

## Running YAML-Defined Workflows

### Via CLI (recommended)

```bash
# List all configured workflows
uv run cli workflow list

# Dry-run: see the execution plan without running
uv run cli workflow run markdownize/docs --dry-run

# Execute a workflow with a named preset
uv run cli workflow run markdownize/docs

# Pass ad-hoc overrides
uv run cli workflow run markdownize --base-dir /data/pdfs --to /data/md

# Force recomputation (bypass caches)
uv run cli workflow run markdownize/docs --force
```

### Via `PrefectFlowFactory` (programmatic)

```python
from genai_tk.workflow import PrefectFlowFactory

# Resolve + compile from config by workflow name / preset
factory = PrefectFlowFactory.from_profile("markdownize/docs", values={"batch_size": 10})
results = factory.run()                              # execute immediately
factory.serve(name="nightly-markdownize", cron="0 2 * * *")  # or serve
```

### Via `flow_from_yaml` (inline YAML — great for notebooks and scripts)

`flow_from_yaml` parses a workflow **inline** and returns a standard Prefect `@flow` — no
config directory needed.  Accepts a YAML **string**, a **`Path`**, or a **`dict`**:

```python
from genai_tk.workflow import flow_from_yaml

YAML = """
workflows:
  convert_docs:
    description: "PDF → Markdown"
    run: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
    defaults:
      base_dir: /data/pdfs
      output_dir: /data/md
      batch_size: 5
"""

flow = flow_from_yaml(YAML)
flow()   # execute — visible in the Prefect UI
```

Multi-step, from a file:

```python
from pathlib import Path
from genai_tk.workflow import flow_from_yaml

flow = flow_from_yaml(
    Path("config/workflows/my_pipeline.yaml"),
    workflow_name="full_pipeline",
    values={"batch_size": 10},
)
flow()
```

---

## Writing a New `@flow`

A minimal flow integrated with the toolkit:

```python
# myapp/my_flow.py
from pathlib import Path
from prefect import flow, task
from genai_tk.utils.prefect_server import prefect_server


@task
def convert_file(src: Path, dest: Path) -> str:
    dest.write_text(src.read_text().upper())
    return str(dest)


@flow(name="uppercase-files")
def uppercase_flow(input_dir: str, output_dir: str) -> list[str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    futures = [
        convert_file.submit(src, out / src.name)
        for src in Path(input_dir).glob("*.txt")
    ]
    return [f.result() for f in futures]


if __name__ == "__main__":
    server = prefect_server()
    server.ensure_running()
    server.configure_api_url()
    uppercase_flow(input_dir="./in", output_dir="./out")
```

### Manifest-Based Incremental Processing

For flows that process files, use a manifest to skip unchanged inputs:

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
    return Manifest.model_validate_json(path.read_text()) if path.exists() else Manifest()


def save_manifest(manifest: Manifest, path: Path) -> None:
    path.write_text(manifest.model_dump_json(indent=2))


@flow
def my_file_flow(input_dir: str, output_dir: str, force: bool = False) -> None:
    out = Path(output_dir)
    manifest_path = out / "manifest.json"
    manifest = load_manifest(manifest_path)

    for src in Path(input_dir).glob("**/*.md"):
        file_hash = buffer_digest(src.read_bytes())
        key = str(src)
        entry = manifest.entries.get(key)
        if not force and entry and entry.source_hash == file_hash:
            continue  # skip unchanged

        dest = out / src.name.replace(".md", ".json")
        # … process src → dest …
        manifest.entries[key] = ManifestEntry(source_hash=file_hash, output_path=str(dest))

    save_manifest(manifest, manifest_path)
```

---

## Exposing a `@flow` as a YAML Workflow Step

Once you have a `@flow` function, expose it in YAML so it can be composed and run via CLI:

```yaml
# config/workflows/my_workflows.yaml
workflows:
  uppercase_files:
    description: "Convert text files to uppercase"
    run: myapp.my_flow.uppercase_flow
    defaults:
      input_dir: "${paths.data_root}/in"
      output_dir: "${paths.data_root}/out"
    params:
      input_dir: {required: true}
      output_dir: {required: true}
```

```bash
uv run cli workflow run uppercase_files --base-dir ./texts --to ./output
```

See [workflows.md](workflows.md) for the full YAML DSL reference.

---

## Key Classes and Functions

| Symbol | Module | Purpose |
|--------|--------|---------|
| `prefect_server()` | `genai_tk.utils.prefect_server` | Singleton — start / stop / configure the local server |
| `PrefectFlowFactory` | `genai_tk.workflow` | Build and run a Prefect flow from a compiled workflow |
| `PrefectFlowFactory.from_profile()` | `genai_tk.workflow` | Create factory from a workflow name/preset string |
| `flow_from_yaml()` | `genai_tk.workflow` | Parse YAML inline and return a `@flow` object |
| `WorkflowCompiler` | `genai_tk.workflow` | Compile a `WorkflowDefV2` into a `CompiledWorkflow` |
| `execute_workflow()` | `genai_tk.workflow` | Execute a `ResolvedWorkflowInvocation` |


