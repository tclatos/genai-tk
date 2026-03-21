# Prefect Workflow Orchestration

GenAI Toolkit uses [Prefect](https://docs.prefect.io/) for orchestrating multi-step, long-running
tasks — document conversion, RAG ingestion, structured extraction — so that each step is
observable, retryable, and optionally parallelised.

All flows run in **ephemeral mode by default** (no external server required).  Connecting to a
deployed Prefect server unlocks the dashboard, persistent run history, and scheduling.

---

## Quick Overview

| Flow | CLI trigger | Description |
|------|-------------|-------------|
| `markdownize_flow` | `cli tools markdownize` | Convert PDF / DOCX / PPTX → Markdown |
| `ppt2pdf_flow` | `cli tools ppt2pdf` | Convert PPT / PPTX / ODP → PDF via LibreOffice |
| `rag_prefect_flow` | `cli rag ingest` | Chunk, embed, and upsert documents into vector store |
| `baml_extraction_flow` | `cli baml extract` | Run BAML structured extraction on Markdown files |

---

## Runtime Modes

### Ephemeral (default)

The flow runs **inside the calling process** with no external daemon.  This is the default when
no Prefect API URL is configured.  Suitable for local development and CI.

```bash
# All CLI flow commands work out-of-the-box — no Prefect server needed
uv run cli tools markdownize ./docs ./output --recursive
```

Prefect logging is suppressed (`WARNING` level) to keep terminal output clean.

### Deployed Prefect Server

Point flows at a running Prefect server to get the dashboard, scheduling, and run history.

**Configuration (preferred):**
```yaml
# config/init/overrides.yaml
prefect:
  api_url: http://127.0.0.1:4200/api
```

**Environment variable (alternative):**
```bash
export GENAI_PREFECT_API_URL=http://127.0.0.1:4200/api
```

Start a local server:
```bash
prefect server start
# Server is available at http://127.0.0.1:4200
```

When `api_url` is set, flows connect to the server automatically — no code change needed.

### Mode Selection Logic

```
GENAI_PREFECT_API_URL env var set?  → use that URL
  else prefect.api_url in config?   → use that URL
    else                            → ephemeral in-process mode
```

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

### RAG Ingestion (`rag_prefect_flow`)

Processes documents into the configured vector store: load → chunk → embed → upsert.

```bash
uv run cli rag ingest ./documents
uv run cli rag ingest ./documents --recursive --force
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

All flows can be called directly from Python using the `run_flow_ephemeral` helper, which
wraps the call inside the correct Prefect runtime context:

```python
from genai_tk.extra.prefect.runtime import run_flow_ephemeral
from genai_tk.extra.markdownize_prefect_flow import markdownize_flow
from genai_tk.extra.rag.rag_prefect_flow import rag_ingest_flow

# Convert documents
run_flow_ephemeral(
    markdownize_flow,
    source_dir="./docs",
    output_dir="./output",
    recursive=True,
)

# Ingest into vector store
run_flow_ephemeral(
    rag_ingest_flow,
    source_dir="./output",
    force=False,
)
```

Or use the context manager directly for finer control:

```python
from genai_tk.extra.prefect.runtime import ephemeral_prefect_settings
from genai_tk.extra.markdownize_prefect_flow import markdownize_flow

with ephemeral_prefect_settings():
    result = markdownize_flow(source_dir="./docs", output_dir="./out")
```

---

## Writing a New Flow

A minimal Prefect flow integrated with the toolkit:

```python
# myapp/my_flow.py
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
from genai_tk.extra.prefect.runtime import run_flow_ephemeral


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
