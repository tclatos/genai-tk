---
name: kg-document-graph
description: Build and navigate the genai-graph Document Graph — the Folder/Document/MarkdownSection provenance hierarchy, DocumentGraphFactory, tree_parser, ingest_document_graph/drop_document_graph, and the vectorless agentic RAG navigation tools. Use when editing genai_graph/kg/document_graph, writing a document-ingestion pipeline, or giving an agent tools to walk a document corpus by heading.
---

# GenAI Graph Document Graph

## Read First

- `docs/document-graph.md` — full schema, factories, CLI, querying
- `genai_graph/kg/document_graph/ingest.py` — `ingest_document_graph`, `drop_document_graph`
- `genai_graph/kg/document_graph/tree_parser.py` — heading-hierarchy parser
- `genai_graph/kg/document_graph/repository.py` — section CRUD
- `genai_graph/kg/factories/document_graph_factory.py` — `DocumentGraphFactory`, `DocumentGraphBundle`
- `genai_graph/kg/factories/document_factory.py` — `DocumentDirectoryFactory`
- `genai_graph/kg/query/document_graph_tools.py` — navigation functions + LangChain tools
- `genai_graph/kg/nodes/document.py`, `genai_graph/kg/nodes/document_section.py` — node models

## What it is

A generic, **vectorless** representation of a document corpus: which folder files came
from, file metadata, and — for Markdown — the heading hierarchy. It is the provenance layer
every entity-extraction factory attaches to, and a substrate for agentic RAG where an agent
walks a document's table of contents and reads section text directly (no embeddings).

```
Folder ──CONTAINS──▶ Document ──HAS_SECTION──▶ MarkdownSection ──HAS_SUBSECTION──▶ MarkdownSection ──…
```

| Node | Key | Notes |
|---|---|---|
| `Folder` | `folder_id` | A directory, `.zip` archive, or a single file's parent. |
| `Document` | `content_hash` (xxHash of raw bytes) | Provenance anchor; carries `filename`, `relative_path`, `path`, `mime_type`, `markdown_hash`, `token_count`, `section_count`. |
| `MarkdownSection` | `section_id` = `{markdown_hash}::{sequence}` | One heading-delimited section (heading + body up to the next heading). A synthetic level-0 root section captures heading-less documents/preambles. |

Sections form a flat table with `parent_section_id`; the hierarchy is materialized as
`HAS_SUBSECTION` edges. A section's `text` is its own content only (non-overlapping), so
concatenating every section in `sequence` order reconstructs the original Markdown exactly.

**Identity/dedup:** `Document` and `MarkdownSection` are keyed by content hash, so
re-ingesting unchanged files is a no-op MERGE. When an entity factory
(`MarkdownBamlFactory`/`JsonFileBackedFactory` via `DocumentMixin`) also produces a
`Document` node for the same file, it MERGEs into the *same* node — provenance and extracted
entities share one graph node.

There are **no `Chunk` nodes** — no chunking or embeddings. Chunking/embedding RAG is a
separate path (`DocumentDirectoryFactory` as a base for custom pipelines).

## Factories

| Factory | Produces | Use when |
|---|---|---|
| `DocumentGraphFactory` | `Folder → Document → MarkdownSection` | You want the navigable heading hierarchy over Markdown. |
| `DocumentDirectoryFactory` | Plain `Document` nodes | File-level provenance only (no sections), or a base for custom RAG. |
| `MarkdownBamlFactory` | Your entity nodes + a provenance `Document` (`MENTIONS`) | Inline BAML entity extraction from Markdown. |
| `JsonFileBackedFactory` | Your entity nodes + a provenance `Document` | Entities already extracted as JSON files. |

`DocumentGraphFactory(sources=[...], include=["*.md"], exclude=[])` — `sources` accepts a
mix of directories, files, and `.zip` archives; each becomes a `Folder`.

## Ingest

```python
from genai_graph.kg.backend import KuzuBackend
from genai_graph.kg.document_graph.ingest import ingest_document_graph, drop_document_graph
from genai_graph.kg.factories.document_graph_factory import DocumentGraphFactory

backend = KuzuBackend()
backend.connect("./data/kg/tree.db")
factory = DocumentGraphFactory(sources=["./docs"], include=["*.md"])
result = ingest_document_graph(backend, factory)  # MERGEs everything
result = ingest_document_graph(backend, factory, force=True)  # rebuild sections for existing docs
drop_document_graph(backend)  # drop Section/Document/Folder tables
```

`ingest_document_graph` builds the schema, parses each file's heading hierarchy
(`tree_parser`), and MERGEs in one call. `force=True` rebuilds sections for documents
already in the graph (e.g. after a heading edit).

## End-to-end via the workflow engine

`docgraph_build_step` (in `genai_graph/orchestration/workflow_steps.py`) ties
markdownization, entity extraction, and the document graph into **one** database:

1. Optionally markdownize `sources` (PPT/PDF/… or pre-existing Markdown) via
   `genai_tk.workflow.markdownize.markdownize_flow`.
2. Run each configured entity `factories` (e.g. a `MarkdownBamlFactory` subclass) into one
   KG named `kg_name`.
3. Optionally ingest the `Folder → Document → MarkdownSection` graph over the same Markdown
   into the *same* database (`build_document_graph`, default `true`) — `Document` nodes
   MERGE by content hash with the ones the entity factories create.

This backs both `cli docgraph run` (ad-hoc sources) and a project's named profiles via
`cli kg create <name>` — see `kg-workflows`.

## Navigation tools (vectorless agentic RAG)

`genai_graph.kg.query.document_graph_tools` exposes read-only Cypher-backed helpers used by
the CLI, an agent's tools, and the Textual TUI:

```python
from genai_graph.kg.query.document_graph_tools import (
    list_documents,
    get_document_toc,
    get_section_content,
    reconstruct_document,
    reconstruct_section,
    search_sections,
    create_document_graph_tools,
)
```

Documents are addressed by content hash (full or prefix), `markdown_hash`, filename, or
source path. `create_document_graph_tools(db_path)` wraps these as LangChain `BaseTool`s:
`list_documents`, `get_document_toc`, `get_section_content`, `search_sections`. Wire them
into an agent profile (see `kg-query` and `genai-tk/add-tool`).

The agentic RAG loop: `list_documents` → `get_document_toc` (the map) →
`get_section_content` (only the sections worth reading) → answer. No embeddings, no chunk
retrieval — the agent reads exactly the section text it navigated to.

## CLI

```bash
cli docgraph build ./docs --db ./data/kg/tree.db           # markdownize + build document graph
cli docgraph build ./RFQ.zip --db ./data/kg/tree.db --profile fast
cli docgraph run --workflow rainbow_extract -s ./some_file.pptx   # project entity-extraction workflow
cli docgraph list --db ./data/kg/tree.db
cli docgraph toc <filename-or-hash> --db ./data/kg/tree.db
cli docgraph cat <filename-or-hash> --db ./data/kg/tree.db
cli docgraph search "keyword" --db ./data/kg/tree.db
cli docgraph tui --db ./data/kg/tree.db
```

`cli docgraph build` always markdownizes first then ingests; `cli docgraph run` runs a
project's `docgraph_build`-based workflow against ad-hoc or configured sources. See `kg-cli`.

## Change Workflow

1. Schema changes to `Folder`/`Document`/`MarkdownSection` live in
   `kg/nodes/document.py` and `kg/nodes/document_section.py`; keep the content-hash keys —
   they are what make re-ingest a no-op and what let entity factories MERGE into the same
   `Document`.
2. Heading-parsing changes go in `tree_parser.py`; verify the "concatenate sections in
   `sequence` order == original Markdown" invariant with a round-trip test.
3. New navigation helpers go in `document_graph_tools.py`; expose them both as raw
   functions and inside `create_document_graph_tools` so agents and the CLI share them.
4. Re-ingest with `force=True` (or `--force graph`) after a parser/section change.

## Commands

```bash
cli docgraph build ./docs --db ./data/kg/tree.db --force md   # re-run markdown conversion
cli docgraph build ./docs --db ./data/kg/tree.db --force graph # re-ingest, reuse markdown cache
uv run just test
```

## Avoid

- Do not add `Chunk` nodes or embeddings to the Document Graph — it is deliberately
  vectorless; chunking RAG is a separate path.
- Do not key `Document`/`MarkdownSection` by anything but content hash — that breaks the
  MERGE-with-entity-factories guarantee.
- Do not hand-roll navigation Cypher in agent code — reuse `document_graph_tools`.
- Do not markdownize manually before `cli docgraph build` — it markdownizes internally.

## Complements

- `kg-factories` — `DocumentGraphFactory`/`MarkdownBamlFactory`/`DocumentDirectoryFactory`.
- `kg-query` — `create_document_graph_tools` and the broader Cypher/Text-to-Cypher story.
- `kg-workflows` — `docgraph_build_step` and `cli docgraph run`.
- `genai-tk/workflow-engine` — `markdownize_flow` used in step 1.
