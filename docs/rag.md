# RAG Systems (`genai_tk.extra.rag` · `genai_tk.core.retriever_factory`)

> **Quick nav:** [Design](#design) · [Retriever Types](#retriever-types) · [Configuration](#yaml-configuration) · [Python API](#python-api) · [CLI](#cli-commands) · [Batch Ingestion](#batch-ingestion-prefect-flow) · [Agent Tools](#using-retrievers-as-agent-tools) · [PostgreSQL](#postgresql-hybrid-search)

---

## Design

The RAG layer is built around two central abstractions:

| Class | Module | Role |
|-------|--------|------|
| `ManagedRetriever` | `genai_tk.core.retriever_factory` | Wraps any LangChain `BaseRetriever`. Adds async query, document ingestion, and deletion. |
| `RetrieverFactory` | `genai_tk.core.retriever_factory` | Reads a `retrievers.<tag>` block from YAML and returns a `ManagedRetriever`. |

### Layered architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Your code / agent                    │
└────────────────────────┬─────────────────────────────────┘
                         │  RetrieverFactory.create("my_tag")
┌────────────────────────▼─────────────────────────────────┐
│              ManagedRetriever                            │
│  aquery()  aadd_documents()  adelete_store()  get_stats()│
└──────┬─────────────────┬────────────────────┬────────────┘
       │                 │                    │
  VectorStore      BM25DocumentStore    EnsembleRetriever
  (Chroma / In-    (bm25s, disk cache)  (weighted fusion)
   Memory / PG)
       │
  RecordManager (optional dedup)
```

**Key design decisions:**

- **Async-first** — `aquery` / `aadd_documents` are `async`. Sync wrappers (`query`, `add_documents`) use `asyncio.run()` for CLI and notebook convenience.
- **YAML-driven** — every retriever is a named config block; no Python code changes needed to add or switch retrievers.
- **Composable** — ensemble and reranked types reference other named retrievers, allowing arbitrarily deep compositions.
- **Document stores are separate from retrievers** — `ManagedRetriever.store` holds the write path; `ManagedRetriever.retriever` holds the read path. Read-only retrievers (ZeroEntropy, reranked) simply have `store = None`.
- **EmbeddingsStore is a pure factory** — it creates `VectorStore` instances only; all query/ingest logic lives in `ManagedRetriever`.

---

## Retriever Types

### `vector` — dense similarity search

Backed by an `EmbeddingsStore` (Chroma, InMemory, or PgVector).

```yaml
retrievers:
  my_store:
    type: vector
    embeddings_store: chroma_indexed   # key in embeddings_store: section
    top_k: 4
    search_type: similarity            # or mmr
    record_manager_url: ~              # auto SQLite when null + persistent store
```

- Uses `asimilarity_search(query, k=k, filter=filter)` directly on the vector store.
- If the backing store is persistent and `record_manager_url` is not set, a SQLite record manager is **auto-created** at `data/record_manager/<config_tag>.db` to deduplicate ingested documents.

### `bm25` — keyword / sparse search

Uses the `bm25s` library. The index is built from documents and persisted to disk.

```yaml
retrievers:
  bm25_local:
    type: bm25
    k: 4
    preprocessing: default    # or "spacy" for lemmatisation
    spacy_model: en_core_web_sm
    cache_dir: ~              # auto: data/bm25_cache/<config_tag>/
```

Index files:

| File | Content |
|------|---------|
| `data/bm25_cache/<tag>/bm25_index/` | bm25s vectorizer (pickle) |
| `data/bm25_cache/<tag>/documents.json` | original documents with full metadata |

The index is rebuilt from scratch on each `aadd_documents()` call. `get_or_load_retriever()` lazy-loads from disk on first query after a restart.

### `ensemble` — weighted fusion

Combines any number of other named retrievers using `EnsembleRetriever` (Reciprocal Rank Fusion).

```yaml
retrievers:
  hybrid:
    type: ensemble
    retrievers:
      - ref: my_store      # must be a key in retrievers:
        weight: 0.7
      - ref: bm25_local
        weight: 0.3
```

Weights are normalised to sum to 1.0 before being passed to `EnsembleRetriever`. Each `ref` is resolved by calling `RetrieverFactory.create()` recursively.

### `reranked` — contextual compression / reranking

Wraps another retriever with a reranking step that re-scores and filters results.

```yaml
retrievers:
  best_results:
    type: reranked
    retriever: hybrid          # any key in retrievers:
    reranker: embeddings       # "embeddings" | "cohere" | "cross_encoder"
    top_k: 3
    fetch_k: 10                # how many docs the base retriever fetches
    reranker_model: ~          # optional: model name for cohere/cross_encoder
```

| `reranker` | Backend | Extra dependency |
|-----------|---------|-----------------|
| `embeddings` | `EmbeddingsFilter` (semantic similarity ≥ 0.7) | none |
| `cohere` | `CohereRerank` | `uv add langchain-cohere` |
| `cross_encoder` | `HuggingFaceCrossEncoder` | `uv add sentence-transformers` |

### `pg_hybrid` — PostgreSQL vector + full-text search

Combines pgvector similarity with PostgreSQL full-text search (tsvector) in a single query.

```yaml
retrievers:
  pg_hybrid:
    type: pg_hybrid
    embeddings: default        # key in embeddings: section
    postgres: default          # key in postgres: section
    table_name_prefix: embeddings
    hybrid_search: true
    top_k: 4
    hybrid_search_config:
      tsv_lang: pg_catalog.english
      fusion_function_parameters:
        primary_results_weight: 0.7   # vector weight
        secondary_results_weight: 0.3  # full-text weight
```

Requires the `postgres` dependency group:
```bash
uv add langchain-postgres psycopg2-binary asyncpg
```

For embedded PostgreSQL (no server needed):
```bash
uv add pgembed pgembed-pgvector
```

### `zero_entropy` — ZeroEntropy external retriever

Read-only retriever backed by the ZeroEntropy document search SDK.

```yaml
retrievers:
  ze_docs:
    type: zero_entropy
    collection_name: my_collection
    k: 5
    retrieval_type: documents
```

---

## YAML Configuration

All retriever config lives under the `retrievers:` top-level key. The `postgres:` key configures PG connection sources.

### Retriever `type` field — short aliases vs qualified names

The `type` field in each retriever config accepts either:

1. **Short aliases** (backward compatible):
   ```yaml
   type: vector
   type: bm25
   type: ensemble
   type: reranked
   type: pg_hybrid
   type: zero_entropy
   ```

2. **Fully-qualified class names** (stable, recommended for new code):
   ```yaml
   type: genai_tk.core.retrievers.VectorRetriever
   type: genai_tk.core.retrievers.BM25Retriever
   type: genai_tk.core.retrievers.EnsembleRetriever
   type: genai_tk.core.retrievers.RerankedRetriever
   type: genai_tk.core.retrievers.PgHybridRetriever
   type: genai_tk.core.retrievers.ZeroEntropyRetriever
   ```

Qualified names allow you to write **custom retriever builders** in your own codebase without modifying genai-tk. See [Extending with custom retrievers](#extending-with-custom-retrievers) below.

### Backend `backend` field — short aliases vs qualified names

Vector store backends in `embeddings_store:` blocks also accept short aliases or qualified names:

```yaml
embeddings_store:
  my_store:
    backend: Chroma                                    # short alias
    # OR
    backend: genai_tk.core.vector_backends.ChromaBackend   # qualified name
    config:
      storage: '::memory::'
```

Supported backends:

| Short alias | Qualified name | Backend |
|-----------|---|---------|
| `Chroma` | `genai_tk.core.vector_backends.ChromaBackend` | Chroma (in-memory or persistent) |
| `InMemory` | `genai_tk.core.vector_backends.InMemoryBackend` | Ephemeral in-process store |
| `PgVector` | `genai_tk.core.vector_backends.PgVectorBackend` | PostgreSQL with pgvector + full-text |

### Full reference

```yaml
# config/baseline.yaml (or your app_conf.yaml)

retrievers:

  # ── vector ──────────────────────────────────────────────
  default:
    type: vector
    embeddings_store: in_memory_chroma  # key in embeddings_store:
    top_k: 4
    search_type: similarity             # similarity | mmr
    record_manager_url: ~               # null → auto SQLite for persistent stores

  persistent:
    type: vector
    embeddings_store: chroma_indexed
    top_k: 4

  # ── bm25 ────────────────────────────────────────────────
  bm25_local:
    type: bm25
    k: 4
    preprocessing: default              # default | spacy
    spacy_model: en_core_web_sm
    cache_dir: ~                        # null → data/bm25_cache/<tag>/

  # ── ensemble ────────────────────────────────────────────
  hybrid_ensemble:
    type: ensemble
    retrievers:
      - ref: persistent
        weight: 0.7
      - ref: bm25_local
        weight: 0.3

  # ── reranked ────────────────────────────────────────────
  hybrid_reranked:
    type: reranked
    retriever: hybrid_ensemble
    reranker: embeddings                # embeddings | cohere | cross_encoder
    top_k: 3
    fetch_k: 10

  # ── pg_hybrid (requires postgres dependency) ────────────
  # pg_hybrid:
  #   type: pg_hybrid
  #   embeddings: default
  #   postgres: default
  #   hybrid_search: true
  #   top_k: 4
  #   hybrid_search_config:
  #     tsv_lang: pg_catalog.english


postgres:
  default:
    mode: external
    url: postgresql+asyncpg://${oc.env:POSTGRES_USER,postgres}:${oc.env:POSTGRES_PASSWORD,password}@localhost:5432/genai

  # embedded:
  #   mode: pgembed
  #   data_dir: ${paths.data_root}/pgembed
  #   extensions: [vector]
```

### `embeddings_store:` blocks (used by `type: vector`)

These are referenced from `retrievers.<tag>.embeddings_store`:

```yaml
embeddings_store:
  in_memory_chroma:
    backend: Chroma
    embeddings: default
    table_name_prefix: embeddings
    config:
      storage: '::memory::'         # in-memory (no persistence)

  chroma_indexed:
    backend: Chroma
    embeddings: default
    table_name_prefix: embeddings
    config:
      storage: ${paths.data_root}/vector_store   # persistent on disk

  local_fast:
    backend: Chroma
    embeddings: bge-small-en@local  # local FastEmbed model
    table_name_prefix: embeddings
    config:
      storage: ${paths.data_root}/vector_store_local_fast
```

---

## Extending with Custom Retrievers

You can define a custom retriever builder in your own codebase and reference it by its fully-qualified class name in YAML — no need to modify genai-tk.

### Step 1: Define a builder class

Create a module in your project with a builder class that has:
- `config_model` — a Pydantic v2 config class for parsing the sub-dict
- `build(cfg, config_tag, resolver)` — a classmethod that returns a `ManagedRetriever`

**Example: `myapp/retrievers/custom.py`**

```python
from pydantic import BaseModel
from collections.abc import Callable
from typing import Any

from genai_tk.core.retriever_factory import ManagedRetriever


class MyCustomConfig(BaseModel):
    """Config for MyCustomRetriever."""
    service_url: str
    api_key: str
    top_k: int = 4


class MyCustomRetriever:
    """Builder for custom retrievers backed by an external service."""

    config_model = MyCustomConfig

    @classmethod
    def build(
        cls,
        cfg: MyCustomConfig,
        config_tag: str,
        resolver: Callable[[str], Any],
    ) -> ManagedRetriever:
        """Build a ManagedRetriever backed by a custom service."""
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.documents import Document

        class CustomServiceRetriever(BaseRetriever):
            def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
                # Call your service
                import requests
                resp = requests.get(
                    f"{cfg.service_url}/search",
                    params={"q": query, "top_k": cfg.top_k},
                    headers={"Authorization": f"Bearer {cfg.api_key}"},
                )
                docs = [Document(page_content=r["content"], metadata=r.get("meta", {}))
                        for r in resp.json()]
                return docs

            async def _aget_relevant_documents(self, query: str, **kwargs) -> list[Document]:
                return self._get_relevant_documents(query)  # fallback to sync

        retriever = CustomServiceRetriever()
        return ManagedRetriever(
            retriever=retriever,
            store=None,  # read-only
            default_k=cfg.top_k,
            config_tag=config_tag,
        )
```

### Step 2: Reference in YAML

```yaml
retrievers:
  my_custom_service:
    type: myapp.retrievers.custom.MyCustomRetriever
    service_url: https://search.example.com/api
    api_key: ${oc.env:MY_SERVICE_API_KEY}
    top_k: 5
```

### Step 3: Use it like any other retriever

```python
from genai_tk.core.retriever_factory import RetrieverFactory

managed = RetrieverFactory.create("my_custom_service")
docs = await managed.aquery("my question")
```

The `resolver` parameter in `build()` is a reference to `RetrieverFactory.create`, which you can use to recursively build composed retrievers (e.g., if you want to wrap another retriever).

### Creating a retriever

```python
from genai_tk.core.retriever_factory import RetrieverFactory

managed = RetrieverFactory.create("hybrid_ensemble")
```

`create()` reads `retrievers.hybrid_ensemble` from the active config, builds all referenced sub-retrievers recursively, and returns a `ManagedRetriever`.

### Querying

```python
# Async (preferred — inside async code / agents)
docs = await managed.aquery("What is vector search?", k=5)

# With a metadata filter
docs = await managed.aquery("pricing", filter={"source": "contracts"})

# Sync (CLI, notebooks, tests)
docs = managed.query("What is vector search?")
```

### Ingesting documents

```python
from langchain_core.documents import Document

docs = [
    Document(page_content="Vector search uses embeddings.", metadata={"source": "intro.md"}),
    Document(page_content="BM25 is a keyword ranking function.", metadata={"source": "bm25.md"}),
]

# Async
await managed.aadd_documents(docs)

# Sync
managed.add_documents(docs)
```

Only retrievers with a document store (`managed.has_store == True`) support ingestion. Read-only types (ZeroEntropy, reranked with no writable base) raise `RuntimeError`.

### Deleting all stored documents

```python
# Async
success = await managed.adelete_store()

# Sync
managed.delete_store()
```

Chroma collections and BM25 cache directories are cleaned up. PgVector deletion is not yet implemented.

### Introspection

```python
print(managed.get_stats())
# {'config_tag': 'hybrid_ensemble', 'default_k': 4, 'vector_backend': 'Chroma', ...}

print(managed.has_store)    # True / False
print(managed.default_k)    # 4

configs = RetrieverFactory.list_available_configs()
# ['default', 'bm25_local', 'hybrid_ensemble', 'hybrid_reranked', ...]
```

### Building a RAG chain

```python
from genai_tk.core.retriever_factory import RetrieverFactory
from genai_tk.core.llm_factory import get_llm
from genai_tk.core.prompts import def_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

managed = RetrieverFactory.create("hybrid_ensemble")
llm = get_llm()

prompt = def_prompt(
    system="You are a helpful assistant. Answer using only the provided context.",
    user="Context:\n{context}\n\nQuestion: {question}",
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": managed.retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("What is hybrid search?")
```

### Document stores directly

The `BM25DocumentStore` and `VectorDocumentStore` classes can be used independently:

```python
from pathlib import Path
from genai_tk.core.retriever_factory import BM25DocumentStore

store = BM25DocumentStore(cache_dir=Path("data/bm25_cache/my_index"))
await store.aadd_documents(docs)
results = await store.aget_relevant_documents("my query", k=3)
```

---

## CLI Commands

All RAG operations are available under `cli rag`:

```bash
uv run cli rag --help
```

### `add-files` — ingest a directory

```bash
uv run cli rag add-files ./my_docs/ --retriever persistent
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--retriever` | `-r` | `default` | Retriever config tag |
| `--include` | `-i` | `**/*` | Glob patterns to include (repeatable) |
| `--exclude` | `-e` | — | Glob patterns to exclude (repeatable) |
| `--recursive/--no-recursive` | | `--recursive` | Search subdirectories |
| `--force` | `-f` | off | Reprocess files even if unchanged |
| `--batch-size` | `-b` | 10 | Parallel file tasks |
| `--chunk-size` | | 512 | Max tokens per chunk |

```bash
# Only Markdown files, exclude drafts
uv run cli rag add-files ./docs/ -r hybrid_ensemble \
    --include "**/*.md" --exclude "**/drafts/**"

# Re-index everything
uv run cli rag add-files ./docs/ -r persistent --force
```

### `query` — search a retriever

```bash
uv run cli rag query "What is hybrid search?" --retriever hybrid_ensemble
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--retriever` | `-r` | `default` | Retriever config tag |
| `--k` | | 4 | Number of results |
| `--filter` | | — | Metadata filter JSON |
| `--full` | | off | Show full document content |
| `--max-length` | `-l` | 100 | Content preview length |

```bash
# With filter and more results
uv run cli rag query "pricing" -r persistent --k 10 --filter '{"source":"contracts"}'

# Full content view
uv run cli rag query "setup guide" --full
```

### `embed` — embed a single text snippet

```bash
uv run cli rag embed persistent --text "Hybrid search combines dense and sparse retrieval."

# With metadata
uv run cli rag embed persistent --text "BM25 explanation" --metadata '{"source": "manual"}'

# From stdin
echo "Some text to index" | uv run cli rag embed persistent
```

### `delete` — clear a retriever's store

```bash
uv run cli rag delete persistent
# Prompts for confirmation

uv run cli rag delete persistent --force
# Skips confirmation
```

### `info` — show retriever statistics

```bash
uv run cli rag info hybrid_ensemble
```

### `list-retrievers` — show all configured retrievers

```bash
uv run cli rag list-retrievers
```

---

## Batch Ingestion — Prefect Flow

`rag_file_ingestion_flow` processes directories in parallel batches using Prefect tasks.

```python
from genai_tk.extra.prefect.runtime import run_flow_ephemeral
from genai_tk.extra.rag.rag_prefect_flow import rag_file_ingestion_flow

result = run_flow_ephemeral(
    rag_file_ingestion_flow,
    root_dir="./documents",
    retriever_name="persistent",
    max_chunk_tokens=512,
    include_patterns=["**/*.md", "**/*.txt"],
    exclude_patterns=["**/node_modules/**"],
    recursive=True,
    force=False,
    batch_size=10,
)

print(result)
# {
#   "total_files": 142,
#   "processed_files": 38,
#   "skipped_files": 104,
#   "total_chunks": 512
# }
```

The flow uses content hashing to skip unchanged files (unless `force=True`). It only hashes files when the backing store is a persistent Chroma collection (checked via `managed._vector_store`).

See [prefect.md](prefect.md) for Prefect server setup and deployed flow patterns.

---

## Using Retrievers as Agent Tools

`RAGToolFactory` wraps a `ManagedRetriever` in an async LangChain tool.

### In an agent YAML profile

```yaml
# config/agents/langchain.yaml
langchain_agents:
  profiles:
    - name: KnowledgeAgent
      type: react
      llm: default
      system_prompt: "You are a knowledgeable assistant. Use the search tool to answer questions."
      tools:
        - spec: rag_search
          config:
            retriever: hybrid_ensemble
            tool_name: knowledge_search
            tool_description: "Search the company knowledge base"
            default_filter: {source: docs}
            top_k: 5
```

### Programmatically

```python
from genai_tk.tools.langchain.rag_tool_factory import RAGToolConfig, RAGToolFactory
from genai_tk.core.llm_factory import get_llm

config = RAGToolConfig(
    retriever="hybrid_ensemble",
    tool_name="knowledge_search",
    tool_description="Search the company knowledge base for relevant information.",
    default_filter={"source": "docs"},
    top_k=5,
)

tool = RAGToolFactory(get_llm()).create_tool(config)

# In an agent
result = await tool.ainvoke({"query": "What is the refund policy?"})

# With a runtime filter (merged with default_filter)
result = await tool.ainvoke({
    "query": "API limits",
    "filter": '{"section": "pricing"}',
})
```

### Convenience function

```python
from genai_tk.tools.langchain.rag_tool_factory import create_rag_tool_from_config

tool = create_rag_tool_from_config({
    "retriever": "hybrid_ensemble",
    "tool_name": "search_docs",
    "top_k": 5,
})
```

### `RAGToolConfig` fields

| Field | Default | Description |
|-------|---------|-------------|
| `retriever` | required | Key in `retrievers:` YAML section |
| `tool_name` | `rag_search` | Tool name seen by the LLM |
| `tool_description` | built-in | Tool description shown to agent |
| `default_filter` | `None` | Always-applied metadata filter |
| `top_k` | `4` | Max documents returned |

---

## PostgreSQL Hybrid Search

For production-scale deployments, `type: pg_hybrid` offloads both vector search and full-text search to PostgreSQL using the `pgvector` extension.

### Setup

```bash
# External PostgreSQL
uv add langchain-postgres asyncpg

# Or embedded PostgreSQL (no server required, good for local dev)
uv add pgembed pgembed-pgvector
```

### Configuration

```yaml
postgres:
  default:
    mode: external
    url: postgresql+asyncpg://${oc.env:POSTGRES_USER,postgres}:${oc.env:POSTGRES_PASSWORD,password}@localhost:5432/genai

  embedded:
    mode: pgembed
    data_dir: ${paths.data_root}/pgembed
    extensions: [vector]

retrievers:
  pg_hybrid:
    type: pg_hybrid
    embeddings: default
    postgres: default              # or "embedded"
    table_name_prefix: embeddings
    hybrid_search: true
    top_k: 4
    hybrid_search_config:
      tsv_lang: pg_catalog.english
      fusion_function_parameters:
        primary_results_weight: 0.7
        secondary_results_weight: 0.3
```

### Python

```python
from genai_tk.extra.postgres import get_pg_engine, get_postgres_url

# Get the async engine (singleton, cached per config tag)
engine = await get_pg_engine("default")

# Get the connection URL string
url = get_postgres_url("default")
```

---

## Data Flow Summary

```
File/text
    │
    ▼ cli rag add-files  /  managed.add_documents()
rag_prefect_flow (batch, hash-dedup)
    │
    ▼ MarkdownChunker (token-aware splitting)
list[Document]
    │
    ├──► VectorDocumentStore.aadd_documents()
    │         │
    │         ├── RecordManager.index()  [if persistent + dedup enabled]
    │         └── VectorStore.aadd_documents()
    │
    └──► BM25DocumentStore.aadd_documents()
              └── BM25FastRetriever.from_documents() → disk cache


Query
    │
    ▼ managed.aquery(query, k, filter)
    │
    ├── _vector_store.asimilarity_search()   [type: vector / pg_hybrid]
    ├── BM25DocumentStore.aget_relevant_documents()   [type: bm25]
    └── retriever.ainvoke()   [type: ensemble / reranked / zero_entropy]
                │
                ▼ (ensemble) EnsembleRetriever → RRF → top-k
                ▼ (reranked) ContextualCompressionRetriever → reranker → top-k
```
