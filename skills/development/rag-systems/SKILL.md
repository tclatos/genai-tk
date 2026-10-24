---
name: rag-systems
description: Build or debug genai-tk RAG retrievers, ingestion, chunking, hybrid search, reranking, and RAG agent tools. Use when editing docs/rag.md areas or config/rag.yaml.
---

# GenAI Toolkit RAG Systems

## Read First

- `docs/rag.md`
- `genai_tk/core/factories/retriever_factory.py`
- `genai_tk/core/retrievers/`
- `genai_tk/workflow/rag/`
- `config/examples/rag.yaml`

## Architecture

`RetrieverFactory` reads `retrievers.<tag>` from YAML and returns a `ManagedRetriever`. `ManagedRetriever` owns async query, ingestion, deletion, and stats. Retriever implementations remain composable and can point to other named retrievers.

## Retriever Types

| Type | Code | Notes |
|---|---|---|
| Dense vector | `genai_tk/core/retrievers/vector.py` | Uses `embeddings_store` config keys |
| BM25 | `genai_tk/core/retrievers/bm25.py`, `genai_tk/workflow/retrievers/bm25s_retriever.py` | Persists local keyword index |
| Ensemble | `genai_tk/core/retrievers/ensemble.py` | Weighted fusion over named retrievers |
| Reranked | `genai_tk/core/retrievers/reranked.py` | Wraps another retriever |
| PostgreSQL hybrid | `genai_tk/core/retrievers/pg_hybrid.py` | pgvector plus full-text search |
| ZeroEntropy | `genai_tk/core/retrievers/zero_entropy.py`, `genai_tk/workflow/retrievers/zeroentropy.py` | Read-only external retrieval |

## Change Workflow

1. Start with `config/examples/rag.yaml`; add retriever config before adding code.
2. Add new retriever classes under `genai_tk/core/retrievers/` only when composition cannot express the behavior.
3. Keep ingestion logic async-first; sync wrappers are for CLI/notebooks.
4. Preserve metadata through chunking, ingestion, retrieval, and agent tool output.
5. Add tests for config resolution and behavior under `tests/unit_tests/core/` or `tests/unit_tests/workflow/rag/`.

## Vector Backends and Migration

- Use fully qualified backend classes: `InMemoryBackend` for an ephemeral instance, `ZvecBackend` for local persistence, and `PgVectorBackend` for PostgreSQL.
- Chroma is retired. Preserve the explicit migration error for legacy configs before embeddings initialization or filesystem access. Never reinterpret or remove existing Chroma directories.
- New `persistent_zvec` and `local_zvec` retrievers use fresh paths; re-ingest trusted source files and verify results before retiring old data.
- Zvec native collections are opened per operation under a file lock, writes are flushed, and handles are always closed. Existing-path errors must propagate rather than creating a replacement collection.
- Keep Zvec persistence, upsert/delete, concurrent factory, metadata, and legacy-data-preservation regression coverage. Use the public `iter_documents()` capability for ingestion deduplication, not native SDK internals.
- Zvec advertises SHA-256 record-manager keys to respect its 64-character ID limit. Preserve existing Blake2b keys for other backends and test clearing followed by re-ingestion.
- Use the persisted Zvec schema metric for equality-filter ranking when reopening an index; creation defaults must not change ranking or rewrite existing data.
- Zvec supports similarity search and equality dictionary filters; do not imply Chroma filter syntax or MMR compatibility. InMemory instances do not share state across factory calls or CLI runs.

## Commands

```bash
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/core/test_retriever_factory.py -q
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/workflow/rag -q
uv run cli rag --help
```

## Avoid

- Do not make RAG code depend directly on a specific vector backend unless it is a backend implementation.
- Do not drop document metadata during splitting or reranking.
- Do not add network-only behavior to unit tests.
