---
name: genai-tk-rag-systems
description: Build or debug genai-tk RAG retrievers, ingestion, chunking, hybrid search, reranking, and RAG agent tools. Use when editing docs/rag.md areas or config/rag.yaml.
---

# GenAI Toolkit RAG Systems

## Read First

- `docs/rag.md`
- `genai_tk/core/factories/retriever_factory.py`
- `genai_tk/core/retrievers/`
- `genai_tk/workflow/rag/`
- `config/rag.yaml`

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

1. Start with `config/rag.yaml`; add retriever config before adding code.
2. Add new retriever classes under `genai_tk/core/retrievers/` only when composition cannot express the behavior.
3. Keep ingestion logic async-first; sync wrappers are for CLI/notebooks.
4. Preserve metadata through chunking, ingestion, retrieval, and agent tool output.
5. Add tests for config resolution and behavior under `tests/unit_tests/core/` or `tests/unit_tests/workflow/rag/`.

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
