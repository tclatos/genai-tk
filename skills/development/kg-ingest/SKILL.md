---
name: kg-ingest
description: Ingest data into a genai-graph Ladybug (Kuzu-compatible) backend — KuzuBackend, create_graph/restart_database, MERGE/upsert semantics, parquet cache + fingerprint invalidation, embeddings via EmbeddingsHandler, and HNSW vector-index drop/recreate around merge. Use when editing genai_graph/kg/ingest or genai_graph/kg/backend.py, debugging a kg create ingest failure, or tuning cache/force-stage behavior.
---

# GenAI Graph Ingestion & Backend

## Read First

- `docs/graph_construction.md` — factories, canonical types, schema merging, CLI reference
- `docs/cache_management.md` — parquet cache invalidation
- `docs/workflows.md` — force stages (`parquet`/`graph`/`embed`/`all`)
- `genai_graph/kg/backend.py` — `KgBackend`, `KuzuBackend`, `create_backend_from_config`
- `genai_graph/kg/ingest/extract.py` — `create_graph`, `create_schema`, `extract_graph_data`, `restart_database`
- `genai_graph/kg/ingest/merge.py` — `merge_nodes_batch`, `merge_relationships_batch`, `ParquetCollector`
- `genai_graph/kg/embeddings_handler.py` — `EmbeddingsHandler`
- `genai_graph/orchestration/tasks.py` — `drop_vector_indexes_task`, `create_vector_indexes_task`

## Backend

```python
from genai_graph.kg.backend import (
    KgBackend,  # abstract base (ABC)
    KuzuBackend,  # Ladybug — the real backend
    create_backend,  # create_backend("kuzu")
    create_in_memory_backend,  # KuzuBackend connected to ":memory:"
    create_backend_from_config,  # from graph_db.<key> YAML
)
```

- `KuzuBackend` is a Ladybug (maintained Kuzu fork) backend with full Cypher support.
  `Neo4jBackend` exists as a placeholder and raises `NotImplementedError` — do not use it.
- `connect(path)` loads the vector extension automatically so vector tables accept inserts.
- Query helpers: `execute(query, parameters)`, `execute_get_as_df(query, parameters, union=True)`.
- There is **no** `backend.import_neo4j_json(...)` method — Neo4j imports go through the
  `neo4j_import` package / `cli neo4j` (see `kg-neo4j-import`).

## Ingest a single in-memory model

```python
from genai_graph.kg.backend import create_backend_from_config
from genai_graph.kg.ingest import create_graph

backend = create_backend_from_config("my_graph")
create_graph(backend, project, schema)  # creates tables + MERGEs data in one call
```

`create_graph(backend, model, schema)` runs schema creation then extraction+merge.
`restart_database()` returns a fresh in-memory backend (handy for tests/notebooks).

```python
from genai_graph.kg.ingest import restart_database

backend = restart_database()
```

## MERGE / upsert semantics

Ingestion uses `MERGE` (Cypher upsert), so re-ingesting unchanged data is a no-op and
re-ingesting changed data updates in place. Identity is driven by the node's `key_from`
value (see `kg-schema`). Dedup across factories is by `label`
(`GraphRegistry.build_combined_schema`).

## Parquet cache & fingerprints

Factories cache intermediate DataFrames as Parquet per source, keyed by a fingerprint of
the source data. Only changed sources re-run. Force a rebuild with `--force parquet`
(rebuilds import caches, implies a graph rebuild) or `--force all` (full clean rebuild).
`--delete-first` drops the destination DB without touching upstream caches.

```bash
cli kg create <name> --force parquet   # rebuild import caches
cli kg create <name> --force graph     # re-ingest, reuse upstream caches
cli kg create <name> --delete-first    # drop DB, keep caches
cli kg create <name> --force all       # full clean rebuild
```

## Embeddings

`GraphNode.index_fields` marks fields to embed. During ingestion, `EmbeddingsHandler`
(`kg/embeddings_handler.py`) computes embeddings via genai-tk's `EmbeddingsFactory`
(with caching) and stores them in a `{field}_embedding` column. A tuple
`(field, model_id)` in `index_fields` overrides the embedding model for that field.

```python
from genai_graph.kg.embeddings_handler import EmbeddingsHandler

handler = EmbeddingsHandler(embeddings_id="qwen3_06b@openrouter")
vec = handler.compute_embeddings("cloud-native platform")
per_field = handler.compute_field_embeddings(node_data, index_fields=["description"])
```

## Vector indexes around MERGE

Ladybug forbids updating a vector property in place when it is covered by an HNSW index
(`Cannot set property vec ... because it is used in one or more indexes`). The KG flow
handles this automatically: `drop_vector_indexes_task` runs before the MERGE pass and
`create_vector_indexes_task` runs after. The index name is `{field}_index` on the
`{field}_embedding` column, metric `cosine`.

If you still see the error, use `--delete-first` to start from a clean database. When
building a custom flow that re-merges embeddings, mirror this drop→merge→create order.

## Querying the backend directly

```python
df = backend.execute_get_as_df("MATCH (p:Project)-[:FOR_CLIENT]->(c:Company) RETURN p.title, c.name")
rows = backend.execute("MATCH (n) RETURN labels(n), count(*)")
```

For agent-facing query tools (Text-to-Cypher, Document Graph navigation), see `kg-query`.

## Change Workflow

1. Change the schema/factory first (see `kg-schema` / `kg-factories`); ingestion follows.
2. Keep MERGE semantics — never replace `merge_*` with `insert_*` for incremental builds.
3. If you add a new backend method, add it to `KgBackend` (abstract) and implement it on
   `KuzuBackend`; do not call Kuzu-specific methods through the abstract interface in
   shared flow code without an `isinstance(backend, KuzuBackend)` guard.
4. Reproduce the drop→merge→create vector-index order in any custom re-ingest path.
5. Test with `restart_database()` (in-memory) for fast, hermetic unit tests.

## Commands

```bash
uv run cli kg create <name> --dry-run
uv run cli kg create <name> --force graph
uv run cli kg info
uv run cli kg cypher "MATCH (n) RETURN labels(n), count(*)"
uv run just test
```

## Avoid

- Do not use `Neo4jBackend` — it is an unimplemented placeholder.
- Do not call `insert_node`/`insert_relationship` for incremental builds — they create
  duplicates; use the `merge_*` path.
- Do not re-merge into a vector-indexed property without dropping indexes first.
- Do not bypass `create_backend_from_config` by hardcoding DB paths in domain code —
  use the `graph_db` config / `KgManager` path resolution.

## Complements

- `genai-tk/core-models` — the `EmbeddingsFactory` that `EmbeddingsHandler` wraps.
- `kg-schema` — `key_from`/`index_fields` drive identity and embeddings here.
- `kg-workflows` — the `kg_create_step`/`kg_build_step` flows that orchestrate ingestion.
- `kg-neo4j-import` — the Neo4j JSONL → Ladybug path (separate from this backend).
