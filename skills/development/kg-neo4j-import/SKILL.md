---
name: kg-neo4j-import
description: Import Neo4j JSONL exports into a genai-graph Ladybug database — SchemaAnalyzer, Neo4jToKuzuConverter, the cli neo4j analyze/convert/subset/import/query/info commands, and the typed Neo4jImportFactory with Neo4jNodeMapping/Neo4jRelationMapping. Use when migrating a Neo4j graph, editing genai_graph/neo4j_import, or wiring a Neo4j source into a multi-factory KG.
---

# GenAI Graph Neo4j Import

## Read First

- `docs/graph-authoring-patterns.md` — Pattern 3 (Neo4j exports)
- `genai_graph/neo4j_import/__init__.py` — public exports
- `genai_graph/neo4j_import/schema_analyzer.py` — `SchemaAnalyzer`
- `genai_graph/neo4j_import/converter.py` — `Neo4jToKuzuConverter`
- `genai_graph/neo4j_import/kuzu_manager.py` — Ladybug import helpers
- `genai_graph/neo4j_import/commands.py` — `Neo4jCommands` (`cli neo4j`)
- `genai_graph/kg/factories/neo4j_factory.py` — `Neo4jImportFactory`, `Neo4jNodeMapping`, `Neo4jRelationMapping`

## Two ways to bring Neo4j data in

| Path | What it does | When to use |
|---|---|---|
| **Raw import** (`cli neo4j import`) | Preserves Neo4j labels/relationship types/properties verbatim into Ladybug. | Migrating a whole Neo4j graph as-is; one-off load. |
| **Typed factory** (`Neo4jImportFactory`) | Selective, Pydantic-typed import via `Neo4jNodeMapping`/`Neo4jRelationMapping`; MERGEs with other factories by label. | You want a curated subset that unifies with BAML/JSON/table sources in one KG. |

## CLI

```bash
# Analyze a Neo4j JSONL export → Ladybug schema statements
cli neo4j analyze export.jsonl -o schema.cypher

# Convert JSONL → per-label JSON files (for COPY FROM)
cli neo4j convert export.jsonl ./ladybug_import

# Create a small test subset
cli neo4j subset export.jsonl subset.jsonl --max-nodes 20 --max-rels 20

# Import into a Ladybug DB (raw, verbatim labels/rels)
cli neo4j import export.jsonl --db path/to/ladybug.db -f

# Query an imported DB
cli neo4j query "MATCH (n) RETURN labels(n), count(*)" --db path/to/ladybug.db

# DB info
cli neo4j info --db path/to/ladybug.db
```

All path arguments accept `${paths.*}` config variables (resolved via
`genai_tk.config_mgmt.file_patterns.resolve_config_path`).

## Programmatic API

```python
from genai_graph.neo4j_import import SchemaAnalyzer, Neo4jToKuzuConverter
from pathlib import Path

analyzer = SchemaAnalyzer(Path("export.jsonl"))
analyzer.analyze()
statements = analyzer.generate_kuzu_schema()  # CREATE NODE TABLE / CREATE REL TABLE …

converter = Neo4jToKuzuConverter(Path("export.jsonl"))
stats = converter.convert(Path("./ladybug_import"))  # per-label JSON files
```

`SchemaAnalyzer` extracts node labels, relationship types, and properties and emits
Ladybug `CREATE NODE TABLE` / `CREATE REL TABLE` statements. `Neo4jToKuzuConverter` writes
separate JSON files per label/rel type, suitable for Ladybug `COPY FROM`.

## Typed factory (selective, merges with other sources)

When you want a curated Neo4j subset that unifies with other factories in one KG, use
`Neo4jImportFactory` with explicit mappings (see `kg-factories`):

- `Neo4jNodeMapping(neo4j_label=..., property_mappings={neo4j_prop: pydantic_field},
   key_field=..., name_field=..., index_fields=[...])` — only listed props are imported.
- `Neo4jRelationMapping` — `from_node`/`to_node` classes + `neo4j_rel_type`.
- The factory auto-populates `mapped_props["id"]` from `key_field`, so do **not** add an
  `id: str` field to the Pydantic model unless `key_field="id"` (see `kg-schema-maintenance`).
- Add the rel type string to `get_included_rel_types()` so it is imported.

Register it under `graphs:` in a workflow profile like any factory (see `kg-workflows`).

## Change Workflow

1. For a one-off migration, use the CLI (`analyze` → `import`); no code needed.
2. For a recurring, typed source, add a `Neo4jImportFactory` subclass with mappings and
   register it in a workflow profile so it MERGEs with other factories by label.
3. Keep `neo4j_label` exact (it matches source data) but let the canonical Pydantic
   `__name__`/`table_name` drive the Ladybug table when unifying types.
4. Resolve paths through `${paths.*}` so the CLI works across profiles/environments.

## Commands

```bash
cli neo4j analyze export.jsonl -o schema.cypher
cli neo4j import export.jsonl --db ./data/kg/neo.db -f
uv run just test
```

## Avoid

- Do not add `id: str` to Neo4j-mapped models unless `key_field="id"` — the factory
  generates `id` from `key_field`.
- Do not mix the raw `cli neo4j import` path and the typed `Neo4jImportFactory` path into
  the same database expecting them to MERGE — raw import keeps Neo4j labels verbatim and
  does not go through the `GraphRegistry` label dedup.
- Do not import relationship types you did not add to `get_included_rel_types()`.

## Complements

- `kg-factories` — `Neo4jImportFactory` and the mapping types in context.
- `kg-ingest` — the Ladybug backend the importer writes into.
- `kg-schema-maintenance` — the "is the `id` field dead?" check for Neo4j-mapped models.
- `kg-cli` — the `cli neo4j` command group wiring.
