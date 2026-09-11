---
name: kg-factories
description: Author genai-graph factories that turn a data source into graph nodes — JsonFileBackedFactory, TableBackedFactory, DocumentGraphFactory, MarkdownBamlFactory (inline BAML entity extraction), DocumentDirectoryFactory, Neo4jImportFactory, SimilarityFactory, canonical node reuse, and GraphRegistry multi-source merge. Use when adding a new data source to a Knowledge Graph, writing a factory subclass, or editing genai_graph/kg/factories.
---

# GenAI Graph Factories

## Read First

- `docs/graph-authoring-patterns.md` — pattern catalog (JSON, tables, Neo4j, documents, inline BAML, similarity, canonical reuse, registry merge)
- `docs/graph-definition-guide.md` — steps 5–6 (factory + ingest)
- `docs/graph_construction.md` — factories, canonical types, schema merging, CLI reference
- `docs/baml_extraction_guide.md` — BAML schema → entity graph factory patterns
- `genai_graph/kg/factories/__init__.py` — public exports
- `genai_graph/kg/factories/base.py` — `KgFactory` abstract base
- `genai_graph/kg/factories/document_mixin.py` — `DocumentMixin` (provenance `Document` node)

## Public API

```python
from genai_graph.kg.factories import (
    KgFactory,
    JsonFileBackedFactory,
    TableBackedFactory,
    DocumentDirectoryFactory,
    DocumentGraphFactory,
    DocumentGraphBundle,
    MarkdownBamlFactory,
    DocumentMixin,
    Neo4jFactory,
    Neo4jImportFactory,
    Neo4jNodeMapping,
    Neo4jRelationMapping,
    SimilarityFactory,
    SimilaritySpec,
    SimilarityResult,
)
```

## Choosing a factory

| Data source | Factory | Notes |
|---|---|---|
| JSON files (one dir per model, or a list of objects) | `JsonFileBackedFactory` | Most common. `data_root` + `source_model`. |
| Excel / CSV / SQL tables | `TableBackedFactory` | Load a DataFrame → Pydantic rows. |
| Neo4j JSONL export | `Neo4jImportFactory` (+ `Neo4jNodeMapping`/`Neo4jRelationMapping`) | See `kg-neo4j-import`. |
| Markdown corpus, navigable TOC | `DocumentGraphFactory` | `Folder → Document → MarkdownSection`. See `kg-document-graph`. |
| File-level provenance only | `DocumentDirectoryFactory` | Plain `Document` nodes, no sections. |
| LLM entity extraction from Markdown (inline) | `MarkdownBamlFactory` | Calls a BAML function per file; result cached as JSON. |
| Semantic similarity relations | `SimilarityFactory` (+ `SimilaritySpec`) | Computes embedding-similarity edges between node fields. |

## Pattern — JSON files

```python
from genai_graph.kg.factories import JsonFileBackedFactory
from genai_graph.kg.schema import GraphNode, GraphRelation, GraphSchema
from pydantic import BaseModel


class Customer(BaseModel):
    id: str
    name: str


class Opportunity(BaseModel):
    id: str
    title: str
    customer: Customer


class OpportunityGraph(JsonFileBackedFactory):
    schema = GraphSchema(
        root_model_class=Opportunity,
        nodes=[
            GraphNode(node_class=Opportunity, name_from="title", key_from="id"),
            GraphNode(node_class=Customer, name_from="name", key_from="id"),
        ],
        relations=[GraphRelation(from_node=..., to_node=..., name="FOR_CUSTOMER")],
    )
    source_model = Opportunity
    source_dir = "data/opportunities"
```

`get_keys()` (file discovery) and `get_struct_data_by_key()` (JSON → model) come from the
base class. Implement `build_schema()` (or set a `schema` attribute) and the data hooks.

## Pattern — inline BAML extraction (MarkdownBamlFactory)

Subclass `MarkdownBamlFactory`, implement `build_schema()` and `extract_from_markdown()`.
`get_document_schema_elements(root_node)` (from `DocumentMixin`) adds the provenance
`Document` node + a `MENTIONS` relation, so the same `Document` node MERGEs with the one
`DocumentGraphFactory` produces for the same file.

```python
from genai_graph.kg.factories import MarkdownBamlFactory
from genai_graph.kg.schema import GraphSchema


class ReviewedOpportunityGraph(MarkdownBamlFactory):
    def build_schema(self) -> GraphSchema:
        nodes, relations = self.get_document_schema_elements(ReviewedOpportunityNode)
        return GraphSchema(
            root_model_class=ReviewedOpportunity, nodes=[ReviewedOpportunityNode, *nodes], relations=relations
        )

    def extract_from_markdown(self, md_text: str) -> BaseModel:
        from genai_tk.extra.structured.baml_processor import BamlStructuredProcessor

        processor = BamlStructuredProcessor(
            model_cls=ReviewedOpportunity,
            function_name="ExtractRainbow",
            kvstore_id="",
        )
        return processor.analyze_document("doc", md_text)
```

`md_root` selects the Markdown dir; `json_cache_root` enables the JSON extraction cache
(keyed by mtime) so re-runs are cheap. This is the main consumer of genai-tk's
`BamlStructuredProcessor` — see `genai-tk/baml-structured-extraction`.

## Pattern — canonical node reuse across factories

Define shared node models and `GraphNode` singletons once and import them in every factory.
`GraphRegistry.build_combined_schema()` dedups nodes by `label`
(`table_name or node_class.__name__`), so a canonical `customer_node` referenced by two
factories becomes one `Customer` table.

```python
# schema/canonical_nodes.py
from genai_graph.kg.schema import GraphNode

customer_node = GraphNode(node_class=Customer, name_from="name", key_from="id")

# schema/order_graph.py
from schema.canonical_nodes import customer_node


class OrderGraph(JsonFileBackedFactory):
    schema = GraphSchema(root_model_class=Order, nodes=[order_node, customer_node], relations=[...])
```

When unifying a BAML-generated type with a hand-written one, the canonical class `__name__`
**must match the BAML type's `__name__`** (the table name follows the class name) — extend
the BAML type (`class Partner(BamlPartner): ...`) rather than renaming. See
`kg-schema-maintenance` for the full merge procedure.

## Neo4j factories (mapping-based)

`Neo4jImportFactory` is driven by `Neo4jNodeMapping` / `Neo4jRelationMapping`:
- `neo4j_label` — exact label string from the export.
- `property_mappings: {neo4j_prop: pydantic_field}` — only listed props are imported.
- `key_field` — PRIMARY KEY field (must be a `property_mappings` target). The factory
  auto-populates `mapped_props["id"]` from `key_field`.
- `name_field`, `index_fields` — display name and embedding/text index fields.

Do **not** add an `id: str` field to the Pydantic model unless `key_field="id"` — the
factory generates `id` itself (see `kg-schema-maintenance`, "is the `id` field dead?").

## Change Workflow

1. Pick the factory base from the table above; prefer composition (canonical nodes +
   `GraphRegistry`) over a new factory class.
2. Add a new factory class only when no existing base expresses the behavior.
3. Keep `build_schema()` pure (no I/O); put data access in the base-class hooks
   (`get_keys`, `get_struct_data_by_key`, `extract_from_markdown`, …).
4. Register the factory in a workflow profile (`config/workflows/*.yaml`) under `graphs:`
   with a `factory:` dotted path — see `kg-workflows`.
5. Run `cli kg create <name> --dry-run` then `--force graph` to validate end-to-end.

## Commands

```bash
uv run cli kg create <name> --dry-run
uv run cli kg create <name> --force graph
uv run cli kg schema
uv run just test
```

## Avoid

- Do not duplicate canonical node definitions across factories — import the singleton.
- Do not rename a class that extends a BAML-generated type; the `__name__` must match for
  table unification.
- Do not put I/O or LLM calls inside `build_schema()`.
- Do not reference `embedding_field=` / `EmbeddingField`; use `GraphNode.index_fields` and,
  for similarity relations, `SimilarityFactory` + `SimilaritySpec`.

## Complements

- `genai-tk/baml-structured-extraction` — the `BamlStructuredProcessor` that
  `MarkdownBamlFactory` calls inline.
- `kg-schema` — `GraphSchema`/`GraphNode`/`GraphRelation` that factories build.
- `kg-neo4j-import` — the Neo4j import pipeline and `Neo4jImportFactory` mappings.
- `kg-document-graph` — `DocumentGraphFactory` and the Document Graph schema.
- `kg-schema-maintenance` — merging/unifying node types across factories.
