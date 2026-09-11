---
name: kg-schema
description: Define and compile a genai-graph GraphSchema — GraphNode/GraphRelation, name_from/key_from/table_name, AUTO_ID, field-path auto-deduction, p_ edge-property exclusion, coherence warnings, ResolvedSchema rendering, and embedding index_fields. Use when authoring or debugging a Knowledge Graph schema (genai_graph/kg/schema) or when a kg create / factory build emits schema warnings.
---

# GenAI Graph Schema Definition

## Read First

- `docs/graph-definition-guide.md` — 5-minute guide: models → GraphNode → schema → ingest → query
- `docs/schema-compilation.md` — field-path deduction, `table_name`, exclusion mechanics, compiler functions
- `genai_graph/kg/schema/core.py` — `GraphNode`, `GraphRelation`, `GraphSchema`
- `genai_graph/kg/schema/compiler.py` — `compile_schema`, `deduce_*`, `compute_excluded_fields`, `validate_schema_coherence`
- `genai_graph/kg/schema/resolved.py` — `ResolvedSchema`, `VectorIndexInfo`
- `genai_graph/kg/schema/registry.py` — `GraphRegistry` (multi-graph merge)
- `genai_graph/kg/schema/__init__.py` — public exports

## Public API

```python
from genai_graph.kg.schema import (
    GraphNode,
    GraphRelation,
    GraphSchema,
    GraphRegistry,
    ResolvedSchema,
    VectorIndexInfo,
    generate_schema_description,
    # compiler functions
    build_model_field_map,
    compile_schema,
    compute_excluded_fields,
    deduce_node_field_paths,
    deduce_relation_field_paths,
    validate_schema_coherence,
)
```

## Defining a schema

Node models are plain Pydantic v2 models; nested model fields become relations.

```python
from pydantic import BaseModel


class Company(BaseModel):
    name: str
    sector: str | None = None


class Project(BaseModel):
    title: str
    client: Company  # → relation auto-detected
    team: list[Person] = Field(default_factory=list)
```

Wrap each model in a `GraphNode` and declare relations:

```python
from genai_graph.kg.schema import GraphNode, GraphRelation, GraphSchema

company_node = GraphNode(node_class=Company, name_from="name", key_from="name")
project_node = GraphNode(node_class=Project, name_from="title", key_from="title")

schema = GraphSchema(
    root_model_class=Project,  # entry point for field-path traversal
    nodes=[project_node, company_node],
    relations=[
        GraphRelation(from_node=project_node, to_node=company_node, name="FOR_CLIENT"),
    ],
)
```

Constructing `GraphSchema` runs `compile_schema` automatically (field-path deduction,
exclusion computation, coherence validation). Warnings are emitted as `UserWarning`.

## GraphNode knobs that matter

| Knob | Purpose |
|---|---|
| `node_class` | The Pydantic model this node is built from. |
| `name_from` | Field (or callable `(data, node_type) -> str`) used as the display `name`. |
| `key_from` | Field used as the Ladybug PRIMARY KEY. Default `"AUTO_ID"` → UUID per node. May be a callable. Use a real field when you want stable identity across re-ingests (MERGE). |
| `table_name` | Override the Ladybug table/label (defaults to `node_class.__name__`). Set this when two classes share a name, or to unify a BAML-generated type. |
| `index_fields` | `list[str | tuple[str, str]]` of fields to embed for vector similarity. A tuple `(field, model_id)` overrides the embedding model for that field. This is the implemented embedding knob (not `embedding_field=`). |
| `extra_classes` | Embedded struct models stored as MAP/STRUCT properties on the node. |
| `description` | Free text used in docs and LLM prompts. |
| `explicitly_defined` | `True` for mapping-defined nodes (Neo4j) to skip orphan warnings. |

`GraphNode.label` resolves to `table_name or node_class.__name__` and is the dedup key
across factories (see `GraphRegistry.build_combined_schema`).

## Relations

```python
GraphRelation(from_node=project_node, to_node=company_node, name="FOR_CLIENT")
```

- Omit `field_paths` to let `deduce_relation_field_paths` find them by BFS over
  `root_model_class`. Set `field_paths=[{"from": "", "to": "client"}]` explicitly when the
  same node type appears in multiple places or nested lists.
- The relation `name` is the Cypher relationship type (`[:FOR_CLIENT]`).

## Edge properties — the `p_` convention

Fields prefixed `p_` on a relation endpoint are edge properties (stored on the relation,
excluded from the node). `compute_excluded_fields` populates `node.excluded_fields` from
them. You can also exclude fields manually via `GraphNode(excluded_fields={"internal_notes"})`.

```python
class Contract(BaseModel):
    supplier: Supplier
    p_start_date_: str  # edge property → stored on the relation
    internal_id: str  # node property
```

## Coherence validation

`validate_schema_coherence(schema)` (also `schema.get_warnings()`) returns warning strings,
emitted as `UserWarning` at construction:

- `"No field paths found for X"` — node not reachable from `root_model_class`.
- `"Two different node classes share the label 'X'"` — set `table_name` on one.
- `"Class X is referenced in relationships but has no GraphNode"` — add a `GraphNode` or
  embed the struct instead. See `kg-schema-maintenance` for the fix playbook.

## Multi-graph merge — GraphRegistry

`GraphRegistry` (singleton via `get_graph_registry()`) loads factories for the active KG
profile and merges their schemas. `build_combined_schema()` dedups nodes by `label` and
relations by `(from_label, to_label, name)`. The first selected graph's `root_model_class`
is used for the combined schema.

```python
from genai_graph.kg.schema import get_graph_registry

registry = get_graph_registry()
combined = registry.build_combined_schema()
print(combined.get_warnings())
```

> Note: `docs/graph-authoring-patterns.md` Pattern 7 calls this `KgRegistry`; the
> implemented class is `GraphRegistry` (exported from `genai_graph.kg.schema`).

## Rendering for humans and LLMs

```python
from genai_graph.kg.schema import ResolvedSchema

resolved = ResolvedSchema.from_graph_schema(schema)
print(resolved.to_markdown())  # table summary
resolved.to_html_file("schema.html")  # interactive D3 diagram
resolved.to_json_str()  # D3 JSON for tools/prompts
```

Inject BAML field/class descriptions via
`ResolvedSchema.from_graph_schema(schema, descriptions=_parse_baml_descriptions(file_map))`.

## Change Workflow

1. Change the Pydantic models and `GraphNode`/`GraphRelation` declarations first; the
   compiler re-derives field paths and exclusions automatically.
2. Only reach for explicit `field_paths`, `table_name`, or `excluded_fields` when
   auto-deduction is ambiguous (multiple paths) or a label collides.
3. Run `validate_schema_coherence(schema)` (or `schema.get_warnings()`) and fix every
   warning before ingesting — they signal real identity/dedup bugs.
4. Re-ingest with `cli kg create <name> --force graph` (or `--delete-first`) after a schema
   change so the Ladybug tables match.

## Commands

```bash
uv run cli kg schema                       # print the active profile's schema
uv run cli kg create <name> --force graph  # rebuild DB after a schema change
uv run just test
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/kg -q   # if a kg unit path exists
```

## Avoid

- Do not add an `id: str` field to a model unless `key_from="id"` — Neo4j/Json factories
  auto-generate `id` from `key_field`; a stray `id` is dead code (see `kg-schema-maintenance`).
- Do not rely on `embedding_field=` / `EmbeddingField` — the implemented knob is
  `GraphNode.index_fields` (computed by `EmbeddingsHandler`).
- Do not set `table_name` to fix a collision without also updating relations that reference
  the node — relations dedup by label, so a rename can silently drop edges.
- Do not silence `UserWarning`s from schema construction; fix the underlying issue.

## Complements

- `genai-tk/core-models` — the LLM/embeddings factories that `EmbeddingsHandler` builds on.
- `kg-factories` — factories that produce the Pydantic instances a schema consumes.
- `kg-schema-maintenance` — procedures for merging node types, dead-field checks, warnings.
