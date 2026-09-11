---
name: kg-query
description: Query a genai-graph Knowledge Graph and build Cypher-aware agents — direct Cypher via the backend, text2cypher_chain/query_kg, the kg_cypher_query LangChain tool and build_kg_agent_system_prompt, CALL QUERY_VECTOR_INDEX semantic search with $query_vector, Document Graph navigation tools, and SimilarityFactory similarity relations. Use when editing genai_graph/kg/query, writing a KG agent, or wiring graph tools into an agent profile.
---

# GenAI Graph Querying & Cypher Agents

## Read First

- `docs/kg_explorer.md` — Streamlit Cypher UI + Text-to-Cypher
- `docs/document-graph.md` — Document Graph navigation tools
- `genai_graph/kg/query/text2cypher.py` — `SYSTEM_PROMPT`, `text2cypher_chain`, `query_kg`
- `genai_graph/kg/query/agent.py` — `build_kg_agent_system_prompt`, `create_kg_cypher_tool`
- `genai_graph/kg/query/document_graph_tools.py` — Document Graph tools
- `genai_graph/kg/factories/similarity.py` — `SimilarityFactory`, `SimilaritySpec`
- `genai_graph/kg/query/__init__.py` — public exports

## Public API

```python
from genai_graph.kg.query import (
    SYSTEM_PROMPT,  # canonical Cypher authoring guidelines
    text2cypher_chain,  # question -> Runnable producing Cypher
    query_kg,  # end-to-end question -> Cypher -> results
    build_kg_agent_system_prompt,  # schema-aware agent system prompt
    create_kg_cypher_tool,  # LangChain tool executing Cypher
)
from genai_graph.kg.query.document_graph_tools import create_document_graph_tools
from genai_graph.kg.factories import SimilarityFactory, SimilaritySpec
```

## 1. Direct Cypher (backend)

```python
from genai_graph.kg.backend import create_backend_from_config

backend = create_backend_from_config("my_graph")
df = backend.execute_get_as_df("MATCH (p:Project)-[:FOR_CLIENT]->(c:Company) RETURN p.title, c.name")
```

See `kg-ingest` for the full backend API. This is what the CLI `cli kg cypher`/`cli kg query` use.

## 2. Text-to-Cypher (one-shot)

```python
from genai_graph.kg.query import query_kg, text2cypher_chain

chain = text2cypher_chain(question, llm="default", kg_config_name="my_graph")
cypher = chain.invoke({...})  # produces a Cypher statement
result = query_kg(question, llm="default", kg_config_name="my_graph")  # end-to-end
```

`SYSTEM_PROMPT` (in `text2cypher.py`) is the canonical guidance: use only schema labels/rels,
shortest path ≤ 4 hops, start with `MATCH`/`OPTIONAL MATCH`, end with `RETURN`, `toLower()` +
`CONTAINS` for strings, no APOC, `LIMIT 30`, `RETURN DISTINCT`. Reuse it when building custom
text-to-Cypher prompts so behavior stays consistent.

## 3. KG agent tool (LangChain)

`create_kg_cypher_tool(*, backend_config="default", kg_config_name=None, console=None, debug=False)`
returns a LangChain tool named **`kg_cypher_query`** that executes read-only Cypher and
returns a markdown table. If the Cypher contains `$query_vector`, pass the user `question`
so the tool computes the embedding (requires `kg_build.embeddings.default` in config).

```python
from genai_graph.kg.query import create_kg_cypher_tool, build_kg_agent_system_prompt

tool = create_kg_cypher_tool(kg_config_name="my_graph")
system_prompt = build_kg_agent_system_prompt(kg_config_name="my_graph")
# tool name: "kg_cypher_query"; args: cypher_query (str), question (str="")
```

`build_kg_agent_system_prompt(single_tool_mode=False, kg_config_name=None)` embeds the
graph schema (loaded from the canonical schema JSON, falling back to the markdown file) and
the Cypher guidelines. It requires a schema file to exist — generate one with
`cli kg create` or `cli kg schema --regen --kg <profile>`.

Wire the tool into an agent profile like any LangChain tool (see `genai-tk/add-tool`):

```yaml
agents:
  kg_agent:
    harness: langchain
    type: react
    llm: default
    tools:
      - function: genai_graph.kg.query.create_kg_cypher_tool
        kg_config_name: my_graph
```

Note: `create_kg_cypher_tool` is a factory returning a `BaseTool`; reference it via
`function:` so the profile loader calls it. (The `factory:` spec expects a `list[BaseTool]`.)

## 4. Vector / semantic search

Vector indexes live on `{field}_embedding` columns (index name `{field}_index`), created by
`create_vector_indexes_task` during `cli kg create` (see `kg-ingest`). Query them with
`CALL QUERY_VECTOR_INDEX`:

```cypher
CALL QUERY_VECTOR_INDEX('L3', 'description_index', $query_vector, 10)
WITH node AS l3, distance
RETURN DISTINCT l3.name, l3.description, distance
ORDER BY distance LIMIT 10
```

`$query_vector` is a runtime parameter — the `kg_cypher_query` tool embeds the user question
and injects it when the placeholder is present. The schema JSON's `### Vector-Indexed Fields`
section tells the model which indexes exist. Kuzu returns cosine **distance**
(lower = more similar); `SimilarityFactory` converts it to similarity.

## 5. Document Graph navigation tools

For corpus Q&A without embeddings (vectorless agentic RAG):

```python
from genai_graph.kg.query.document_graph_tools import create_document_graph_tools

tools = create_document_graph_tools("./data/kg/tree.db")
# -> [list_documents, get_document_toc, get_section_content, search_sections]
```

Loop: `list_documents` → `get_document_toc` (the map) → `get_section_content` (only the
sections worth reading) → answer. See `kg-document-graph`.

## 6. Similarity relations (build time)

`SimilarityFactory` (`kg/factories/similarity.py`) creates typed relationships between nodes
whose embedding cosine similarity exceeds a threshold. It produces **no new nodes** — it
reads embeddings from already-ingested nodes and runs after `create_vector_indexes_task`.

```yaml
graphs:
  - factory: mypackage.schema.matcher.MyMatcher
    similarities:
      - relationship: POSSIBLE_OFFERING
        from: TechnicalApproach.architecture   # source node + embedding field
        to: L3.description                     # target node + HNSW-indexed field
        iterate_over: from                     # loop over the smaller side
        threshold: 0.8
        top_k: 5
```

`iterate_over` is a performance knob — set it to the side with fewer nodes so the HNSW index
on the larger side does the work. Relationship direction is always `(from)-[:REL]->(to)`.

## Change Workflow

1. Keep `SYSTEM_PROMPT` as the single source of Cypher authoring guidance; both
   `text2cypher_chain` and `build_kg_agent_system_prompt` reference it.
2. When adding a query tool, expose it as a LangChain `BaseTool` and reuse
   `create_backend_from_config` + the schema JSON — don't hand-roll backend wiring.
3. Schema-affecting changes require regenerating the schema file (`cli kg schema --regen`)
   so `build_kg_agent_system_prompt` picks up new labels/properties/vector indexes.
4. For new similarity relations, add a `SimilaritySpec` to a `SimilarityFactory` subclass
   rather than writing ad-hoc similarity Cypher.

## Commands

```bash
cli kg cypher "MATCH (n) RETURN labels(n), count(*)"
cli kg query "Which companies have the most projects?"
cli kg schema --regen --kg my_graph
uv run just test
```

## Avoid

- Do not invent labels/relationships/properties not in the schema — `SYSTEM_PROMPT` forbids it.
- Do not use APOC — Ladybug does not support it.
- Do not write `[:TYPE>]` / `[:TYPE<]`; the arrow is always outside the brackets: `(a)-[:T]->(b)`.
- Do not compute query embeddings yourself — use the `$query_vector` placeholder and let
  `kg_cypher_query` / `query_kg` embed the question.
- Do not reference `SimilarityFactory` from `genai_graph.kg.query` — it lives in
  `genai_graph.kg.factories`.

## Complements

- `genai-tk/agent-profiles` — how `tools:`/`harness:`/`type:` wire into a running agent.
- `genai-tk/add-tool` — the `function:`/`factory:`/`class:` tool spec format.
- `kg-document-graph` — the Document Graph navigation tools in detail.
- `kg-ingest` — vector index creation and `$query_vector` embedding setup.
- `kg-explorer` — the Streamlit UI over these same query primitives.
