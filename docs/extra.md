# Extra Module (`genai_tk.extra`)

> **Quick nav:** [Agent Graphs](#agent-graphs-extragraphs) · [RAG Systems](#rag-systems-workflowrag) · [Data Loaders](#data-loaders-workflowloaders) · [Retrievers](#retrievers-workflowretrievers) · [Anonymization](#anonymization-extraanonymization) · [BAML / Structured Extraction](#structured-extraction-baml-extrastructured) · [Image Analysis](#image-analysis-extraimage_analysis) · [KV Store](#kv-store-extrakv_store)

## Overview

The `extra` module contains non-pipeline tooling: agent graphs, anonymization, image analysis, KV store, BAML extraction, and the Mistral OCR / PPT conversion commands.

ETL-oriented components (RAG, loaders, retrievers, and Prefect flows) live in **`genai_tk.workflow`** — see [prefect.md](prefect.md) and [rag.md](rag.md) for full documentation.

## Agent Graphs (`extra.graphs`)

Pre-built, specialized agent implementations using LangGraph for common patterns.

### ReAct Agent (`custom_react_agent.py`)

Standard Reasoning + Acting agent built from scratch using LangGraph's Functional API.

**Features:**
- Thought-Action-Observation loop
- Tool binding and execution
- Message history management
- Checkpointer support for persistence

**Usage:**
```python
from genai_tk.extra.graphs.custom_react_agent import create_custom_react_agent
from genai_tk.core.factories.llm_factory import get_llm
from langgraph.checkpoint.memory import MemorySaver

# Create agent
llm = get_llm("gpt_4o@openai")
tools = [your_tools_here]
checkpointer = MemorySaver()

agent = create_custom_react_agent(llm, tools, checkpointer)

# Use agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "Your query"}]
})
```

**When to Use:**
- Need custom agent logic
- Building specialized workflows
- Fine-grained control over agent behavior

### SQL Agent (`sql_agent.py`)

Specialized agent for SQL database interactions with natural language queries.

**Features:**
- SQL generation from natural language
- Query execution and result interpretation
- Schema exploration and discovery
- Error handling and query correction

**Configuration:**
```yaml
tools:
  - spec: sql_tools
    config:
      database_url: sqlite:///./data.db
      schema: public
      include_tables: ["users", "orders"]
      sample_rows: 3
```

**Usage:**
```python
from genai_tk.extra.graphs.sql_agent import create_sql_querying_graph
from langchain_community.utilities import SQLDatabase

# Create agent for database
db = SQLDatabase.from_uri("sqlite:///./data.db")
agent = create_sql_querying_graph(llm=get_llm(), db=db)

# Query database naturally
result = agent.invoke({
    "messages": [{"role": "user", "content": "How many users signed up last month?"}]
})
```

**Common Patterns:**
```python
# With custom examples
from genai_tk.extra.graphs.sql_agent import create_sql_querying_graph

examples = [
    {"query": "How many users?", "sql": "SELECT COUNT(*) FROM users"},
    {"query": "Top products", "sql": "SELECT product_id, COUNT(*) as count FROM orders GROUP BY product_id ORDER BY count DESC LIMIT 5"}
]

graph = create_sql_querying_graph(
    llm=get_llm(),
    db=db,
    examples=examples,
    top_k=10
)
```

### ReAct with Structured Output (`react_agent_structured_output.py`)

ReAct agent that outputs validated Pydantic models instead of free-form text.

**Features:**
- Type-safe responses
- Automatic model validation
- Tool calls with structured output
- Error recovery for validation failures

**Usage:**
```python
from pydantic import BaseModel
from genai_tk.extra.graphs.react_agent_structured_output import (
    create_react_structured_output_graph
)

class ResearchResult(BaseModel):
    """Research findings."""
    title: str
    key_points: list[str]
    sources: list[str]
    confidence: float

agent = create_react_structured_output_graph(
    llm=get_llm(),
    tools=tools,
    out_model_class=ResearchResult
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Research AI trends"}]
})

# result.output is validated ResearchResult instance
print(result.output.title)
print(result.output.key_points)
```

## RAG Systems (`workflow.rag`)

> **Full documentation:** see **[rag.md](rag.md)** for the complete guide covering design, all retriever types, YAML config reference, Python API, CLI commands, Prefect batch ingestion, agent tool integration, and PostgreSQL hybrid search.

RAG files live under `genai_tk/workflow/rag/` (chunkers, CLI commands) and `genai_tk/workflow/prefect/flows/rag_flow.py` (ingestion flow).

**Quick reference:**

```python
from genai_tk.core.factories.retriever_factory import RetrieverFactory

# Create retriever from YAML config tag
managed = RetrieverFactory.create("hybrid_ensemble")

# Query (async-first)
docs = await managed.aquery("What is vector search?", k=5)

# Ingest documents
await managed.aadd_documents(my_docs)
```

```bash
# CLI
uv run cli rag add-files ./docs/ --retriever persistent
uv run cli rag query "hybrid search" --retriever hybrid_ensemble
uv run cli rag list-retrievers
```

## Data Loaders (`workflow.loaders`)

Utilities for loading and processing various document formats.  Files live under `genai_tk/workflow/loaders/`.

### Markdown Loader (`markdown_loader.py`)

Loads and parses markdown files with metadata preservation.

**Features:**
- Frontmatter extraction (YAML/TOML)
- Code block preservation
- Link extraction
- Metadata conversion to document attributes

**Usage:**
```python
from genai_tk.workflow.loaders.markdown_loader import MarkdownLoader

loader = MarkdownLoader()
documents = loader.load_file("document.md")

# Extract metadata
for doc in documents:
    print(f"Title: {doc.metadata.get('title')}")
    print(f"Tags: {doc.metadata.get('tags')}")
```

### OCR Loader (`mistral_ocr.py`)

Extract text from images and PDFs using Mistral's OCR service.

**Features:**
- Image-to-text conversion
- PDF text extraction
- Handwriting recognition
- Table structure preservation

**Configuration:**
```yaml
extra:
  ocr:
    provider: mistral
    api_key_env: MISTRAL_API_KEY
    image_quality: high
```

**Usage:**

> **Requires:** `uv add mistralai`

```python
from genai_tk.workflow.loaders.mistral_ocr import MistralOCRLoader

loader = MistralOCRLoader(api_key="your-key")

# Extract from image
text = await loader.extract_text("image.png")

# Extract from PDF
pages = await loader.extract_from_pdf("document.pdf")
for page_num, text in enumerate(pages):
    print(f"Page {page_num}: {text}")
```

## Retrievers (`workflow.retrievers`)

> See **[rag.md](rag.md)** for the full retriever documentation.

Low-level retriever implementations live under `genai_tk/workflow/retrievers/` and are used by `RetrieverFactory`:

- `BM25FastRetriever` — bm25s-backed keyword retriever with optional Spacy preprocessing
- `ZeroEntropyRetriever` — read-only retriever backed by the ZeroEntropy SDK

## Utility Functions

### Anonymization

PII detection and replacement using [Presidio](https://microsoft.github.io/presidio/) + [Faker](https://faker.readthedocs.io/).
Core logic lives in `genai_tk/workflow/anonymization/` and is shared by the Prefect ETL flow
and the `AnonymizationMiddleware` agent middleware — identical behaviour in both contexts.

> **Requires:** `uv add presidio-analyzer presidio-anonymizer spacy`

**Standalone usage (batch / scripting):**

```python
from faker import Faker
from genai_tk.workflow.anonymization.core import AnonymizationConfig, anonymize_text
from genai_tk.workflow.anonymization.presidio_detector import (
    CustomRecognizerConfig, PresidioDetector, PresidioDetectorConfig,
)

config = PresidioDetectorConfig(
    analyzed_fields=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"],
    # Add domain-specific entities via regex recognizers:
    custom_recognizers=[
        CustomRecognizerConfig(
            entity_name="COMPANY",
            patterns=[r"(?i)\b(Acme Corp|Tech Solutions)\b"],
            context=["company", "firm"],
        ),
        CustomRecognizerConfig(
            entity_name="PRODUCT",
            patterns=[r"(?i)\b(WidgetPro|CloudMaster)\b"],
            context=["product", "service"],
        ),
    ],
)
detector = PresidioDetector(config=config)
Faker.seed(42)
faker = Faker(["en_US"])

anonymized, mapping = anonymize_text(
    "John Smith at Acme Corp: john@acme.com",
    detector=detector,
    faker=faker,
)
# mapping = {"John Smith": "Jane Doe", "Acme Corp": "Hoeger LLC", ...}
```

**Supported entity types** (Presidio built-ins + custom via `CustomRecognizerConfig`):
`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, `LOCATION`, `IBAN_CODE`,
`US_SSN`, `IP_ADDRESS`, `URL`, `DATE_TIME`, `ORG` — and custom `COMPANY`, `PRODUCT`, `PROJECT`.

For agent use, configure `AnonymizationMiddleware` — see [middleware-pii-and-routing.md](middleware-pii-and-routing.md).
For ETL/batch use, configure the `anonymize` workflow — see [workflows.md](workflows.md).

### Image Analysis (`image_analysis.py`)

Computer vision capabilities for image understanding and analysis.

**Features:**
- Image classification
- Object detection
- Scene understanding
- Multi-image analysis

**Usage:**
```python
from genai_tk.extra.image_analysis import image_query_message
from genai_tk.core.factories.llm_factory import get_llm

# Build a multimodal message with an image
llm = get_llm()
messages = image_query_message(
    {"image_path": "image.jpg"},
    {"query": "What objects are in this image?"}
)
response = llm.invoke(messages)
```

### GPT Researcher Helper (`gpt_researcher_helper.py`)

Integration with GPT Researcher for autonomous research tasks.

**Features:**
- Research report generation
- Source aggregation
- Citation management
- Multi-source synthesis

**Usage:**

> **Requires:** `uv add gpt-researcher`

```python
from genai_tk.extra.gpt_researcher_helper import ResearchAgent

researcher = ResearchAgent(llm=get_llm())

report = await researcher.research("Latest trends in AI 2024")
print(report.title)
print(report.content)
print(report.citations)
```

### KV Store Registry (`kv_store_registry.py`)

Registry for key-value stores used in caching and persistence.

**Supported backends** (selected by `type` field):

| `type` | Class | Notes |
|---|---|---|
| `LocalFileStore` | `LocalFileStoreConfig` | Files on disk — default for dev/prod |
| `SQLStore` | `SQLStoreConfig` | Any SQLAlchemy DSN (sqlite, postgresql) |
| `memory` | `MemoryStoreConfig` | Ephemeral temp dir — use for tests |

**Configuration:**
```yaml
kv_store:
  default:
    type: LocalFileStore
    path: ${paths.data_root}/kv_store
  sql_cache:
    type: SQLStore
    path: postgresql://${oc.env:POSTGRES_USER}:${oc.env:POSTGRES_PASSWORD}@localhost:5432/cache
  # Tests / pytest profile:
  default:
    type: memory
```

**Usage:**
```python
from genai_tk.extra.kv_store_registry import get_kv_store

store = get_kv_store()                      # uses "default" entry
store = get_kv_store("sql_cache", namespace="llm_cache")

# ByteStore interface (LangChain compatible)
store.mset([("key", b"value")])
values = store.mget(["key"])
```

**Typed config access (Case 2):**
```python
from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.extra.kv_store_registry import KvStoreConfig

all_stores = global_config().section_dict("kv_store", KvStoreConfig, inject_name=False)
```

### PostgreSQL / PgVector (`core.vector_backends.pgvector`)

PostgreSQL vector database integration for scalable embeddings. All Postgres connection management and PgVector factory logic lives in `genai_tk/core/vector_backends/pgvector.py`.

**Features:**
- Vector similarity search
- Hybrid search (keyword + semantic via pgvector)
- Embedded PostgreSQL via pgembed (no server needed for dev)
- Connection pooling / engine singleton per config tag

**Configuration:**
```yaml
postgres:
  default:
    mode: external
    url: postgresql+asyncpg://${oc.env:POSTGRES_USER}:${oc.env:POSTGRES_PASSWORD}@localhost:5432/db
  embedded:
    mode: pgembed
    data_dir: ${paths.data_root}/pgembed
    extensions: [vector]

embeddings_store:
  pg_store:
    backend: genai_tk.core.vector_backends.PgVectorBackend
    embeddings: default
    config:
      postgres: default
      hybrid_search: false
```

**Usage:**
```python
from genai_tk.core.vector_backends.pgvector import get_pg_engine, PgVectorConfig

# Get a cached async PGEngine
engine = get_pg_engine("default")

# Create a store via EmbeddingsStore
from genai_tk.core.embeddings_store import EmbeddingsStore
store = EmbeddingsStore.create_from_config("pg_store")
vs = store.get_vector_store()
vs.add_documents(documents)
results = vs.similarity_search("query", k=5)
```

> See [rag.md](rag.md) for full PostgreSQL hybrid search documentation.

## Structured Extraction (`extra.structured`)

BAML-powered structured output extraction.  See **[baml.md](baml.md)** for the complete guide
covering setup, CLI commands, programmatic API, and troubleshooting.

**Quick reference:**

```bash
# Single extraction
uv run cli baml run ExtractResume -i "John Smith; SW engineer"

# Batch extraction from a directory of Markdown files
uv run cli baml extract ./docs ./output --recursive --function ExtractRainbow
```

```python
from genai_tk.extra.structured.baml_processor import BamlStructuredProcessor
from baml_client.types import Resume

processor = BamlStructuredProcessor[Resume](
    function_name="ExtractResume",
    llm="gpt_4o@openai",
)
results = await processor.abatch_analyze_documents(ids, markdown_texts)
```

## Document Conversion (`extra.tools`)

See **[prefect.md](prefect.md)** for the complete Prefect integration guide.

```bash
# Convert PDF / DOCX / PPTX to Markdown
uv run cli tools markdownize ./input ./output --recursive

# High-quality PDF extraction via Mistral OCR
uv run cli tools markdownize ./pdfs ./output --mistral-ocr

# Convert presentations to PDF
uv run cli tools ppt2pdf ./slides ./pdfs --recursive
```

## Advanced Patterns

### Pattern 1: Multi-Step RAG with SQL

```python
from genai_tk.extra.graphs.sql_agent import create_sql_querying_graph
from genai_tk.core.embeddings_store import EmbeddingsStore

# Step 1: Create SQL graph for structured data
sql_graph = create_sql_querying_graph(llm=get_llm(), db=db)

# Step 2: Create RAG for unstructured data
vector_store = EmbeddingsStore.from_config()
rag_retriever = vector_store.as_retriever()

# Step 3: Use in higher-level orchestrating agent with tools for both
from genai_tk.agents.langchain.factory import create_langchain_agent

# Create a ReAct agent that can use both SQL and RAG tools
agent = await create_langchain_agent(profile, extra_tools=[
    rag_retriever.as_tool()
])
```

### Pattern 2: Structured Output RAG

```python
from pydantic import BaseModel
from genai_tk.extra.graphs.react_agent_structured_output import (
    create_react_structured_output_graph
)

class DocumentSummary(BaseModel):
    title: str
    key_insights: list[str]
    action_items: list[str]
    sentiment: str

# RAG + Structured output
agent = create_react_structured_output_graph(
    llm=get_llm(),
    tools=[rag_retriever],
    out_model_class=DocumentSummary
)

result = agent.invoke({"messages": [...]})
```

### Pattern 3: Privacy-Aware Data Processing

The recommended approach is to configure `AnonymizationMiddleware` in the agent profile — it
automatically anonymizes every human message before it reaches the LLM and restores PII in
the response, with full thread isolation for concurrent conversations.

```yaml
# config/agents/langchain/simple.yaml
langchain_agents:
  privacy_agent:
    name: "Privacy Agent"
    type: react
    llm: default
    middlewares:
      - class: genai_tk.agents.langchain.middleware.anonymization_middleware:AnonymizationMiddleware
        analyzed_fields: [PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD]
        faker_seed: 42
        fuzzy_deanonymize: true
```

For programmatic use (e.g. pre-processing documents before ingestion):

```python
from faker import Faker
from genai_tk.workflow.anonymization.core import anonymize_text
from genai_tk.workflow.anonymization.presidio_detector import PresidioDetector, PresidioDetectorConfig

detector = PresidioDetector(config=PresidioDetectorConfig())
Faker.seed(42)
faker = Faker(["en_US"])

safe_text, mapping = anonymize_text(user_input, detector=detector, faker=faker)

result = await agent.ainvoke({"messages": [{"role": "user", "content": safe_text}]})

# Restore PII in response
final = result["messages"][-1].content
for real, fake in mapping.items():
    final = final.replace(fake, real)
```

## Performance Optimization

**Chunking Strategy:**
```python
# For long documents
chunker = MarkdownChunker(
    chunk_size=2048,      # Larger chunks
    chunk_overlap=400     # More overlap
)
```

**Embedding Caching:**
```python
# Cache embeddings to reduce API calls
from langchain.embeddings import CacheBackedEmbeddings

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings,
    cache_store
)
```

**Hybrid Retrieval:**
```python
# Combine semantic and keyword search
ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # Weight vector search more
)
```

## See Also

- [Core Module](core.md) - LLM Factory and configuration
- [Agents Module](agents.md) - Agent implementations
- [Configuration Guide](../config/README.md) - Detailed configuration
- [MCP Servers](mcp-servers.md) - Model Context Protocol
- [Sandbox Support](sandbox_support.md) - Sandboxed execution
- [Testing Guide](TESTING_GUIDE.md) - Testing patterns
- [Development Guidelines](../AGENTS.md) - Code standards
