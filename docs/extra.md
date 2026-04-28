# Extra Module (`genai_tk.extra`)

> **Quick nav:** [Agent Graphs](#agent-graphs-extragraphs) · [RAG Systems](#rag-systems-extrarag) · [Data Loaders](#data-loaders-extraloaders) · [Retrievers](#retrievers-extraretrievers) · [Anonymization](#anonymization-extraanonymization) · [BAML / Structured Extraction](#structured-extraction-baml-extrastructured) · [Image Analysis](#image-analysis-extraimage_analysis) · [KV Store](#kv-store-extrakv_store)

## Overview

The `extra` module provides advanced AI capabilities beyond core agent functionality, including specialized agent graphs, RAG pipelines, data loaders, and utility functions for complex AI workflows.

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
from genai_tk.core.llm_factory import get_llm
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

## RAG Systems (`extra.rag`)

> **Full documentation:** see **[rag.md](rag.md)** for the complete guide covering design, all retriever types, YAML config reference, Python API, CLI commands, Prefect batch ingestion, agent tool integration, and PostgreSQL hybrid search.

**Quick reference:**

```python
from genai_tk.core.retriever_factory import RetrieverFactory

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

## Data Loaders (`extra.loaders`)

Utilities for loading and processing various document formats.

### Markdown Loader (`markdown_loader.py`)

Loads and parses markdown files with metadata preservation.

**Features:**
- Frontmatter extraction (YAML/TOML)
- Code block preservation
- Link extraction
- Metadata conversion to document attributes

**Usage:**
```python
from genai_tk.extra.loaders.markdown_loader import MarkdownLoader

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
from genai_tk.extra.loaders.mistral_ocr import MistralOCRLoader

loader = MistralOCRLoader(api_key="your-key")

# Extract from image
text = await loader.extract_text("image.png")

# Extract from PDF
pages = await loader.extract_from_pdf("document.pdf")
for page_num, text in enumerate(pages):
    print(f"Page {page_num}: {text}")
```

## Retrievers (`extra.retrievers`)

> See **[rag.md](rag.md)** for the full retriever documentation.

Low-level retriever implementations used by `RetrieverFactory`:

- `BM25FastRetriever` — bm25s-backed keyword retriever with optional Spacy preprocessing
- `ZeroEntropyRetriever` — read-only retriever backed by the ZeroEntropy SDK

## Utility Functions

### Anonymization (`custom_presidio_anonymizer.py`)

PII (Personally Identifiable Information) detection and anonymization.

**Features:**
- Named entity recognition for PII
- Multiple anonymization strategies
- Custom entity patterns
- Redaction reporting

**Usage:**

> **Requires:** `uv add presidio-analyzer presidio-anonymizer`

```python
from genai_tk.extra.custom_presidio_anonymizer import (
    PresidioAnonymizer,
    AnonymizationStrategy
)

anonymizer = PresidioAnonymizer()

# Detect PII
pii_list = anonymizer.analyze("John Smith's email is john@example.com")

# Anonymize
result = anonymizer.anonymize(
    "John Smith works at Acme Corp",
    strategy=AnonymizationStrategy.REDACT  # or REPLACE, HASH
)
```

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
from genai_tk.core.llm_factory import get_llm

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

**Supported Stores:**
- SQLite
- Redis
- In-Memory (dict)
- File-based

**Configuration:**
```yaml
kv_store:
  default: sqlite
  sqlite:
    path: ./data/cache.db
  redis:
    url: redis://localhost:6379
```

**Usage:**
```python
from genai_tk.extra.kv_store_registry import KvStoreRegistry

registry = KvStoreRegistry()
store = registry.get_store("sqlite")

# Set and get values
store.put("key", "value")
value = store.get("key")
```

### PGVector Factory (`pgvector_factory.py`)

PostgreSQL vector database integration for scalable embeddings.

**Features:**
- Vector similarity search
- Hybrid search (keyword + semantic)
- Composite indexes
- Connection pooling

**Configuration:**
```yaml
embeddings_store:
  type: pgvector
  pgvector:
    connection_string: postgresql://user:pass@localhost/db
    collection_name: documents
    distance_metric: cosine
```

**Usage:**
```python
from genai_tk.extra.pgvector_factory import PGVectorStore

store = PGVectorStore.from_config()

# Add embeddings
store.add_documents(documents)

# Search
results = store.similarity_search("query", k=5)
```

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

```python
from genai_tk.extra.custom_presidio_anonymizer import PresidioAnonymizer

# Step 1: Anonymize sensitive data
anonymizer = PresidioAnonymizer()
safe_text = anonymizer.anonymize(user_input)

# Step 2: Process safely
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": safe_text}]
})

# Step 3: De-anonymize if needed
final_result = anonymizer.deanonymize(result)
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
