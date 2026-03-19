# Extra Module (`genai_tk.extra`)

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
from genai_tk.core import LLMFactory
from langgraph.checkpoint.memory import MemorySaver

# Create agent
llm = LLMFactory.create("gpt_4o@openai")
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
from genai_tk.extra.graphs.sql_agent import create_sql_agent
from langchain_community.utilities import SQLDatabase

# Create agent for database
db = SQLDatabase.from_uri("sqlite:///./data.db")
agent = create_sql_agent(llm=get_llm(), db=db)

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
    create_react_agent_structured_output
)

class ResearchResult(BaseModel):
    """Research findings."""
    title: str
    key_points: list[str]
    sources: list[str]
    confidence: float

agent = create_react_agent_structured_output(
    llm=get_llm(),
    tools=tools,
    output_schema=ResearchResult
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Research AI trends"}]
})

# result.output is validated ResearchResult instance
print(result.output.title)
print(result.output.key_points)
```

## RAG Systems (`extra.rag`)

Retrieval-Augmented Generation pipeline components.

### RAG Commands (`commands_rag.py`)

CLI commands for RAG operations.

```bash
# Create vector index from documents
cli rag create-index ./my_documents/

# Query the vector index
cli rag query "What are the main features?"

# Add documents to existing index
cli rag add-documents ./new_docs/

# List available indexes
cli rag list-indexes
```

### Markdown Chunking (`markdown_chunking.py`)

Intelligent document chunking that respects markdown structure.

**Features:**
- Markdown-aware splitting (respects headers, sections)
- Configurable chunk size and overlap
- Preserves semantic structure
- Metadata extraction from headers

**Usage:**
```python
from genai_tk.extra.rag.markdown_chunking import MarkdownChunker

chunker = MarkdownChunker(
    chunk_size=1024,
    chunk_overlap=200,
    preserve_headers=True
)

with open("document.md") as f:
    documents = chunker.split_text(f.read())

# documents have metadata about hierarchy
for doc in documents:
    print(f"Section: {doc.metadata.get('section')}")
    print(f"Level: {doc.metadata.get('heading_level')}")
```

### RAG Prefect Flow (`rag_prefect_flow.py`)

Orchestrated RAG pipeline using Prefect for scheduling and monitoring.

**Features:**
- Document loading and processing
- Embedding generation
- Vector database updates
- Scheduled indexing

**Usage:**
```python
from genai_tk.extra.rag.rag_prefect_flow import rag_pipeline

# Run RAG pipeline
result = rag_pipeline.run(
    docs_path="/path/to/documents",
    index_name="my_documents",
    chunk_size=1024,
    model="gpt_4o@openai"
)
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

Search and retrieval implementations for RAG systems.

**Available Retrievers:**
- `BM25Retriever` - Keyword-based search
- `EnsembleRetriever` - Combines multiple retrievers
- `MultiQueryRetriever` - Query expansion for better results
- `VectorStoreRetriever` - Semantic search with embeddings

**Usage Example:**
```python
from langchain.retrievers import BM25Retriever
from genai_tk.core.embeddings_store import EmbeddingsStore

# Vector-based retrieval
vector_store = EmbeddingsStore.from_config()
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# BM25 (keyword) retrieval
bm25_retriever = BM25Retriever.from_documents(documents)

# Ensemble for best results
from langchain.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

# Use in RAG chain
results = ensemble_retriever.get_relevant_documents("query")
```

## Utility Functions

### Anonymization (`custom_presidio_anonymizer.py`)

PII (Personally Identifiable Information) detection and anonymization.

**Features:**
- Named entity recognition for PII
- Multiple anonymization strategies
- Custom entity patterns
- Redaction reporting

**Usage:**
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
from genai_tk.extra.image_analysis import analyze_image

# Single image analysis
analysis = await analyze_image(
    "image.jpg",
    query="What objects are in this image?",
    llm=get_llm()
)

# Multiple images comparison
comparison = await compare_images(
    ["image1.jpg", "image2.jpg"],
    query="Highlight differences",
    llm=get_llm()
)
```

### GPT Researcher Helper (`gpt_researcher_helper.py`)

Integration with GPT Researcher for autonomous research tasks.

**Features:**
- Research report generation
- Source aggregation
- Citation management
- Multi-source synthesis

**Usage:**
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
    create_react_agent_structured_output
)

class DocumentSummary(BaseModel):
    title: str
    key_insights: list[str]
    action_items: list[str]
    sentiment: str

# RAG + Structured output
agent = create_react_agent_structured_output(
    llm=get_llm(),
    tools=[rag_retriever],
    output_schema=DocumentSummary
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
