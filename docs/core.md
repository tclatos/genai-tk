# Core Module (`genai_tk.core`)

## Overview

The `core` module provides foundational components for building AI applications. It handles Language Models, Embeddings, vector databases, caching, and provider management. All components are designed around configuration-driven initialization and support multiple AI providers.

## Key Components

### LLM Factory (`llm_factory.py`)

**Purpose:** Factory pattern for creating and managing Language Learning Models from various providers.

**Features:**
- Support for 100+ LLM providers through LiteLLM and direct integrations
- Model identification via `model_id@provider` format (e.g., `gpt_4o@openai`)
- Runtime model switching and fallback mechanisms
- JSON output mode for structured responses
- Integrated caching and streaming capabilities
- Configuration through `models_providers.yaml` and environment variables

**Configuration:**
```yaml
llm:
  models:
    default: gpt_4o@openai
    fast: gpt_35_turbo@openai
    local: mistral@ollama
  cache:
    method: default  # or "in-memory", "aiolite"
    ttl: 3600
```

**Usage:**
```python
from genai_tk.core.llm_factory import get_llm

# Get default LLM from config
llm = get_llm()

# Get specific model
llm = get_llm(llm="gpt_4o@openai")

# Use with JSON mode for structured output
llm = get_llm(llm="gpt_4o@openai", json_mode=True)

# Stream responses
llm = get_llm()
for chunk in llm.stream("Tell me about AI"):
    print(chunk.content, end="", flush=True)
```

**Related:**
- `models_db.py` - Model registry and metadata storage
- `providers.py` - Provider configuration and API key management
- `cache.py` - Response caching system

### Embeddings Factory (`embeddings_factory.py`)

**Purpose:** Factory for creating and managing embedding models for semantic search and vector operations.

**Features:**
- Support for multiple embedding providers (OpenAI, HuggingFace, Ollama, etc.)
- Configurable dimension and batch size
- Automatic caching with `CacheBackedEmbeddings`
- Flexible initialization from YAML configuration
- KV store integration for persistence

**Configuration:**
```yaml
embeddings:
  default: text_embedding_3_small@openai
  models:
    dense: text_embedding_ada_002@openai
    sparse: bge_small@huggingface
  cache:
    store: sqlite  # or redis, file
```

**Usage:**
```python
from genai_tk.core.embeddings_factory import get_embeddings, EmbeddingsFactory

# Get default embeddings
embeddings = get_embeddings()

# Get specific model
embeddings = get_embeddings(embeddings="text_embedding_3_small@openai")

# Embed documents
vectors = embeddings.embed_documents(["Document 1", "Document 2"])

# Embed query
query_vector = embeddings.embed_query("search term")
```

**Related:**
- `embeddings_store.py` - Vector database management

### Embeddings Store (`embeddings_store.py`)

**Purpose:** Manages vector databases and embeddings storage for RAG applications.

**Features:**
- Multi-backend support (Chroma, Weaviate, PGVector, etc.)
- Dynamic store creation from configuration
- Persistent and temporary store modes
- Serialization for checkpointing

**Configuration:**
```yaml
embeddings_store:
  type: chroma  # chroma, weaviate, pgvector, qdrant
  chroma:
    persist_directory: ./data/chroma
    is_persistent: true
  pgvector:
    connection_string: postgresql://user:pass@localhost/db
```

**Usage:**
```python
from genai_tk.core.embeddings_store import EmbeddingsStore

# Create store from config
store = EmbeddingsStore.from_config()

# Or specify type directly
store = EmbeddingsStore.create("chroma", persist_dir="./data")

# Add documents
store.add_documents(documents)

# Search
results = store.similarity_search("query", k=5)
```

### Cache System (`cache.py`)

**Purpose:** Intelligent caching for LLM responses to reduce API costs and improve performance.

**Features:**
- Multiple cache backends (in-memory, SQLite, Redis)
- Automatic key generation from prompts
- TTL support for expiring old entries
- Optional semantic caching

**Cache Methods:**
- `default` - File-based cache
- `in_memory` - Fast in-memory cache
- `aiolite` - Async SQLite implementation
- `none` - Disable caching

**Configuration:**
```yaml
llm:
  cache:
    method: default
    ttl: 3600  # 1 hour expiration
```

**Usage:**
```python
from genai_tk.core.cache import LlmCache, CacheMethod

# Set cache method globally
LlmCache.set_method("default")

# Create cached LLM
llm = get_llm(cache=CacheMethod.DEFAULT)

# Control caching per call
response = llm.invoke("prompt", cache=False)
```

### Chain Registry (`chain_registry.py`)

**Purpose:** Central registry for discovering and managing LangChain chains and runnables.

**Features:**
- Dynamic chain registration and discovery
- Named chain instances for reuse
- Configuration-driven chain initialization
- Support for custom chain builders

**Usage:**
```python
from genai_tk.core.chain_registry import ChainRegistry

# Register a chain
registry = ChainRegistry()
registry.register("my_chain", my_chain_instance)

# Retrieve a chain
chain = registry.get("my_chain")

# List all chains
chains = registry.list_chains()
```

### MCP Client (`mcp_client.py`)

**Purpose:** Model Context Protocol (MCP) client for integrating external tools via standard MCP servers.

**Features:**
- Multi-server MCP client management
- Tool discovery and dynamic loading
- Prompt template integration
- Server lifecycle management

**Configuration:**
```yaml
mcp_servers:
  - name: math_server
    command: python
    args: ["math_server.py"]
  - name: weather_server
    command: python
    args: ["weather_server.py"]
```

**Usage:**
```python
from genai_tk.core.mcp_client import get_mcp_tools_info, get_mcp_servers_dict

# Get info about all available tools
tools_info = await get_mcp_tools_info()

# Get server configurations
servers = get_mcp_servers_dict(filter=["math_server"])

# Use in agent
mcp_tools = await mcp_client.get_tools()
agent.tools.extend(mcp_tools)
```

**Related:** See [MCP Servers Documentation](../docs/mcp-servers.md)

### Prompts (`prompts.py`)

**Purpose:** Utilities for constructing and managing prompts with consistent formatting.

**Features:**
- Dedenting and whitespace handling
- Dictionary and list message formats
- System and user message templates
- Integration with LangChain prompt templates

**Usage:**
```python
from genai_tk.core.prompts import def_prompt, dedent_ws, dict_input_message

# Create prompt with system and user messages
prompt = def_prompt(
    system="You are a helpful AI assistant",
    user="Explain quantum computing"
)

# Format messages as dictionary
messages = dict_input_message(
    user="What is machine learning?",
    system="You are an educator"
)

# Dedent multi-line strings
text = dedent_ws("""
    This is a
    multi-line string
""")
```

### Models Database (`models_db.py`)

**Purpose:** Central database of supported LLM and embedding models with metadata.

**Features:**
- Model registry with cost and capability information
- Provider mapping for model discovery
- Context window and token limit tracking
- Dynamic model loading from JSON

**Usage:**
```python
from genai_tk.core.models_db import get_models_db

# Get model database
db = get_models_db()

# Query model info
model = db.get_model("gpt_4o@openai")
print(f"Context window: {model.context_window}")
print(f"Cost: {model.cost_per_1k}")

# List all models for a provider
openai_models = db.list_models_for_provider("openai")
```

### Providers (`providers.py`)

**Purpose:** Provider configuration, API key management, and LLM class resolution.

**Features:**
- Secure API key loading from environment
- Provider information registry
- Dynamic class loading from module paths
- Support for custom providers

**Configuration:**
```yaml
providers:
  openai:
    api_key_env: OPENAI_API_KEY
    base_url: https://api.openai.com/v1
  custom:
    api_key_env: CUSTOM_API_KEY
    base_url: http://custom-server:8000
```

**Usage:**
```python
from genai_tk.core.providers import (
    get_provider_info,
    get_provider_api_key,
    get_provider_api_env_var
)

# Get provider info
info = get_provider_info("openai")
print(f"Module: {info.module}")
print(f"Class: {info.langchain_class}")

# Get API key
api_key = get_provider_api_key("openai")

# Get env var name
env_var = get_provider_api_env_var("openai")
```

## Configuration

Configuration for LLM models, embeddings, and caching lives in `config/providers/` and `config/baseline.yaml`.
See [docs/configuration.md](configuration.md) for the full system reference.

## Common Patterns

### Pattern 1: Multi-Model Comparison
```python
from genai_tk.core.llm_factory import get_llm

models = ["gpt_4o@openai", "claude_3@anthropic", "mixtral@groq"]

for model_id in models:
    llm = get_llm(llm=model_id)
    response = llm.invoke("Your prompt here")
    print(f"{model_id}: {response.content}")
```

### Pattern 2: Structured Output with Caching
```python
from genai_tk.core.llm_factory import get_llm
from pydantic import BaseModel

class Result(BaseModel):
    answer: str
    confidence: float

# Use JSON mode with caching
llm = get_llm(json_mode=True)
response = llm.with_structured_output(Result).invoke("prompt")
```

### Pattern 3: RAG with Embeddings
```python
from genai_tk.core.embeddings_factory import get_embeddings
from genai_tk.core.embeddings_store import EmbeddingsStore

# Get embeddings and store
embeddings = get_embeddings()
store = EmbeddingsStore.from_config()

# Add documents
store.add_documents(documents)

# Retrieve similar documents
context = store.similarity_search("user query", k=5)
```

## Error Handling

Core components handle optional dependencies gracefully:

```python
from genai_tk.core.embeddings_store import EmbeddingsStore

try:
    # May fail if pgvector not installed
    store = EmbeddingsStore.create("pgvector", ...)
except ImportError as e:
    logger.warning(f"PGVector not available: {e}")
    # Fall back to alternative
    store = EmbeddingsStore.create("chroma")
```

## See Also

- [Agents Documentation](agents.md) - Agent implementations using core components
- [Extra Documentation](extra.md) - Extended AI capabilities and RAG
- [Configuration Reference](configuration.md) - Full config system reference
- [LLM Selection](llm-selection.md) - Model IDs, tags, models.dev
- [MCP Servers](mcp-servers.md) - Model Context Protocol integration
- [Testing Guide](TESTING_GUIDE.md) - Testing patterns
