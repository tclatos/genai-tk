# GenAI Toolkit (`genai-tk`)

A comprehensive toolkit for building AI applications with LangChain, LangGraph, and modern AI frameworks.

## Overview

GenAI Toolkit provides reusable components, agents, and utilities for building sophisticated AI applications. It focuses on:

- **Multi-Agent Workflows** - Build complex agent interactions with LangGraph
- **RAG (Retrieval Augmented Generation)** - Full RAG pipeline support with multiple vector stores and retrievers
- **Structured Output (BAML)** - Type-safe extraction from documents with guaranteed Pydantic results
- **Prefect Orchestration** - Parallel, manifest-tracked flows for document conversion, RAG ingestion, and batch extraction
- **Rich CLI** - All features accessible from the terminal; extensible with a single class and one YAML line
- **Framework-Specific Tools** - Extensive tool ecosystems for LangChain and SmolAgents
- **Type Safety** - Pydantic-based structured data handling with dynamic models
- **Enhanced Configuration** - Flexible, hierarchical config system with directory auto-discovery
- **Provider Agnostic** - Support for OpenAI, Anthropic, local models, and 100+ providers via LiteLLM
- **Developer Experience** - Works from any project directory; ephemeral Prefect server requires no external daemon
- **Modular Architecture** - Clean separation between core, extra, tools, and utilities
- **Data Processing** - Built-in OCR (Mistral batch), anonymization, and image analysis capabilities

## ✨ Recent Enhancements

- **Unified LangChain Agents** — `cli agents langchain` with YAML profiles for `react`, `deep`, and `custom` agent types
- **Docker Sandbox** — `cli sandbox start/stop/status/pull` for zero-overhead sandboxed execution
- **Browser Automation** — sandbox and direct (host Playwright) modes with credential hiding and SKILL.md support
- **Deer-flow Integration** — in-process ByteDance multi-agent system with skills, MCP, and web UI
- **BAML Structured Extraction** — `cli baml run / extract` with incremental manifests and KV-store caching ([docs/baml.md](docs/baml.md))
- **Prefect Flows** — ephemeral in-process runner; connect to a server with one env var ([docs/prefect.md](docs/prefect.md))
- **Extended CLI** — all commands documented with examples; one-class extension pattern ([docs/cli.md](docs/cli.md))

## Installation

**From git (recommended for projects using uv):**

```bash
# Core only
uv add git+https://github.com/tclatos/genai-tk@main

# Core + common extras (PostgreSQL, browser control, HuggingFace models)
uv add "genai-tk[extra] @ git+https://github.com/tclatos/genai-tk@main"

# Core + Deer-flow Python deps (still requires make deer-flow-install, see below)
uv add "genai-tk[deer-flow] @ git+https://github.com/tclatos/genai-tk@main"

# Everything
uv add "genai-tk[all] @ git+https://github.com/tclatos/genai-tk@main"
```

**Development installation (clone & edit):**

```bash
git clone https://github.com/tclatos/genai-tk.git
cd genai-tk
uv sync          # core + dev deps
uv sync --all-groups  # core + every dependency group
```

**Deer-flow setup** (requires a separate step — the backend is a separate repo):

```bash
# If you have the source repo cloned:
make deer-flow-install

# If you installed via uv add (no Makefile available):
uv run cli agents deerflow setup            # clones to ~/deer-flow by default
uv run cli agents deerflow setup --path ~/projects/deer-flow  # custom location

# Then add the printed line to your .env:
# DEER_FLOW_PATH=~/deer-flow
```

### Quick install test

```bash
# Sanity-check with the built-in fake model (no API key needed)
uv run cli core llm -i "tell me a joke" -m parrot_local@fake

# With a real model (OpenAI example)
uv run cli core llm -i "tell me a joke" -m gpt-4o-mini@openai --stream
```

## Quick Start

**Basic Setup**:
```python
from genai_tk.core import LLMFactory, EmbeddingsFactory
from genai_tk.utils.config_mngr import global_config

# Configuration works from any directory - no setup needed!
config = global_config()

# Create LLM using configured models
llm = LLMFactory.create()  # Uses default from config
# Or specify a model
llm = LLMFactory.create("gpt_4_openai")

# Create embeddings
embeddings = EmbeddingsFactory.create()  # Uses default
```

**Agent Example**:
```python
from genai_tk.extra.graphs import CustomReactAgent
from genai_tk.core import LLMFactory

# Create a ReAct agent
llm = LLMFactory.create("gpt_4_openai")
agent = CustomReactAgent(llm=llm)

# Run from anywhere - config auto-discovered!
result = agent.invoke("What's the weather like today?")
print(result)
```

**CLI Usage**:
```bash
# Show all command groups
uv run cli

# Invoke an LLM directly
uv run cli core llm -i "Tell me a joke" --stream

# Show config, API key status, default models
uv run cli info config

# List all configured agent profiles
uv run cli agents langchain --list

# Single-shot query / interactive chat
uv run cli agents langchain "What's the weather like today?"
uv run cli agents langchain -p Coding --chat

# RAG operations
uv run cli rag ingest my_data/
uv run cli rag query "What are the main features?"

# Structured extraction (BAML)
uv run cli baml run ExtractResume -i "John Smith; SW engineer"
uv run cli baml extract ./docs ./output --recursive --function ExtractSummary

# Document conversion (runs via Prefect, no server needed)
uv run cli tools markdownize ./pdfs ./output --recursive --mistral-ocr
```

See [docs/cli.md](docs/cli.md) for the full command reference.

## LLM Selection

Models are declared in `config/basic/providers/llm.yaml` using the format `model_id@provider`:

```yaml
# config/basic/providers/llm.yaml
llm:
  exceptions:
    - model_id: gpt4o
      providers:
        - openai: gpt-4o
    - model_id: haiku
      providers:
        - openrouter: anthropic/claude-haiku-4-5
```

The default model is configured in your active YAML config (e.g. `config/basic/extra/llm_defaults.yaml`). Override it at runtime with `-m` / `--llm`:

```bash
# Use default model from config
uv run cli core llm -i "Explain transformers"

# Use a specific model by ID@provider
uv run cli core llm -i "Explain transformers" -m gpt-4o@openai

# Use a named tag (configured in llm.models in YAML)
uv run cli core llm -i "Explain transformers" -m cheap_model

# See available models, tags, and API key status
uv run cli info config
```

**Structured Output (BAML)**:
```python
from genai_tk.extra.structured.baml_processor import BamlStructuredProcessor
from baml_client.types import Resume   # generated by baml-cli

processor = BamlStructuredProcessor[Resume](
    function_name="ExtractResume",
    llm="gpt_4o@openai",
)
results = asyncio.run(
    processor.abatch_analyze_documents(doc_ids, markdown_texts)
)
```

See [docs/baml.md](docs/baml.md) for setup, CLI, and the full API.

**Prefect Flows**:
```python
from genai_tk.extra.prefect.runtime import run_flow_ephemeral
from genai_tk.extra.markdownize_prefect_flow import markdownize_flow

# Runs in-process — no Prefect server required
run_flow_ephemeral(markdownize_flow, source_dir="./pdfs", output_dir="./md")
```

Set `GENAI_PREFECT_API_URL` or `prefect.api_url` in config to use a deployed server.
See [docs/prefect.md](docs/prefect.md) for all flows and how to write your own.

**Configuration Management**:
```python
from genai_tk.utils.config_mngr import global_config

# Works from notebooks/, demos/, or any subdirectory
config = global_config()

# Get configuration values
default_model = config.get('llm.models.default')
project_path = config.get('paths.project')

# Switch environments
config.select_config('production')
```

## Documentation

| Document | What it covers |
|----------|---------------|
| [docs/cli.md](docs/cli.md) | **All CLI commands**, flags, adding new commands — start here |
| [docs/core.md](docs/core.md) | LLM Factory, Embeddings, Vector Stores, Caching, Configuration |
| [docs/agents.md](docs/agents.md) | LangChain agents (react/deep/custom), profiles, middleware, checkpointing |
| [docs/extra.md](docs/extra.md) | RAG pipelines, graphs, data loaders, retrievers, privacy tools |
| [docs/baml.md](docs/baml.md) | **Structured output with BAML** — setup, `baml run/extract`, Python API |
| [docs/prefect.md](docs/prefect.md) | **Prefect flows** — ephemeral/server modes, all available flows, writing new ones |
| [docs/mcp-servers.md](docs/mcp-servers.md) | Expose tools/agents as Model Context Protocol servers |
| [docs/deer-flow.md](docs/deer-flow.md) | Deer-flow (ByteDance) integration — CLI, profiles, skills, embedded client |
| [docs/sandbox_support.md](docs/sandbox_support.md) | Docker sandbox setup, `cli sandbox` commands, configuration |
| [docs/browser_control.md](docs/browser_control.md) | Browser automation — sandbox vs direct mode, credentials, skills |
| [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md) | Unit/integration test patterns, fixtures, mocking |

Design and investigation notes are in [`docs/design/`](docs/design/).

## Package Structure

```
genai_tk/
├── agents/                  # Agent implementations
│   ├── langchain/          # Unified LangChain agent (react | deep | custom)
│   │   ├── config.py       # Pydantic config models + loader
│   │   ├── factory.py      # Unified agent factory
│   │   ├── agent.py        # Shell & direct runner
│   │   ├── commands.py     # CLI command registration
│   │   └── middleware/     # Middleware pipeline
│   └── smolagents/         # SmolAgents implementation
├── core/                    # Core AI components [See docs/core.md]
│   ├── llm_factory.py      # LLM creation and management
│   ├── embeddings_factory.py # Embeddings models
│   ├── embeddings_store.py # Vector databases
│   ├── cache.py            # Caching utilities
│   ├── chain_registry.py   # Chain registration system
│   ├── mcp_client.py       # Model Context Protocol client
│   ├── models_db.py        # Model registry and metadata
│   ├── providers.py        # Provider configuration
│   └── prompts.py          # Prompt utilities
├── extra/                   # Extended AI capabilities [See docs/extra.md]
│   ├── graphs/             # Agent graphs (ReAct, SQL, structured output)
│   ├── rag/                # RAG pipeline components
│   ├── loaders/            # Data loaders (OCR, markdown)
│   ├── retrievers/         # Retrieval systems (BM25, ensemble)
│   ├── custom_presidio_anonymizer.py # Data anonymization
│   ├── image_analysis.py   # Image processing and analysis
│   ├── gpt_researcher_helper.py # Research assistant
│   ├── kv_store_registry.py # Key-value store management
│   └── pgvector_factory.py # PostgreSQL vector DB
├── main/                    # CLI and command interface
│   ├── cli.py              # Main CLI entry point
│   ├── commands_agents.py  # Agent-related commands
│   ├── commands_core.py    # Core functionality commands
│   ├── commands_extra.py   # Extended feature commands
│   └── commands_rag.py     # RAG-specific commands
├── tools/                   # Framework-specific tools
│   ├── langchain/          # LangChain-compatible tools
│   │   ├── rag_tool_factory.py
│   │   ├── search_tools_factory.py
│   │   ├── sql_tool_factory.py
│   │   └── web_search_tool.py
│   └── smolagents/         # SmolAgents-compatible tools
│       ├── browser_use.py
│       ├── dataframe_tools.py
│       ├── sql_tools.py
│       └── yfinance_tools.py
├── utils/                   # Utilities and helpers
│   ├── cli/                # CLI utilities
│   │   ├── langchain_setup.py
│   │   ├── langgraph_agent_shell.py
│   │   └── smolagents_shell.py
│   ├── pydantic/           # Pydantic helpers
│   │   ├── dyn_model_factory.py
│   │   ├── kv_store.py
│   │   └── jsonl_store.py
│   ├── config_mngr.py      # Configuration management
│   ├── langgraph.py        # LangGraph utilities
│   └── ...
└── wip/                     # Work in progress

```

## Module Overview

### Core (`genai_tk.core`)

Foundation components for AI applications. See [Core Module Documentation](docs/core.md) for details.

**Key Components:**
- **LLM Factory** - Creates Language Models from 100+ providers via LiteLLM
- **Embeddings Factory** - Manages embedding models for semantic search
- **Embeddings Store** - Vector database backend (Chroma, Weaviate, PGVector, etc.)
- **Models DB** - Central registry of supported models with metadata
- **Cache** - Intelligent response caching (file, in-memory, Redis)
- **MCP Client** - Model Context Protocol integration for external tools
- **Configuration** - Hierarchical, auto-discovering configuration system
- **Providers** - Provider management and API key handling

**Quick Links:**
- [LLM Factory Configuration](docs/core.md#llm-factory) - Model selection and caching
- [Embeddings Setup](docs/core.md#embeddings-factory) - Semantic search configuration
- [Vector Databases](docs/core.md#embeddings-store) - RAG database backends
- [Configuration System](docs/core.md#configuration-files) - YAML-based configuration

### Agents (`genai_tk.agents`)

Agent implementations and orchestration. See [Agents Module Documentation](docs/agents.md) for details.

**LangChain Agents:**
- **Unified Configuration** - YAML-driven agent profiles (react, deep, custom types)
- **ReAct Agent** - Standard reasoning with tool use loops
- **Deep Agent** - Advanced planning and subagents (requires `deepagents` package)
- **Custom Agent** - Functional API-based agents with LangGraph
- **Middleware System** - Tool call logging, rate limiting, summarization
- **Checkpointing** - State persistence for multi-turn conversations
- **MCP Integration** - Load tools from Model Context Protocol servers

**SmolAgents:**
- Simple agent framework for quick prototyping
- Automatic tool composition
- Model-agnostic implementation

**Quick Links:**
- [Agent Configuration](docs/agents.md#configuration-system) - Profile-based setup
- [ReAct Agent](docs/agents.md#react-agent-default) - Standard reasoning agent
- [Deep Agent](docs/agents.md#deep-agent-advanced) - Advanced multi-step reasoning
- [CLI Usage](docs/agents.md#cli-interface) - Command-line operations
- [MCP Integration](docs/agents.md#mcp-servers-integration) - External tool servers

### Extra (`genai_tk.extra`)

Extended AI capabilities beyond core functionality. See [Extra Module Documentation](docs/extra.md) for details.

**Agent Graphs:**
- **ReAct Agent** - Custom implementation with max flexibility
- **SQL Agent** - Natural language database queries
- **Structured Output Agent** - Validated Pydantic model responses

**RAG Systems:**
- **Markdown Chunking** - Semantic document splitting
- **RAG Prefect Flow** - Orchestrated pipelines with scheduling
- **Dynamic Retrieval** - Query expansion and multi-step retrieval

**Data Loaders:**
- **Markdown Loader** - YAML frontmatter and metadata extraction
- **OCR Loader** - Image-to-text via Mistral

**Retrievers:**
- **BM25 Search** - Keyword-based retrieval
- **Ensemble Retriever** - Hybrid semantic + keyword search
- **Multi-Query Retriever** - Query expansion

**Privacy & Analysis:**
- **Presidio Anonymizer** - PII detection and masking
- **Image Analysis** - Computer vision understanding
- **GPT Researcher** - Multi-source synthesis

**Storage:**
- **PGVector** - PostgreSQL vector database
- **KV Store Registry** - Caching backends (SQLite, Redis, in-memory)

**Quick Links:**
- [Agent Graphs](docs/extra.md#agent-graphs) - Specialized agents
- [RAG Systems](docs/extra.md#rag-systems) - Retrieval pipelines
- [Data Loaders](docs/extra.md#data-loaders) - Document processing
- [Privacy Tools](docs/extra.md#utility-functions) - Data anonymization
- [Advanced Patterns](docs/extra.md#advanced-patterns) - Multi-step workflows

### Tools (`genai_tk.tools`)

Framework-specific tool implementations.

- **LangChain Tools** - RAG, search, SQL, web search tools
- **SmolAgents Tools** - Browser automation, data analysis, financial tools

### Utils (`genai_tk.utils`)

Shared utilities and helpers.

- **Configuration Manager** - Hierarchical config system with auto-discovery
- **CLI Utilities** - Shell interfaces for LangChain and SmolAgents
- **Pydantic Helpers** - Dynamic models, validation, and data stores
- **LangGraph Utilities** - LangGraph workflow utilities

## Supported AI Providers

Complete list of supported providers - see [Providers Documentation](docs/core.md#providers) for setup.

- **OpenAI** - GPT-4, GPT-4 Turbo, embeddings, vision - [Config example](docs/core.md#llm-factory)
- **Anthropic** - Claude models (via OpenRouter)
- **Local Models** - Ollama, VLLM, local inference
- **DeepSeek** - DeepSeek models and reasoning  
- **Mistral** - Mistral AI models and embeddings
- **Groq** - Fast inference endpoints
- **Google** - Gemini models and embeddings
- **Azure** - Azure OpenAI and Cognitive Services
- **LiteLLM** - Access to 100+ LLM providers unified API

## Agent Frameworks

### LangChain Agents - See [Comprehensive Documentation](docs/agents.md)

All LangChain-based agents are configured through a single `config/basic/agents/langchain.yaml` file.

**Three Agent Types:**
- **`react`** — Standard ReAct agent via LangChain `create_agent` - [Read more](docs/agents.md#react-agent-default)
- **`deep`** — Advanced multi-step agent via DeepAgents `create_deep_agent` with skills, planning, subagents - [Read more](docs/agents.md#deep-agent-advanced)
- **`custom`** — Functional ReAct agent built from scratch with LangGraph - [Read more](docs/agents.md#custom-agent)

**Quick CLI Usage:**
```bash
# List profiles
cli agents langchain --list

# Single-shot (default profile)
cli agents langchain "Research quantum computing trends"

# Select profile + interactive chat
cli agents langchain -p Coding --chat

# Override type and LLM without editing YAML
cli agents langchain -p Research --type react --llm gpt_41mini@openai "Quick answer"
```

**Complete documentation with examples:** [Agents Module Documentation](docs/agents.md#langchain-agents)

### Deer-flow Integration

GenAI Toolkit integrates with [Deer-flow](https://github.com/bytedance/deer-flow), ByteDance's LangGraph-based agent system.

See [Deer-flow Documentation](docs/Deer_Flow_Integration.md) for setup and usage.

**Quick Usage**:
```bash
cli agents deerflow --chat
cli agents deerflow "Research the latest AI developments"
```

## Configuration

**🚀 Enhanced Configuration System**

GenAI Toolkit features a **flexible, hierarchical configuration system** with these key improvements:

- **📁 Parent Directory Search**: Automatically finds configuration files by searching up the directory tree
- **🎯 Works from Any Directory**: Run commands from notebooks, subdirectories, or anywhere in your project
- **⚙️ Dynamic Path Resolution**: Paths automatically adjust based on project location
- **🔄 Environment Overrides**: Switch between development, testing, and production configurations
- **🔐 Optional Dependencies**: Graceful handling of missing optional dependencies

**Configuration Structure**:
```yaml
# config/app_conf.yaml - Main configuration
default_config: ${oc.env:BLUEPRINT_CONFIG,basic}

paths:
  project: ${oc.env:PWD}  # Auto-detected project root
  config: ${paths.project}/config
  data_root: ${oc.env:HOME}

# config/basic/providers/llm.yaml - LLM configurations  
llm:
  models:
    default: gpt_4_openai
    gpt_4_openai:
      provider: openai
      model: gpt-4-turbo
    gpt_4_groq:
      provider: groq
      model: llama-3.1-70b-versatile

# config/basic/providers/embeddings.yaml - Embedding configurations
embeddings:
  models:
    default: text_small_openai
    text_small_openai:
      provider: openai
      model: text-embedding-3-small

# config/basic/agents/langchain.yaml - Unified agent profiles
langchain_agents:
  defaults:
    type: react
    checkpointer: {type: none}
  default_profile: "Research"
  profiles:
    - name: "Research"
      type: deep
      llm: "gpt_41@openai"
```

**Environment Variables** (loaded from `.env` in project root or parents):
```bash
# API Keys
OPENAI_API_KEY=your-key-here
GROQ_API_KEY=your-groq-key
ANTHROPIC_API_KEY=your-anthropic-key

# Configuration Selection
BLUEPRINT_CONFIG=development  # Switch configuration environments
```

**Usage from Any Directory**:
```python
from genai_tk.utils.config_mngr import global_config

# Works from notebooks/, demos/, or any subdirectory!
config = global_config()
model_id = config.get('llm.models.default')
```

**Detailed Documentation:**
- [Core Module Configuration](docs/core.md#configuration-files) - All configuration options
- [LLM Provider Setup](docs/core.md#llm-factory) - Configuring language models
- [Embeddings Configuration](docs/core.md#embeddings-factory) - Vector embeddings setup
- [Agent Profiles](docs/agents.md#configuration-system) - Agent-specific configuration
- [Vector Databases](docs/core.md#embeddings-store) - RAG database configuration

## Development

```bash
# Install development dependencies
make install-dev

# Format code  
make fmt

# Run linting
make lint

# Run tests
make test

# Run all checks
make check
```

## Testing

```bash
# Unit tests only
make test-unit

# Integration tests only  
make test-integration

# Run specific test
uv run pytest tests/unit_tests/core/test_llm_factory.py::test_basic_call -v

# Run tests by pattern
uv run pytest tests/unit_tests/ -k "test_name_pattern" -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Run `make check` to ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related Projects

- **GenAI Blueprint** (`genai_bp`) - Application framework built on genai-tk
- **LangChain** - Core LLM application framework
- **LangGraph** - Multi-agent workflow engine

## Support & Documentation

**Main Documentation:**
- [Core Module Documentation](docs/core.md) - LLM Factory, Embeddings, Configuration, Caching
- [Agents Module Documentation](docs/agents.md) - Agent Implementation, Profiles, Configuration
- [Extra Module Documentation](docs/extra.md) - Advanced Graphs, RAG, Data Loaders, Retrievers
- [Development Guidelines](AGENTS.md) - Code style, testing, architecture

**Integration & Advanced Topics:**
- [MCP Servers Integration](docs/mcp-servers.md) - Model Context Protocol
- [Deer-flow Integration](docs/Deer_Flow_Integration.md) - Advanced agent framework
- [Testing Guide](docs/TESTING_GUIDE.md) - Testing patterns and best practices
- [Sandbox Support](docs/sandbox_support.md) - Sandboxed code execution
- [Browser Control](docs/browser_control.md) - Web automation

**Get Help:**
- Issues: [GitHub Issues](https://github.com/tclatos/genai-tk/issues)
- Discussions: [GitHub Discussions](https://github.com/tclatos/genai-tk/discussions)