# GenAI Toolkit (`genai-tk`)

A comprehensive toolkit for building AI applications with LangChain, LangGraph, and modern AI frameworks.

## Overview

GenAI Toolkit provides reusable components, agents, and utilities for building sophisticated AI applications. It focuses on:

- **Multi-Agent Workflows** - Build complex agent interactions with LangGraph
- **RAG (Retrieval Augmented Generation)** - Full RAG pipeline support with multiple vector stores and retrievers
- **Framework-Specific Tools** - Extensive tool ecosystems for LangChain and SmolAgents
- **Type Safety** - Pydantic-based structured data handling with dynamic models
- **Enhanced Configuration** - Flexible, hierarchical config system with directory auto-discovery
- **Provider Agnostic** - Support for OpenAI, Anthropic, local models, and 100+ providers via LiteLLM
- **Robust Error Handling** - Graceful handling of optional dependencies and missing configurations
- **Developer Experience** - Works from any project directory, comprehensive CLI with framework-specific shells
- **Modular Architecture** - Clean separation between core, extra, tools, and utilities
- **Data Processing** - Built-in OCR, anonymization, and image analysis capabilities

## ✨ Recent Enhancements

- **📍 Flexible Configuration Discovery**: Automatically finds config files by searching parent directories
- **📁 Work from Anywhere**: Run commands from notebooks, subdirectories, or any project location
- **⚙️ Dynamic Path Resolution**: Paths automatically adjust based on project structure
- **🔄 Environment Switching**: Easy switching between development, testing, production configurations
- **🔐 Optional Dependencies**: Graceful handling of missing packages (e.g., `langchain_postgres`)
- **🛠️ Enhanced CLI**: Comprehensive command-line interface with framework-specific tools
- **🧪 Modular Testing**: Organized unit and integration test structure
- **🔧 Framework-Specific Tools**: Separate tool ecosystems for LangChain and SmolAgents

## Installation

```bash
# Install with core dependencies
uv pip install git+https://github.com/tclatos/genai-tk@main

# Install with all extras  
uv pip install "genai-tk[extra] @ git+https://github.com/tclatos/genai-tk@main"

# Development installation
git clone https://github.com/tclatos/genai-tk.git
cd genai-tk
uv sync
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
# Run agents from command line
genai-tk agents react "What's the weather like today?"

# Use framework-specific shells
genai-tk shell langchain
genai-tk shell smolagents

# RAG operations
genai-tk rag create-index my_data/
genai-tk rag query "What are the main features?"
```

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

## Package Structure

```
genai_tk/
├── core/                    # Core AI components
│   ├── llm_factory.py      # LLM creation and management
│   ├── embeddings_factory.py # Embeddings models
│   ├── embeddings_store.py # Vector databases
│   ├── deep_agents.py      # LangChain-based agents
│   ├── cache.py            # Caching utilities
│   ├── chain_registry.py   # Chain registration system
│   ├── langgraph_runner.py # LangGraph execution engine
│   ├── mcp_client.py       # Model Context Protocol client
│   └── ...
├── extra/                   # Extended AI capabilities
│   ├── graphs/             # Agent graphs (ReAct, SQL, etc.)
│   ├── loaders/            # Data loaders (OCR, etc.)
│   ├── retrievers/         # Retrieval components (BM25, etc.)
│   ├── custom_presidio_anonymizer.py # Data anonymization
│   ├── gpt_researcher_helper.py # Research assistant
│   ├── image_analysis.py   # Image processing
│   └── ...
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
│   ├── crew_ai/            # CrewAI utilities
│   ├── pydantic/           # Pydantic helpers
│   │   ├── dyn_model_factory.py
│   │   ├── kv_store.py
│   │   └── jsonl_store.py
│   ├── config_mngr.py      # Configuration management
│   ├── langgraph.py        # LangGraph utilities
│   └── ...
└── wip/                     # Work in progress
    ├── autogen_utils.py
    ├── browser_use_langchain.py
    └── structured_output.py
```

## Key Components

### Core (`genai_tk.core`)

- **LLM Factory** - Creates Language Models from multiple providers
- **Embeddings Factory** - Provides embeddings for semantic search
- **Embeddings Store** - Vector database management for RAG
- **Deep Agents** - LangChain-based reasoning agents
- **MCP Client** - Model Context Protocol integration
- **Cache** - Intelligent caching system for AI responses
- **Chain Registry** - Centralized chain registration and discovery
- **LangGraph Runner** - Execution engine for LangGraph workflows

### Main (`genai_tk.main`)

- **CLI Interface** - Comprehensive command-line interface
- **Agent Commands** - Agent management and execution commands
- **Core Commands** - Core functionality commands
- **RAG Commands** - RAG pipeline management commands

### Extra (`genai_tk.extra`)

- **Agent Graphs** - ReAct, SQL, and structured output agents
- **Data Loaders** - OCR (Mistral), document processing
- **Retrievers** - BM25S and other retrieval components
- **Presidio Anonymizer** - Data privacy and anonymization
- **GPT Researcher Helper** - Research assistant integration
- **Image Analysis** - Image processing and analysis
- **KV Store Registry** - Key-value store management
- **PGVector Factory** - PostgreSQL vector database integration

### Tools (`genai_tk.tools`)

- **LangChain Tools** - RAG, search, SQL, and web search tools
- **SmolAgents Tools** - Browser automation, data analysis, financial tools

### Utils (`genai_tk.utils`)

- **Configuration Manager** - Hierarchical config system with auto-discovery
- **CLI Utilities** - Shell interfaces for LangChain and SmolAgents
- **CrewAI Utilities** - CrewAI integration helpers
- **Pydantic Helpers** - Dynamic models, validation, and data stores
- **LangGraph Utilities** - LangGraph workflow utilities
- **Rich Widgets** - Rich console components and displays

## Supported AI Providers

- **OpenAI** - GPT models, embeddings, tools
- **Anthropic** - Claude models (via OpenRouter)
- **Local Models** - Ollama, VLLM, local inference
- **DeepSeek** - DeepSeek models and reasoning  
- **Mistral** - Mistral AI models
- **Groq** - Fast inference endpoints
- **LiteLLM** - 100+ LLM providers unified API

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

# config/basic/agents/langchain.yaml - Agent configurations
agents:
  langchain:
    react:
      llm: ${llm.models.default}
      tools: []
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

## Support

- Documentation: [Agents.md](Agents.md)
- Issues: [GitHub Issues](https://github.com/tclatos/genai-tk/issues)
- Discussions: [GitHub Discussions](https://github.com/tclatos/genai-tk/discussions)