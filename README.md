# GenAI Toolkit (`genai-tk`)

A comprehensive toolkit for building AI applications with LangChain, LangGraph, and modern AI frameworks.

## Overview

GenAI Toolkit provides reusable components, agents, and utilities for building sophisticated AI applications. It focuses on:

- **Multi-Agent Workflows** - Build complex agent interactions with LangGraph
- **RAG (Retrieval Augmented Generation)** - Full RAG pipeline support with multiple vector stores
- **Tool Integration** - Extensive tool ecosystem for agents (LangChain & SmolAgents compatible)
- **Type Safety** - Pydantic-based structured data handling with dynamic models
- **Enhanced Configuration** - Flexible, hierarchical config system with directory auto-discovery
- **Provider Agnostic** - Support for OpenAI, Anthropic, local models, and 100+ providers via LiteLLM
- **Robust Error Handling** - Graceful handling of optional dependencies and missing configurations
- **Developer Experience** - Works from any project directory, comprehensive CLI tools

## ✨ Recent Enhancements

- **📍 Flexible Configuration Discovery**: Automatically finds config files by searching parent directories
- **📁 Work from Anywhere**: Run commands from notebooks, subdirectories, or any project location
- **⚙️ Dynamic Path Resolution**: Paths automatically adjust based on project structure
- **🔄 Environment Switching**: Easy switching between development, testing, production configurations
- **🔐 Optional Dependencies**: Graceful handling of missing packages (e.g., `langchain_postgres`)
- **🛠️ Improved CLI**: Enhanced command-line tools with better error handling and help

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
│   └── ...
├── extra/                   # Extended AI capabilities
│   ├── graphs/             # Agent graphs (ReAct, SQL, etc.)
│   ├── tools_langchain/    # LangChain tools
│   ├── tools_smolagents/   # SmolAgents tools
│   ├── chains/             # Reusable AI chains
│   └── ...
└── utils/                   # Utilities and helpers
    ├── config_mngr.py      # Configuration management
    ├── streamlit/          # Streamlit components  
    ├── cli/                # CLI utilities
    └── ...
```

## Key Components

### Core (`genai_tk.core`)

- **LLM Factory** - Creates Language Models from multiple providers
- **Embeddings Factory** - Provides embeddings for semantic search
- **Vector Store Factory** - Creates vector databases for RAG
- **Deep Agents** - LangChain-based reasoning agents
- **MCP Client** - Model Context Protocol integration

### Extra (`genai_tk.extra`)

- **React Agents** - ReAct pattern implementation
- **SQL Agent** - Database querying specialist
- **Research Tools** - GPT Researcher integration
- **Browser Automation** - Web scraping and interaction
- **Tool Collections** - LangChain and SmolAgent tools

### Utils (`genai_tk.utils`)

- **Configuration Manager** - Hierarchical config system
- **Streamlit Components** - Chat interfaces and widgets
- **CLI Utilities** - Command-line tools and shells  
- **Pydantic Helpers** - Dynamic models and validation

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
default_config: ${oc.env:BLUEPRINT_CONFIG,ekg_local}

paths:
  project: ${oc.env:PWD}  # Auto-detected project root
  config: ${paths.project}/config
  data_root: ${oc.env:HOME}

# config/providers/llm.yaml - LLM configurations  
llm:
  - id: gpt_4
    providers:
      - openai: gpt-4-turbo
      - groq: llama-3.1-70b-versatile

embeddings:
  - id: text_small  
    providers:
      - openai: text-embedding-3-small
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

# Test package installation
make test-install
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