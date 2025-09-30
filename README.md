# GenAI Toolkit (`genai-tk`)

A comprehensive toolkit for building AI applications with LangChain, LangGraph, and modern AI frameworks.

## Overview

GenAI Toolkit provides reusable components, agents, and utilities for building sophisticated AI applications. It focuses on:

- **Multi-Agent Workflows** - Build complex agent interactions
- **RAG (Retrieval Augmented Generation)** - Full RAG pipeline support  
- **Tool Integration** - Extensive tool ecosystem for agents
- **Type Safety** - Pydantic-based structured data handling
- **Configuration Management** - Flexible, hierarchical configuration
- **Provider Agnostic** - Support for OpenAI, Anthropic, local models, and more

## Installation

```bash
# Install with core dependencies
uv pip install git+https://github.com/tcaminel-pro/genai-tk@main

# Install with all extras  
uv pip install "genai-tk[extra] @ git+https://github.com/tcaminel-pro/genai-tk@main"

# Development installation
git clone https://github.com/tcaminel-pro/genai-tk.git
cd genai-tk
uv sync
```

## Quick Start

```python
from genai_tk.core import LLMFactory, EmbeddingsFactory
from genai_tk.extra.graphs import CustomReactAgent
from genai_tk.utils import ConfigManager

# Create LLM and embeddings
llm = LLMFactory.create("openai/gpt-4")
embeddings = EmbeddingsFactory.create("openai/text-embedding-3-small")

# Create a ReAct agent
agent = CustomReactAgent(llm=llm)
result = agent.invoke("What's the weather like today?")
```

## Package Structure

```
genai_tk/
├── core/                    # Core AI components
│   ├── llm_factory.py      # LLM creation and management
│   ├── embeddings_factory.py # Embeddings models
│   ├── vector_store_factory.py # Vector databases
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

Configuration is managed through YAML files with environment variable support:

```yaml
# config/llm.yaml
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

Environment variables are loaded from `.env`:

```bash
OPENAI_API_KEY=your-key-here
GROQ_API_KEY=your-groq-key
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
- Issues: [GitHub Issues](https://github.com/tcaminel-pro/genai-tk/issues)
- Discussions: [GitHub Discussions](https://github.com/tcaminel-pro/genai-tk/discussions)