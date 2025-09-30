# Changelog

All notable changes to genai_tk will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-09-30

### Added
- **Initial release** - GenAI Toolkit extracted from genai-blueprint
- **Core AI Components** 
  - LLM Factory with multi-provider support (OpenAI, Anthropic, local models)
  - Embeddings Factory for semantic search capabilities
  - Vector Store Factory for RAG applications
  - Deep Agents with LangChain integration
  - MCP Client for Model Context Protocol
  - Chain Registry for reusable AI processing chains
  - Structured Output handling with Pydantic
  - Prompts collection and templates
  - Caching layer for expensive operations

- **Extra AI Capabilities**
  - ReAct Agent implementation
  - SQL Agent for database querying
  - GPT Researcher integration
  - Image analysis capabilities
  - Custom Presidio anonymization
  - LangChain Tools collection (web search, multi-search, SQL tools, config loader)
  - SmolAgents Tools (browser automation, DataFrame tools, SQL tools, YFinance integration)
  - BM25 retriever implementation
  - Mistral OCR document loader
  - PostgreSQL vector database factory
  - Knowledge Graph utilities (Cognee, Kuzu integration)

- **Utilities and Helpers**
  - Configuration management with hierarchical YAML support
  - Logger factory with structured logging
  - Streamlit components (callback handlers, auto-scroll, chat interfaces)
  - CLI utilities (LangChain setup, agent shells, config display)
  - Pydantic utilities (dynamic models, KV stores, field manipulation)
  - Data processing helpers (collection helpers, data loading, SQL utils)
  - CrewAI integration utilities
  - Spacy model management

- **Testing and Quality**
  - Comprehensive unit test suite covering all core components
  - Integration tests for complex workflows
  - GitHub Actions CI/CD pipeline
  - Ruff formatting and linting
  - Python 3.12 support with modern type annotations
  - uv package manager integration

- **Development Experience**
  - Modern Python 3.12 project structure
  - Absolute imports throughout (no relative imports)
  - Comprehensive documentation with Agents.md
  - Makefile with common development tasks
  - Type safety with beartype runtime checking
  - Proper package discovery and installation

### Technical Details
- **Python Version**: Requires Python >=3.12,<3.13
- **Package Manager**: Built for uv with dependency groups
- **Build System**: Uses hatchling with proper package configuration
- **Dependencies**: Modular dependency groups (core, extra, dev)
- **Import Style**: Absolute imports only (`from genai_tk.core import ...`)
- **Type System**: Modern Python 3.12 type annotations (`|` syntax, `| None`)

### Migration Notes
This is the initial standalone release of the GenAI Toolkit, extracted from the monolithic genai-blueprint project. All imports have been updated from `src.ai_core` → `genai_tk.core`, `src.ai_extra` → `genai_tk.extra`, and `src.utils` → `genai_tk.utils`.

### Installation
```bash
# Install core toolkit
uv pip install git+https://github.com/tcaminel-pro/genai-tk@main

# Install with all extras
uv pip install "genai_tk[extra] @ git+https://github.com/tcaminel-pro/genai-tk@main"
```

### Supported Providers
- OpenAI (GPT models, embeddings, tools)
- Anthropic (Claude models via OpenRouter)
- Local models (Ollama, VLLM)
- DeepSeek (reasoning models)
- Mistral AI (models and embeddings)
- Groq (fast inference)
- LiteLLM (100+ providers unified)
- And many more through LangChain integrations