# GenAI Toolkit Agents and Components

This document outlines the core agents, components, and capabilities provided by the GenAI Toolkit (`genai-tk`).

## Core Components (`genai_tk.core`)

### LLM and Embeddings Infrastructure
- **LLM Factory** (`llm_factory.py`) - Creates and configures Language Models from multiple providers
- **Embeddings Factory** (`embeddings_factory.py`) - Provides embeddings models for semantic search
- **Vector Store Factory** (`vector_store_registry.py`) - Creates vector databases for RAG applications
- **Providers** (`providers.py`) - Unified interface for AI model providers (OpenAI, Anthropic, local models, etc.)

### Agent Infrastructure
- **Deep Agents** (`deep_agents.py`) - LangChain-based deep reasoning agents
- **LangGraph Runner** (`langgraph_runner.py`) - Executes complex multi-step agent workflows
- **MCP Client** (`mcp_client.py`) - Model Context Protocol client for tool integration
- **Chain Registry** (`chain_registry.py`) - Registry for reusable AI processing chains

### Core Utilities
- **Structured Output** (`structured_output.py`) - Handles structured data extraction from LLMs
- **Prompts** (`prompts.py`) - Collection of optimized prompts and prompt templates
- **Cache** (`cache.py`) - Caching layer for expensive AI operations
- **CLI Commands** (`cli_commands.py`) - Core command-line interface utilities

## Extra Components (`genai_tk.extra`)

### Advanced Agents
- **React Agent** (`graphs/custom_react_agent.py`) - ReAct pattern implementation for reasoning and acting
- **SQL Agent** (`graphs/sql_agent.py`) - Specialized agent for database querying and analysis  
- **React Agent with Structured Output** (`graphs/react_agent_structured_output.py`) - ReAct with typed responses

### Research and Analysis Tools
- **GPT Researcher Chain** (`chains/gpt_researcher_chain.py`) - Automated research and information gathering
- **Image Analysis** (`image_analysis.py`) - Computer vision and image understanding capabilities
- **Custom Presidio Anonymizer** (`custom_presidio_anonymizer.py`) - PII detection and anonymization

### Tool Collections

#### LangChain Tools (`tools_langchain/`)
- **Web Search Tools** (`web_search_tool.py`) - Search engines integration
- **Multi Search Tools** (`multi_search_tools.py`) - Coordinated multi-source searching
- **SQL Tool Factory** (`sql_tool_factory.py`) - Database interaction tools
- **Config Loader Tools** (`config_loader.py`) - Configuration management tools

#### SmolAgents Tools (`tools_smolagents/`)
- **Browser Use** (`browser_use.py`) - Web browser automation
- **DataFrame Tools** (`dataframe_tools.py`) - Pandas data manipulation
- **SQL Tools** (`sql_tools.py`) - Database operations  
- **YFinance Tools** (`yfinance_tools.py`) - Financial data retrieval

### Data Processing
- **Retrievers** (`retrievers/bm25s_retriever.py`) - BM25 search implementation
- **Loaders** (`loaders/mistral_ocr.py`) - Document loading and OCR capabilities
- **KV Store Factory** (`kv_store_factory.py`) - Key-value storage abstraction
- **PGVector Factory** (`pgvector_factory.py`) - PostgreSQL vector database support

## Utilities (`genai_tk.utils`)

### Configuration and Management
- **Config Manager** (`config_mngr.py`) - Centralized configuration management
- **Logger Factory** (`logger_factory.py`) - Structured logging setup
- **Singleton** (`singleton.py`) - Singleton pattern implementation

### Streamlit Components (`streamlit/`)
- **Capturing Callback Handler** (`capturing_callback_handler.py`) - Stream LLM outputs to Streamlit
- **Auto Scroll** (`auto_scroll.py`) - Auto-scrolling for chat interfaces
- **Thread Issue Fix** (`thread_issue_fix.py`) - Threading fixes for Streamlit
- **Clear Result** (`clear_result.py`) - Result clearing utilities
- **Recorder** (`recorder.py`) - Session recording capabilities

### CLI Utilities (`cli/`)
- **LangChain Setup** (`langchain_setup.py`) - LangChain configuration helpers
- **LangGraph Agent Shell** (`langgraph_agent_shell.py`) - Interactive agent shell
- **SmolAgents Shell** (`smolagents_shell.py`) - SmolAgents interactive interface
- **Config Display** (`config_display.py`) - Configuration visualization

### Data Processing
- **Pydantic Utilities** (`pydantic/`) - Dynamic model creation, field adding, KV stores
- **Collection Helpers** (`collection_helpers.py`) - Data structure utilities  
- **Load Data** (`load_data.py`) - Data loading utilities
- **Markdown** (`markdown.py`) - Markdown processing
- **SQL Utils** (`sql_utils.py`) - SQL query helpers
- **Spacy Model Manager** (`spacy_model_mngr.py`) - NLP model management

### Framework Integration
- **CrewAI** (`crew_ai/remove_telemetry.py`) - CrewAI integration utilities
- **LangGraph** (`langgraph.py`) - LangGraph helper functions

## Supported AI Providers

- **OpenAI** - GPT models, embeddings, and tools
- **Anthropic** - Claude models via OpenRouter  
- **Local Models** - Ollama, VLLM, and other local inference
- **DeepSeek** - DeepSeek models and reasoning
- **Mistral** - Mistral AI models and embeddings
- **Groq** - Fast inference endpoints
- **LiteLLM** - Unified API for 100+ LLM providers

## Key Capabilities

- **Multi-Agent Workflows** - Build complex agent interactions with LangGraph
- **RAG (Retrieval Augmented Generation)** - Full RAG pipeline with vector stores
- **Tool Use** - Extensive tool ecosystem for agents
- **Structured Output** - Type-safe data extraction from LLMs  
- **Knowledge Graphs** - Kuzu and Cognee integration for graph reasoning
- **Web Automation** - Browser control and web scraping
- **Database Integration** - SQL, PostgreSQL, and vector databases
- **Research Automation** - Automated research and information synthesis
- **Configuration Management** - Flexible, hierarchical configuration system

## Installation and Usage

```bash
# Install with core dependencies
uv pip install git+https://github.com/tclatos/genai-tk@main

# Install with all extras
uv pip install "genai-tk[extra] @ git+https://github.com/tclatos/genai-tk@main"
```

```python
# Basic usage
from genai_tk.core import LLMFactory, EmbeddingsFactory
from genai_tk.extra.graphs import CustomReactAgent
from genai_tk.utils import ConfigManager

# Create LLM and embeddings
llm = LLMFactory.create("openai/gpt-4")
embeddings = EmbeddingsFactory.create("openai/text-embedding-3-small")

# Create agent
agent = CustomReactAgent(llm=llm)
```

This toolkit provides the foundation for building sophisticated AI applications with a focus on agent-based architectures, multi-modal AI capabilities, and production-ready components.