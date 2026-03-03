# Changelog

All notable changes to genai-tk will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Deep Agent Text-to-SQL Example with Progressive Skill Disclosure**
  - New `text2sql` profile: SQL agent for Chinook music database with dynamic skill loading
  - Core identity (role, safety rules, approach) in `system_prompt` YAML field — always present in every LLM call
  - Two dynamically-loaded `SKILL.md` files via `SkillsMiddleware`:
    - `query-writing/SKILL.md` — SQL query workflows for simple and complex queries
    - `schema-exploration/SKILL.md` — Database discovery and relationship mapping (Chinook schema reference)
  - **Filesystem backend support** (`type: filesystem` in `BackendConfig`)
    - `create_sql_toolkit_tools()` factory: low-level SQL tools (list_tables, get_schema, query_checker, sql_db_query) for agent reasoning
    - `_resolve_interpolation()` helper in config.py: resolves OmegaConf variables (e.g., `${paths.project}`)
  - **Rich tracing enhancements**:
    - `RichToolCallMiddleware._print_llm_call()` shows available skills extracted from system message (skills injected by `SkillsMiddleware`)
    - Tool Call panels highlight skill reads (📖 icon) when LLM accesses SKILL.md files
    - LLM Call panels display `Skills: <names>` metadata alongside model name and message count
  - Factory rewrite `_create_deep_agent()`:
    - Passes `skills=relative_paths` parameter to `create_deep_agent()` (native deepagents support)
    - Auto-creates `FilesystemBackend` if skills present but no backend configured
    - Converts absolute skill paths to backend-relative paths for proper resolution via `SkillsMiddleware`
    - Respects YAML `system_prompt` field (no longer auto-concatenates skill content)
  - Configuration resolution: `_resolve_skill_dirs()` uses OmegaConf to handle `${paths.project}/skills/example` style paths
  - `config/basic/agents/langchain.yaml`: new `text2sql` profile with full documentation and examples
  - Tools: [shared_config_loader.py](genai_tk/tools/langchain/shared_config_loader.py) — `_resolve_config_vars()` helper for OmegaConf interpolation in tool parameters

- **Unified LangChain Agent System** (`cli agents langchain`)
  - Single `langchain.yaml` config file replaces the former `langchain.yaml` + `deepagents.yaml` split
  - New Pydantic config models: `LangchainAgentsConfig`, `AgentProfileConfig`, `AgentDefaults`, `MiddlewareConfig`, `CheckpointerConfig`
  - `type: react | deep | custom` field per profile selects the agent engine
  - Global `defaults` section; profiles inherit and override per-field
  - Middleware instantiation from YAML (`class: module:ClassName` + arbitrary kwargs)
  - Checkpointer configuration (`type: none | memory | class`)
  - `cli agents langchain` replaces `cli agents react` and `cli agents deep`
  - New options: `--profile/-p`, `--type/-t`, `--llm/-m`, `--chat/-c`, `--mcp`, `--stream/-s`, `--list/-l`
  - `--list` renders a Rich table of all profiles with type, LLM, tools, MCP servers, features
  - Console warning when deep-only fields (`skill_directories`, `subagents`) are used on non-deep profiles
  - `genai_tk/agents/deep/` directory removed; `genai_tk/core/deep_agents.py` slimmed to runtime helpers only
  - `genai_tk/tools/langchain/shared_config_loader.py` reduced to `process_langchain_tools_from_config` only
  - `genai_tk/mcp/agent_tool.py` updated to use unified factory
  - New unit tests: `tests/unit_tests/agents/langchain/test_config.py` (32 tests)
  - Rewritten: `tests/unit_tests/tools/langchain/test_shared_config_loader_llm.py` (20 tests)
- **`AioSandboxBackend` — full `SandboxBackendProtocol` implementation**
  - `AioSandboxBackend` now inherits `SandboxBackendProtocol` (extends `BackendProtocol`) from deepagents
  - New `id` property: returns Docker container short ID when running, random hex otherwise
  - New `aexecute(command, *, timeout)` → `ExecuteResponse`
  - New `als_info(path)` → `list[FileInfo]`  (directory listing with metadata)
  - New `aread(file_path, offset, limit)` → paginated file content with 1-based line numbers
  - New `awrite(file_path, content)` → `WriteResult`  (errors if file already exists)
  - New `aedit(file_path, old, new, replace_all)` → `EditResult`
  - New `agrep_raw(pattern, path, glob)` → `list[GrepMatch] | str`
  - New `aglob_info(pattern, path)` → `list[FileInfo]`
  - New `aupload_files(files)` → `list[FileUploadResponse]`
  - New `adownload_files(paths)` → `list[FileDownloadResponse]`
  - 39 unit tests (no Docker required); integration test suite against real container
- **Backend selection in `langchain.yaml`**
  - New `BackendConfig` Pydantic model: `type: none | aio_sandbox | class`
  - `aio_sandbox` type starts `AioSandboxBackend` (Docker sandbox) automatically; all `AioSandboxBackendConfig` fields settable as sibling YAML keys
  - `class` type loads any deepagents `BackendProtocol` via qualified import path; constructor kwargs in `kwargs:` mapping
  - `AgentDefaults.backend` defaults to `type: none`; profiles inherit or override
  - `create_backend()` async factory wired into `create_langchain_agent()`
  - Backend lifecycle managed: `start()` called in factory, `stop()` called in `run_langchain_agent_shell` / `run_langchain_agent_direct` cleanup
  - Console warning when a non-`deep` profile sets an explicit non-none backend
  - `config/basic/agents/langchain.yaml` updated with `backend:` docs and commented examples
  - 20 additional unit tests in `test_config.py` (54 total): `TestBackendConfig`, `TestResolveProfileBackend`, `TestCreateBackend`

### Fixed
- **DeerFlow CLI Output Cleanup**
  - Fixed duplicate response text in CLI output caused by DeerFlow's `stream_mode="values"` yielding full-text per AIMessage
  - Suppressed internal print noise from DeerFlow modules (sandbox acquisition, memory updates, thread titling)
  - Suppressed DeerFlow standard-library logger noise (error messages from non-critical operations like file-not-found in sandbox)
  - Fixed `%s` format string bugs in config bridge logging (now uses f-strings for loguru compatibility)
  - Improved error detection to catch both "Error:" and "Error invoking tool" error formats
  - Strip model chain-of-thought reasoning markers (`assistantanalysis`, `assistantcommentary`, `assistantfinal`) from final response
  - Display host-side output file paths at end of single-shot DeerFlow runs

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
uv pip install git+https://github.com/tclatos/genai-tk@main

# Install with all extras
uv pip install "genai-tk[extra] @ git+https://github.com/tclatos/genai-tk@main"
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