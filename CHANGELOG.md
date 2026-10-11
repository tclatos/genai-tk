# Changelog

All notable changes to genai-tk will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) — [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Core Architecture & Inversion of Control (IoC)
- **Inversion of Control (IoC) Provider Architecture**: Refactored `LlmFactory`, `EmbeddingsFactory`, and `DecisionModelFactory` to an IoC registry pattern. Provider implementations are declared in `config/providers/providers.yaml` via role-based class fields (`chat_class`, `embeddings_class`, `decision_class`), completely eliminating hardcoded `if/elif` branching.
- **Modular Provider Mixins**: Decomposed model behavior handling into composable mixins (`ReasoningMixin` for effort normalization across OpenAI, Anthropic, and Gemini; `RoutingMixin` for OpenRouter/EdenAI gateway routing; `SeedMixin` for deterministic seed placement).
- **Dynamic Provider Registry**: Enabled downstream applications to register proprietary or on-prem models (e.g. vLLM, Ollama) purely in YAML via `global_config().get('providers')`.

#### Fast System 1 Decision Models
- **Sub-100ms Decision Engine** (`genai_tk/core/decision_models/`): Introduced non-autoregressive decision models for high-throughput, low-cost classification, routing, and scoring.
- **Mathematical Primitives**: Typed calibrated primitives: `Noul` ($p \in [0, 1]$ binary probability), `Choice` (categorical distribution over criteria), and `Score` (expected value over an ordered rubric).
- **Typed Class Hierarchy**: Standardized on strict Pydantic v2 models (`BaseQuestion`, `BaseAnswer`, `_RawDecisionItem`), eliminating `Any`.
- **Multi-Backend Drivers**: Added drivers for OpenRouter (Clef Flash, Clef, Jev, Solar Decide), EdenAI Decisions API, Laya (`ModernBERT/mmBERT` in-process router or HTTP daemon), OpenAI (`OpenAIDecisions`), and deterministic offline fake.
- **Batch Processing & Auto-Chunking**: Added `batch_invoke` with automatic chunking based on model-specific `max_questions` limits and CLI classifier commands (`cli core decision`).

#### Observability & Trajectory Tooling
- **NeMo Relay OTLP Telemetry**: Integrated NVIDIA NeMo Relay to stream OpenTelemetry (OTLP) traces directly to Langfuse and LangSmith.
- **Agent Trajectory Observability Format (ATOF)**: File-backed `TrajectoryStore` (`genai_tk/extra/monitoring/trajectory_store.py`) capturing complete cognitive trajectories (thoughts, tool calls, tool outputs, tokens, latencies).
- **Interactive Terminal TUI**: Launched `cli trajectory tui` for interactive step-by-step cognitive replay in the terminal with mouse selection and detailed timing breakdowns.
- **Decision Model Evaluators**: Retired brittle legacy `openevals` and `agentevals` in favor of calibrated Decision Model rubrics and deterministic tool sequence matching.

#### Model Context Protocol (MCP) 2.0
- **MCP 2.0 Standard**: Updated the server and client implementation to official MCP 2.0 with snake_case wire models.
- **Multi-Transport Support**: Native support for `stdio`, `sse`, and `streamable-http` transports with stdio stream isolation protecting JSON-RPC from rogue C/Rust output.
- **Backward Compatibility Shims**: Added shims in `genai_tk.mcp.compat` for seamless interoperability with legacy MCP 1.x and `langchain-mcp-adapters`.

#### Workflow Orchestration & Ingestion Routing
- **Delegation to `prefect-yaml`**: Delegated core workflow DSL compilation and execution to the standalone `prefect-yaml` engine, supporting YAML DAG dependencies (`needs:`), parameter interpolation, and retries.
- **Rule-Selected Ingestion Workflows**: Implemented orthogonal document ingestion via `config/ingest_routes.yaml`. Ordered pathspec rules dispatch specialized sub-workflows (PDF OCR, Office conversion, Web page fetchers, Git repository ingestion) in parallel with single-writer fan-in into graph databases.
- **Self-Healing Prefect Daemon**: Automated recovery of stale locks and SQLite database state during startup.

#### Document Processing & Conversion
- **Docling Converter Integration**: Added IBM Docling converter with layout-aware table extraction and multi-column reading order reconstruction.
- **Config-Driven `ConverterFactory`**: Declarative conversion profiles (`fast`, `medium`, `best`) in `config/markdownize.yaml`.
- **Table Preservation & Splitting**: Automated HTML table sanitization and oversized table chunk splitting with preserved headers for RAG.
- **Linguistic Processing**: Standardized spaCy pipelines on `en_core_web_md` and `fr_core_news_md`, with Snowball stemming for full-text search.

#### Sandbox & Concurrency Hardening
- **`ulimit core=0` Isolation**: Configured Docker sandbox containers with `ulimit core=0` to strictly prevent massive core dumps from exhausting host disk space.
- **LibreOffice Concurrency Isolation**: Injected isolated temporary user profiles (`-env:UserInstallation`) for every concurrent LibreOffice conversion, completely eliminating silent lock-file race condition failures.
- **User Context Propagation**: Unified identity and tenant context propagation through LangChain and DeerFlow harnesses for downstream access control.

#### Documentation Architecture
- **Didactic Numbered Curriculum**: Reorganized architectural documentation in `docs/design/` into a 9-part numbered curriculum starting with configuration and IoC, through core models, harness, sandbox, workflows, and observability.
- **Strict Studies Policy**: Cleaned `docs/studies/` to strictly contain unimplemented research and exploratory surveys, promoting implemented features to design specifications.

---

### Previous Changes

## [0.1.0] — 2025-09-30

Initial standalone release, extracted from genai-blueprint.

- LLM Factory, Embeddings Factory, Vector Store Factory (Chroma, Weaviate, PGVector)
- ReAct Agent, SQL Agent, custom LangGraph graphs
- MCP Client, Chain Registry, Structured Output
- LangChain tools: web search, SQL, RAG, config loader
- SmolAgents tools: browser, DataFrame, SQL, YFinance
- RAG pipeline: BM25, ensemble retriever, Mistral OCR loader
- Knowledge Graph utilities (Cognee, Kuzu)
- Hierarchical YAML configuration with OmegaConf, env var substitution
- Unified CLI (`cli`) with framework-specific shells
- Python 3.12+, uv, ruff, pytest with asyncio support
