# Changelog

All notable changes to genai-tk will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) — [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Onboarding and installation guide restructured in README**: Promoted the Installation section in [README.md](README.md) to the upper portion of the page directly beneath Core Domains & Pillars. Added a dedicated first-exploration section covering essential diagnostic and discovery CLI commands (`cli info doctor`, `cli info config`, `cli info models`, `cli info config-keys`, `cli info commands`), and added an onboarding guide inviting developers to explore and customize declarative YAML configuration files in `config/` ([config/app_conf.yaml](config/app_conf.yaml), [config/providers/](config/providers/), [config/agents/](config/agents/), [config/workflows.yaml](config/workflows.yaml)) using OmegaConf interpolations.
- **Native EdenAI support in models.dev and simplified catalog handling**: Since Eden.ai is now officially supported in [models.dev](https://models.dev/providers/edenai/), removed the custom EdenAI `/v3/models` scraping, authenticated downloads, and dedicated cache files (`edenai_models.json`, `edenai_eur_models.json`) in [genai_tk/core/models_db.py](genai_tk/core/models_db.py). `cli info llm-profile --reload` in [genai_tk/cli/commands_info.py](genai_tk/cli/commands_info.py) now performs a single unified download of `models.dev`, and [genai_tk/core/factories/llm_factory.py](genai_tk/core/factories/llm_factory.py) resolves `@edenai` and `@edenai-eur` directly from `models.dev` without special-case branches or API key requirements.
- **Deep-agent artifacts no longer pollute the project root**: filesystem backends are now wrapped in a `CompositeBackend` whose artifacts root is `data/generated/`, so DeepAgents' offloaded `large_tool_results/` land in a gitignored folder instead of the project root.
- **Sub-agent labelling in trajectories**: `cli trajectory show` and `cli trajectory tui` now tag LLM and tool activity performed by a deep-agent sub-agent with its name. The trajectory model records the owning agent per call (`LlmCall.agent`, `ToolCall.agent`), resolved from the ATOF scope hierarchy.

### Fixed

- **`cli` no longer fails on networks that cannot reach GitHub**: The `nlp` extra pinned the `en_core_web_md` spaCy model to a GitHub release URL, so every `uv run`/`uv sync` had to fetch it and aborted with a tunnel error when GitHub was unreachable. The model is no longer pinned in [pyproject.toml](pyproject.toml); the existing `SpaCyModelManager` downloads it on first use, and [docs/nlp.md](docs/nlp.md) documents the manual `python -m spacy download` alternative.
- **Git repository ingestion no longer drops its clones**: The output-directory exclusion added for markdownize sources (which skips `.cache` content) also swallowed git clones, because [genai_tk/workflow/prefect/flows/git_repo_flow.py](genai_tk/workflow/prefect/flows/git_repo_flow.py) stored them in `<md_output_dir>/.cache`. The `git_repo` flow now keeps its clone cache in a sibling `.cache` directory next to the output, so repositories are ingested again and the `test_git_repo_flow_e2e` CI failure is resolved.
- **Per-document tolerance in markdownize batches**: In [genai_tk/workflow/markdownize/flow.py](genai_tk/workflow/markdownize/flow.py), batch fan-in now collects results per future through `_collect_results`. A failing document is logged and skipped instead of aborting the whole markdownize bucket and discarding its successful outputs.
- **Document Ingestion Output Filtering & Self-Copy Prevention**: In [genai_tk/workflow/markdownize/flow.py](genai_tk/workflow/markdownize/flow.py) and [genai_tk/workflow/routing/dispatcher.py](genai_tk/workflow/routing/dispatcher.py), excluded files residing inside the target Markdown output directory (`md_output_dir`) when resolving input sources, preventing recursive self-processing, `.cache` ingestion, and `SameFileError` when copying Markdown files onto themselves in `_copy_file_task`.- **Fast-Failing Converter Dependency Validation**: Added `check_dependencies()` to `DocumentConverter` and implemented fail-fast dependency verification in `ConverterFactory.validate_converter()` and `markdownize_flow`, raising clean `ImportError` without chaining internal `ModuleNotFoundError` frames or launching dozens of doomed worker tasks with long tracebacks.
- **Environment Variable Fallback**: Added automatic `~/.env` fallback loading in `OmegaConfig.create` and `cli.main` so user-level API credentials (e.g. OpenRouter) are automatically loaded even when invoking CLI commands directly outside of `just`.

### Added

#### Observability, Telemetry & Agent Frontends
- **Official AG-UI Standard Support & FastAPI Gateway**: Extended AUIP in [genai_tk/protocols/auip/](genai_tk/protocols/auip/) with full bidirectional support for the industry-standard [AG-UI protocol](https://github.com/ag-ui-protocol/ag-ui) (`ag-ui-protocol` v1.0.0). Implemented `harness_events_to_ag_ui()` in [genai_tk/protocols/auip/ag_ui_bridge.py](genai_tk/protocols/auip/ag_ui_bridge.py) to translate canonical agent harness streaming events into standard AG-UI events (`RUN_STARTED`, `TEXT_MESSAGE_CONTENT`, `TOOL_CALL_START`, `TOOL_CALL_RESULT`, `REASONING_*`, `RUN_FINISHED`), with full Server-Sent Events encoding.
- **AUIP Server & CLI Commands**: Implemented a standalone FastAPI server in [genai_tk/protocols/auip/server.py](genai_tk/protocols/auip/server.py) and registered `AuipCommands` in [genai_tk/protocols/auip/cli_commands.py](genai_tk/protocols/auip/cli_commands.py) (`cli auip serve`, `cli auip info`), exposing `POST /agent` (AG-UI SSE stream), `POST /api/v1/auip/stream` (AUIP SSE stream), `GET /profiles` (agent profile enumeration), and `GET /health`. Configuration is loaded from `config/auip.yaml` via typed Pydantic v2 settings `AuipServerSettings`.
- **AG-UI errors are visible in the chat**: `harness_events_to_ag_ui()` now also emits run failures (e.g. a missing `harnessing` extra) as an assistant message and logs them server-side, since chat UIs usually hide `RUN_ERROR`. `cli auip serve` needs `uv sync --extra harnessing` for deep agents; `auip-client` shows an "agent is working" indicator during agent start-up.
- **Standalone Web Client (`auip-client`)**: Created a modern React 19 + TypeScript + Vite + Tailwind CSS web interface at [auip-client](/home/tcl/prj/auip-client), integrating `@assistant-ui/react`, `@assistant-ui/react-ag-ui`, and `@ag-ui/client`. Features real-time token streaming, collapsible chain-of-thought/reasoning blocks, tool invocation inspection badges, and an agent profile selector connected dynamically to the `/profiles` API.
- **Agent-User Interaction Protocol (AUIP) & NeMo Relay Bridge**: Implemented the universal, bidirectional Agent-User Interaction Protocol (AUIP) in [genai_tk/protocols/auip/](genai_tk/protocols/auip/) to decouple agent execution from presentation. Lifecycle scopes and marks emitted by NeMo Relay are translated into normalized AUIP events via `RelayAuipBridge` in [genai_tk/protocols/auip/relay_bridge.py](genai_tk/protocols/auip/relay_bridge.py).
- **Multimodal Frontend Adapters**: Updated `cli agents run` in [genai_tk/agents/harness/chat_repl.py](genai_tk/agents/harness/chat_repl.py) to render via `RichAuipRenderer` in [genai_tk/protocols/auip/renderers/rich_console.py](genai_tk/protocols/auip/renderers/rich_console.py), and updated the Streamlit workbench in [genai_tk/webapp/ui_components/harness_workbench.py](genai_tk/webapp/ui_components/harness_workbench.py) to natively consume AUIP events. Documented the remote web application integration pattern (FastAPI SSE / WebSockets) in skill `agent-user-interaction-protocol` and [docs/design/10_agent_user_interaction_protocol.md](docs/design/10_agent_user_interaction_protocol.md).
- **Consolidation of Protocols Package**: Organized interaction and integration protocols under `genai_tk/protocols/`, hosting both `genai_tk/protocols/auip/` and `genai_tk/protocols/mcp/`.
- **Deprecation of `RichToolCallMiddleware`**: Marked `RichToolCallMiddleware` in [genai_tk/agents/langchain/middleware/rich_middleware.py](genai_tk/agents/langchain/middleware/rich_middleware.py) as deprecated. Removed it from agent profile defaults in [config/examples/agents/defaults.yaml](config/examples/agents/defaults.yaml), [config/agents/langchain/browser.yaml](config/agents/langchain/browser.yaml), and sibling project configurations (`wiki`, `genai-graph`, `rfq-analyst`, `ekg-atos`).
- **Architectural Study on NeMo Relay & Agent-User Interaction Protocol (AUIP)**: Published [docs/studies/agent_user_interaction_protocol_and_relay.md](docs/studies/agent_user_interaction_protocol_and_relay.md) evaluating the retirement of `RichToolCallMiddleware` and the unified AUIP design.

#### Skills & Document Authoring
- **Quarto Authoring Community Skill**: Integrated official Posit Quarto authoring skill (`quarto-authoring`) in `skills/community/quarto/`, providing agent guidelines and syntax patterns for `.qmd` authoring, cross-references, callout blocks, table formatting, and multi-format publishing.

#### Agent CLI & Markdown Formatting
- **Automatic Markdown Detection & Formatting in `cli agents run`**: In [genai_tk/agents/harness/chat_repl.py](genai_tk/agents/harness/chat_repl.py) and [genai_tk/agents/harness/commands.py](genai_tk/agents/harness/commands.py), agent responses are now inspected for Markdown formatting via `looks_like_markdown` and rendered as rich Markdown in the terminal. Plain text is preserved, and a new `--raw` CLI option allows bypassing Markdown rendering when raw output is needed for scripts or pipes.
- **Enhanced Markdown Heuristics**: Added Markdown table detection (`| ... |`) and bold text matching to [genai_tk/utils/markdown.py](genai_tk/utils/markdown.py), weighting headings and structural syntax to reliably detect LLM-generated Markdown without false positives on plain text.

#### Core Architecture & Universal Inversion of Control (IoC)
- **Universal Inversion of Control (IoC) Dynamic Registry**: Elevated IoC to a foundational design principle across the framework. Configuration files declare fully qualified Python class names and callables (`module.path.ClassName`), dynamically resolved via `ImportResolver` without modifying core library code:
  - **Model & Decision Providers**: `config/providers/providers.yaml` uses role-based class fields (`chat_class`, `embeddings_class`, `decision_class`), completely eliminating hardcoded `if/elif` branching.
  - **CLI Command Groups**: `config/app_conf.yaml` (`cli.commands`) mounts subclasses of `CliTopCommand` or registration callables dynamically into the root Typer CLI.
  - **Streamlit Webapp Navigation**: `config/app_conf.yaml` (`ui.navigation`) resolves local page paths and cross-package library resources (`genai_tk://demos/agent.py`) without file duplication.
  - **Workflow Orchestration Tasks**: Declarative YAML workflows (`uses:`) resolve flow and task functions dynamically in `prefect-yaml`.
  - **Document Converters**: `config/markdownize.yaml` maps named converters (`docling`, `mistral_ocr`, `markitdown`) to target Python classes via `ConverterFactory`.
- **Modular Provider Mixins**: Decomposed model behavior handling into composable mixins (`ReasoningMixin` for effort normalization across OpenAI, Anthropic, and Gemini; `RoutingMixin` for OpenRouter/EdenAI gateway routing; `SeedMixin` for deterministic seed placement).
- **Dynamic Downstream Extension**: Enabled downstream projects (`genai-graph`, `wiki`, `officeqa`, `rfq_pricing`, `ekg-atos`) to register on-prem servers (e.g. vLLM, Ollama), custom CLI suites, and workflow steps purely in YAML via `global_config()`.

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

### Fixed
- **OpenAI & OpenRouter Chat Model Parameter Sanitization**: Cleaned internal metadata (`provider`, `special_env_vars`, OmegaConf `ListConfig` objects, and `default_quantization`) before passing payloads to `ChatOpenAI`, preventing `TypeError: Completions.create() got an unexpected keyword argument 'provider'` and serialization errors with OpenAI-compatible chat endpoints.

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
