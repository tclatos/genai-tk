# GenAI Toolkit (genai-tk) Documentation Index

Welcome to the **GenAI Toolkit** documentation. This index maps all core concepts, developer guides, design specifications, and research studies across the toolkit.

---

## 🗺️ Architectural Pillars

```mermaid
flowchart TD
    subgraph Core["1. Core Foundation"]
        LLM["LLM & Embeddings Factories (core.md)"]
        CONF["Configuration & Profiles (configuration.md)"]
        RAG["Hybrid RAG & Retrievers (rag.md)"]
    end

    subgraph Agents["2. Agentic Systems"]
        HARNESS["Unified Harness Layer (harness.md)"]
        PROFILES["Agent Profiles (agents.md, deer-flow.md)"]
        BOX["Sandbox & CodeAct (sandbox_support.md, codeact.md)"]
        TOOLS["Tools & MCP (browser_control.md, mcp-servers.md)"]
    end

    subgraph Pipelines["3. Orchestration & Pipelines"]
        WF["YAML Workflow Engine (workflows.md)"]
        PREFECT["Prefect Flows (prefect.md)"]
        DOCS_PROC["Document Processing (markdownize.md, baml.md, nlp.md)"]
    end

    subgraph Observability["4. Observability & Evaluation"]
        BENCH["Unified Benchmark Framework (benchmark_framework.md)"]
        MON["Telemetry & Monitoring (monitoring.md)"]
        TRAJ["Trajectory Replay (trajectory.md)"]
    end

    Core --> Agents
    Core --> Pipelines
    Pipelines --> Observability
    Agents --> Observability
```

---

## 1. Core Framework & Models

| Guide | Description | Key Topics |
|---|---|---|
| [core.md](core.md) | **Core Engine** | `LlmFactory`, `EmbeddingsFactory`, `EmbeddingsStore`, `LlmCache`, `ChainRegistry` |
| [configuration.md](configuration.md) | **Configuration Management** | OmegaConf YAML hierarchy, env substitution (`${oc.env:...}`), profile overlays |
| [llm-selection.md](llm-selection.md) | **LLM Provider Guide** | Provider aliases, context windows, cost/speed trade-offs |
| [rag.md](rag.md) | **RAG Deep-Dive** | `RetrieverFactory`, hybrid search (dense + BM25), rerankers, Chroma/PgVector |

---

## 2. Autonomous Agents & Tooling

| Guide | Description | Key Topics |
|---|---|---|
| [agents.md](agents.md) | **Agent Architecture** | LangChain & DeerFlow agent configuration, agent profiles, ReAct / Deep planning |
| [harness.md](harness.md) | **Harness Abstraction** | Unified `BaseHarness`, shared event streaming, cross-harness middleware |
| [deer-flow.md](deer-flow.md) | **DeerFlow Integration** | Pro-mode planning, native Python execution, interactive chat commands |
| [sandbox_support.md](sandbox_support.md) | **Sandbox Infrastructure** | OpenSandbox Docker container, file isolation, safe command execution |
| [codeact.md](codeact.md) | **Python CodeAct** | Stateful sandboxed Python execution for guaranteed mathematical accuracy |
| [browser_control.md](browser_control.md) | **Browser Automation** | Playwright tools, headless browsing, secure credential autofill |
| [mcp-servers.md](mcp-servers.md) | **MCP Protocol** | Model Context Protocol integration, server configurations, stdio/SSE |
| [middleware-pii-and-routing.md](middleware-pii-and-routing.md) | **Middleware & Privacy** | Presidio anonymization, sensitivity-based model routing, auditing |

---

## 3. Workflows, Ingestion & Structured Processing

| Guide | Description | Key Topics |
|---|---|---|
| [workflows.md](workflows.md) | **Workflow Engine** | YAML-driven task orchestration, step dependencies (`needs:`), CLI integration |
| [prefect.md](prefect.md) | **Prefect Integration** | Writing Prefect flows, ephemeral vs server modes, background workers |
| [markdownize.md](markdownize.md) | **Document Conversion** | PDF/Office conversion ladder: Mistral OCR → Docling → MarkItDown |
| [baml.md](baml.md) | **Structured Extraction** | BAML schema-based LLM extraction, type safety, low latency |
| [nlp.md](nlp.md) | **NLP Subsystem** | spaCy integration, entity recognition, French language pipelines |

---

## 4. Observability, Benchmarking & Deployment

| Guide | Description | Key Topics |
|---|---|---|
| [benchmark_framework.md](benchmark_framework.md) | **Benchmark Engine** | 6-stage evaluation pipeline, metric aggregation, Mafin 2.5 judge rules |
| [monitoring.md](monitoring.md) | **Monitoring & Tracing** | LangFuse, LangSmith, and OpenTelemetry instrumentation |
| [trajectory.md](trajectory.md) | **Agent Trajectories** | NeMo Relay trajectory recording, replay, and post-run analysis |
| [webapp.md](webapp.md) | **Streamlit Web Application** | Multi-page webapp, agent playground, workflow runner, navigation |
| [docker.md](docker.md) | **Docker Deployment** | Multi-stage Dockerfile, just recipes, volume mounts |
| [scaffolding.md](scaffolding.md) | **Scaffolding (`cli init`)** | Bootstrapping new AI apps, configuration presets, Copilot integration |
| [TESTING_GUIDE.md](TESTING_GUIDE.md) | **Testing Guide** | Pytest conventions, fake providers (`parrot_local@fake`), async test patterns |

---

## 5. Architecture Design Curriculum & Studies

| Guide | Description | Key Topics |
|---|---|---|
| [design/README.md](design/README.md) | **Architectural Design Curriculum** | Didactic 9-part numbered curriculum covering configuration, IoC, core models, harness, sandbox, workflows, and observability |
| [design/01_configuration_and_profiles.md](design/01_configuration_and_profiles.md) | **Design 01: Configuration & Profiles** | OmegaConf hierarchy, profile overlays, env interpolation, proxy derivation |
| [design/02_ioc_provider_architecture.md](design/02_ioc_provider_architecture.md) | **Design 02: IoC & Dynamic Registry** | Universal extensibility via `ImportResolver`, role-based class fields, CLI commands, UI navigation, workflows, converters |
| [design/03_core_llm_and_factories.md](design/03_core_llm_and_factories.md) | **Design 03: Core LLM & Factories** | `LlmFactory`, `EmbeddingsFactory`, `EmbeddingsStore`, semantic cache, LCEL |
| [design/04_decision_models_system1.md](design/04_decision_models_system1.md) | **Design 04: Decision Models (System 1)** | Calibrated `Noul`, `Choice`, `Score` classification, sub-100ms drivers |
| [design/05_agent_harness_and_middleware.md](design/05_agent_harness_and_middleware.md) | **Design 05: Agent Harness & Middleware** | Unified `BaseHarness`, canonical event stream, PII & sensitivity routing |
| [design/06_sandbox_and_codeact.md](design/06_sandbox_and_codeact.md) | **Design 06: Sandbox & Python CodeAct** | `AioSandboxBackend`, Docker pre-warming, `ulimit core=0`, LibreOffice isolation |
| [design/07_workflow_engine_and_prefect.md](design/07_workflow_engine_and_prefect.md) | **Design 07: Workflows & Prefect-YAML** | Declarative YAML workflows, `prefect-yaml`, route tables, ingest dispatch |
| [design/08_document_processing_pipeline.md](design/08_document_processing_pipeline.md) | **Design 08: Document Processing** | Converter ladder (Docling, Mistral, MarkItDown), table preservation, BAML |
| [design/09_observability_and_trajectories.md](design/09_observability_and_trajectories.md) | **Design 09: Observability & Trajectories** | NeMo Relay OTLP to Langfuse, ATOF `TrajectoryStore`, terminal TUI |
| [design/10_agent_user_interaction_protocol.md](design/10_agent_user_interaction_protocol.md) | **Design 10: AUIP & Multimodal Frontends** | AUIP events, NeMo Relay bridge, AG-UI server (`cli auip serve`), Rich/Streamlit/web clients |
| [studies/README.md](studies/README.md) | **Research Studies (Unimplemented)** | Surveys and comparative evaluations (Azure Foundry, Cloud Architectures, NemoClaw) |
| [SKILLS.md](SKILLS.md) | **Skills Guide** | 4-tier skills architecture (`runtime`, `development`, `governance`, `vendor`) |

---

## 🚀 Interactive Notebooks

Check out the interactive Jupyter demos in `notebooks/`:
- [harness_quickstart.ipynb](../notebooks/harness_quickstart.ipynb) — Building LangChain & DeerFlow agents with BaseHarness.
- [harness_middleware_demo.ipynb](../notebooks/harness_middleware_demo.ipynb) — PII anonymization and dynamic model routing middleware.
- [sandbox_and_python_executor_demo.ipynb](../notebooks/sandbox_and_python_executor_demo.ipynb) — Stateful Python execution and Docker sandboxing.
- See: [notebooks/README.md](../notebooks/README.md) for setup and execution instructions.
