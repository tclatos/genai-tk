# Architectural Design Documents (genai-tk)

This directory contains the foundational design specifications, architectural blueprints, and implementation guides for **genai-tk**.

The core architecture is presented in a didactic, numbered sequence designed for engineers, architects, and autonomous coding agents.

---

## 📚 Numbered Architecture Curriculum

| # | Design Document | Core Subject & Key Topics |
|---|---|---|
| **01** | [01_configuration_and_profiles.md](01_configuration_and_profiles.md) | **Configuration & Profile Overlays**: OmegaConf hierarchy, `${profile}` overlays, `${oc.env:VAR}` interpolation, proxy derivation (`NO_PROXY`), and typed Pydantic v2 schemas. |
| **02** | [02_ioc_provider_architecture.md](02_ioc_provider_architecture.md) | **Inversion of Control (IoC) & Dynamic Registry**: Universal framework extensibility via `ImportResolver`, role-based provider class fields (`chat_class`, `embeddings_class`, `decision_class`), CLI command registration, UI navigation, workflow step dispatch, and document converters. |
| **03** | [03_core_llm_and_factories.md](03_core_llm_and_factories.md) | **Core LLM, Embeddings & Factories**: `LlmFactory`, `EmbeddingsFactory`, `EmbeddingsStore`, model ID format (`model@provider`), YAML aliases, semantic caching (`LlmCache`), and LCEL `ChainRegistry`. |
| **04** | [04_decision_models_system1.md](04_decision_models_system1.md) | **Fast System 1 Decision Models**: Sub-100ms calibrated classification, typed primitives (`Noul`, `Choice`, `Score`), OpenRouter, EdenAI, Laya (`ModernBERT/mmBERT`), OpenAI drivers, and batch auto-chunking. |
| **05** | [05_agent_harness_and_middleware.md](05_agent_harness_and_middleware.md) | **Agent Harness & Middleware**: Unified `BaseHarness`, LangChain and DeerFlow runtime adapters, canonical event streaming (`HarnessEvent`), Presidio PII redaction, and user context propagation. |
| **06** | [06_sandbox_and_codeact.md](06_sandbox_and_codeact.md) | **Sandbox Isolation & Python CodeAct**: `AioSandboxBackend` protocol, Docker OpenSandbox lifecycle, core dump elimination (`ulimit core=0`), LibreOffice concurrency isolation, and CodeAct execution. |
| **07** | [07_workflow_engine_and_prefect.md](07_workflow_engine_and_prefect.md) | **Workflow Engine & Prefect-YAML**: Declarative YAML workflows, standalone `prefect-yaml` engine, DAG dependencies (`needs:`), and rule-selected ingestion routing. |
| **08** | [08_document_processing_pipeline.md](08_document_processing_pipeline.md) | **Document Processing Pipeline**: Multi-tier converter ladder (Docling, Mistral OCR, MarkItDown), HTML table preservation and splitting for RAG, BAML structured extraction, and NLP pipelines. |
| **09** | [09_observability_and_trajectories.md](09_observability_and_trajectories.md) | **Observability, Trajectories & Evaluation**: NeMo Relay OTLP telemetry exporting to Langfuse, Agent Trajectory Observability Format (ATOF), terminal TUI explorer (`cli trajectory`), and Decision Model evaluators. |
| **10** | [10_agent_user_interaction_protocol.md](10_agent_user_interaction_protocol.md) | **Agent-User Interaction Protocol (AUIP) & Multimodal Frontends**: Decoupling execution from presentation, NeMo Relay lifecycle bridge, retirement of `RichToolCallMiddleware`, and client adapters (Console, TUI, Streamlit, Remote Web). |

---

## 🔬 Empirical Studies, Proposals & Historical References

For external proposals, architecture feasibility studies, empirical benchmarks, and historical specs that have been superseded by the numbered curriculum, consult [docs/studies/](../studies/README.md):
- [Microsoft Copilot Studio Integration Architecture](../studies/copilot_studio_integration.md)
- [Cloud Enterprise Agent Architectures (Azure, AWS, GCP)](../studies/cloud_agent_architectures_ms_aws_gcp.md)
- [Azure AI Foundry Hosted Agents Assessment](../studies/azure_foundry_hosted_agents.md)
- [NemoClaw Harness Architecture Assessment](../studies/nemoclaw_harness_assessment.md)
- [Workflow DSL vs Kestra Assessment](../studies/workflow_dsl_vs_kestra.md)
- [Docker Sandbox Bot Detection Investigation](../studies/sandbox_bot_detection.md)
- [Historical Harness Interoperability Proposal](../studies/harness_interoperability_proposal_historical.md)
- [Historical Sandbox Backend Proposal](../studies/sandbox_backend_historical.md)
- [Historical LLM Factory 3-Route Proposal](../studies/llm_factory_refactoring_historical.md)
- [Historical NeMo Relay Trajectory Phases Spec](../studies/agent_trajectory_nemo_relay_spec.md)
- [Historical Reasoning Effort Parameter Spec](../studies/reasoning_effort_handling_spec.md)
- [Historical Copilot Agent Support Notes](../studies/copilot_agent_support_historical.md)
