# Research Studies & Exploratory Investigations (genai-tk)

This directory contains research studies, comparative evaluations of external frameworks, and exploratory architectural investigations for **genai-tk**.

> **Policy**: Documents in this directory are strictly reserved for external surveys, comparative research, and architectural evaluations that have **not** been implemented in the codebase. Implemented features and architectural blueprints belong in [docs/design/](../design/README.md) or the core user guides.

---

## Index of Unimplemented Studies & Research

| Study / Report | Topic | Key Finding / Focus |
|---|---|---|
| [copilot_studio_integration.md](copilot_studio_integration.md) | **Microsoft Copilot Studio Integration** | Architecture for exposing LangChain/DeepAgent bots via Microsoft 365 Agents SDK |
| [cloud_agent_architectures_ms_aws_gcp.md](cloud_agent_architectures_ms_aws_gcp.md) | **Cloud Enterprise Architectures** | Reference patterns for Graph Navigation, Code Sandboxing, and Multi-Agent Orchestration across Azure, AWS, and GCP |
| [azure_foundry_hosted_agents.md](azure_foundry_hosted_agents.md) | **Azure Foundry Viability** | Technical assessment of Microsoft Azure AI Foundry hosted agents and SDK trade-offs |
| [nemoclaw_harness_assessment.md](nemoclaw_harness_assessment.md) | **Harness Architecture Assessment** | Comparative assessment of NemoClaw vs DeerFlow for genai-tk unified harness (DeerFlow selected; NemoClaw not integrated) |
| [workflow_dsl_vs_kestra.md](workflow_dsl_vs_kestra.md) | **Workflow Engine Comparison** | Comparative study of GenAI-TK YAML workflow DSL vs Kestra orchestration (Kestra not adopted; prefect-yaml created) |
| [sandbox_bot_detection.md](sandbox_bot_detection.md) | **Bot Detection & Browser Sandbox** | Diagnostic investigation of third-party login redirects and bot detection blockers in Docker/OpenSandbox |
| [codebase_audit_2026-09.md](codebase_audit_2026-09.md) | **Codebase Quality & Security Audit** | Comprehensive static analysis audit covering security, performance, and maintainability |
| [harness_interoperability_proposal_historical.md](harness_interoperability_proposal_historical.md) | **Harness Interoperability (Historical)** | Original proposal for cross-harness abstraction (superseded by Design 05) |
| [sandbox_backend_historical.md](sandbox_backend_historical.md) | **AioSandboxBackend (Historical)** | Initial sandbox specification (superseded by Design 06 & sandbox_support.md) |
| [llm_factory_refactoring_historical.md](llm_factory_refactoring_historical.md) | **LLM Factory Refactoring (Historical)** | Early 3-route factory proposal (superseded by Design 02 & 03) |
| [agent_trajectory_nemo_relay_spec.md](agent_trajectory_nemo_relay_spec.md) | **NeMo Relay Trajectory Phases (Historical)** | Multi-phase specification (superseded by Design 09 & trajectory.md) |
| [reasoning_effort_handling_spec.md](reasoning_effort_handling_spec.md) | **Reasoning Effort Handling Spec** | Provider-specific reasoning transport mapping (superseded by Design 03) |
| [copilot_agent_support_historical.md](copilot_agent_support_historical.md) | **Copilot Agent Support (Historical)** | Early design notes for Copilot agent support (superseded by scaffolding.md) |

---

## Promoted & Implemented Specifications

- **Telemetry, Trajectories & Evaluation**: Formerly an exploratory study, now fully implemented (NeMo Relay OTLP to Langfuse, Decision Model evaluators, ATOF TrajectoryStore) and documented in [docs/design/09_observability_and_trajectories.md](../design/09_observability_and_trajectories.md).
- **Workflow Engine & Prefect-YAML**: Workflow orchestration delegated to standalone `prefect-yaml`, documented in [docs/design/07_workflow_engine_and_prefect.md](../design/07_workflow_engine_and_prefect.md).
- **Enterprise Access Control**: Implemented in `genai-graph` and documented in [genai-graph design 03](../../genai-graph/docs/design/03_access_control_and_security_trimming.md).

---

## Related Documentation

- [docs/README.md](../README.md) — Master documentation index
- [docs/design/README.md](../design/README.md) — Architectural design curriculum
- [docs/benchmark_framework.md](../benchmark_framework.md) — Unified multi-dataset benchmark framework
- [docs/benchmarks_financebench_officeqa.md](../benchmarks_financebench_officeqa.md) — FinanceBench and OfficeQA Pro benchmark details
- [docs/deerflow_architecture_and_benchmarks.md](../deerflow_architecture_and_benchmarks.md) — Multi-dataset benchmark evaluation report
