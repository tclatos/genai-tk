# Empirical Studies, Benchmarks & Architectural Evaluations

This directory contains research studies, empirical benchmark assessments, architectural evaluations, and diagnostic reports produced during the development and evaluation of **genai-tk**.

Unlike core framework documentation (in `docs/`) which documents stable user-facing APIs, and active design specifications (in `docs/design/`), documents in this directory record empirical evidence, framework comparisons, benchmark findings, and technical investigations.

---

## Index of Studies

| Study / Report | Topic | Key Finding / Focus |
|---|---|---|
| [deerflow_architecture_and_benchmarks.md](deerflow_architecture_and_benchmarks.md) | **Benchmark Evaluation Report** | Multi-dataset benchmark evaluation (OfficeQA, MMLongBench) comparing DeerFlow vs DeepAgents, Python CodeAct calculations, and Document Graph navigation |
| [cloud_agent_architectures_ms_aws_gcp.md](cloud_agent_architectures_ms_aws_gcp.md) | **Cloud Enterprise Architectures** | Implementation patterns for Graph Navigation, Code Sandboxing, and Multi-Agent Orchestration across Azure, AWS, and GCP |
| [azure_foundry_hosted_agents.md](azure_foundry_hosted_agents.md) | **Azure Foundry Viability** | Technical assessment of Microsoft Azure AI Foundry hosted agents and SDK trade-offs |
| [nemoclaw_harness_assessment.md](nemoclaw_harness_assessment.md) | **Harness Architecture Assessment** | Comparative assessment of NemoClaw vs DeerFlow for genai-tk unified harness |
| [workflow_dsl_vs_kestra.md](workflow_dsl_vs_kestra.md) | **Workflow Engine Comparison** | Comparative study of GenAI-TK YAML workflow DSL vs Kestra orchestration |
| [workflow_engine_prefect_redesign.md](workflow_engine_prefect_redesign.md) | **Prefect Engine Redesign** | Architectural critique and roadmap for the Prefect-native workflow engine |
| [sandbox_bot_detection.md](sandbox_bot_detection.md) | **Bot Detection & Browser Sandbox** | Diagnostic investigation and resolution of login redirect blockers in Docker/OpenSandbox |
| [access_control_agentic_rag.md](access_control_agentic_rag.md) | **Enterprise Access Control Study** | Security design investigation evaluating 3 approaches to document-level ACL in Agentic RAG |
| [deer_flow_input_box_patch.md](deer_flow_input_box_patch.md) | **UI Stale Model Investigation** | Diagnostic post-mortem and frontend patch for DeerFlow model selection |
| [codebase_audit_2026-09.md](codebase_audit_2026-09.md) | **Codebase Quality & Security Audit** | Comprehensive static analysis audit covering security, performance, and maintainability |

---

## Related Documentation

- [docs/README.md](../README.md) — Master documentation index
- [docs/design/README.md](../design/README.md) — Active design specifications and proposals
- [docs/benchmark_framework.md](../benchmark_framework.md) — Unified multi-dataset benchmark framework
- [docs/benchmarks_financebench_officeqa.md](../benchmarks_financebench_officeqa.md) — FinanceBench and OfficeQA Pro benchmark details
