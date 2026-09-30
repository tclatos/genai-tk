# Design Specifications & Architecture Guides (genai-tk)

This directory contains design documents, architecture specifications, and implementation guides for subsystems in **genai-tk**.

For empirical evaluations, benchmark reports, and diagnostic investigations, see [docs/studies/](../studies/README.md).

---

## Active & Implemented Design Documents

| Document | Topic | Status |
|---|---|---|
| [llm_factory_refactoring.md](llm_factory_refactoring.md) | Three-route LLM provider architecture (OpenAI-compatible, fake, specialized) | **Implemented** in `genai_tk/core/` |
| [harness_interoperability_proposal.md](harness_interoperability_proposal.md) | Unified `BaseHarness` design abstracting DeerFlow and LangChain | **Implemented** in `genai_tk/agents/harness/` |
| [copilot-agent-support.md](copilot-agent-support.md) | VS Code Copilot agent skills and `.instructions.md` integration | **Implemented** in `skills/copilot/` |
| [copilot-studio-integration.md](copilot-studio-integration.md) | Microsoft Copilot Studio integration architecture via Microsoft 365 Agents SDK | **Proposal / In Progress** |
| [sandbox_backend.md](sandbox_backend.md) | `AioSandboxBackend` protocol and low-level execution architecture | **Implemented** in `genai_tk/agents/sandbox/` |
| [agent_trajectory_nemo_relay.md](agent_trajectory_nemo_relay.md) | NeMo Relay ATOF trajectory store and `cli trajectory` tools | **Implemented** (Phases 0–3) |
| [markdown_knowlege_tree.md](markdown_knowlege_tree.md) | Markdown Knowledge Tree (`mdktree`) specification | **Design Brief** |
| [reasoning_effort_handling.md](reasoning_effort_handling.md) | Handling of `reasoning_effort` across OpenAI, Anthropic, Gemini providers | **Reference Guide** |
