# GenAI Toolkit (genai-tk) Interactive Notebooks

This directory contains interactive Jupyter notebooks demonstrating advanced agent capabilities, unified harness abstraction, middleware pipelines, and sandbox execution in **genai-tk**.

---

## 🗺️ Notebook Catalog

| Notebook | Focus | Key Concepts Demonstrated |
|---|---|---|
| [harness_quickstart.ipynb](harness_quickstart.ipynb) | **Unified Agent Harness Quickstart** | Creating and executing agents across **LangChain** and **DeerFlow** harnesses using the single `BaseHarness` API, streaming event iteration, and MCP tool attachment. |
| [harness_middleware_demo.ipynb](harness_middleware_demo.ipynb) | **Harness Middleware & Privacy Pipelines** | Cross-harness middleware: automated PII anonymization with Microsoft Presidio, reversible masking, sensitivity-based model routing, and audit logging. |
| [sandbox_and_python_executor_demo.ipynb](sandbox_and_python_executor_demo.ipynb) | **Sandboxing & Stateful CodeAct Execution** | Executing untrusted Python code safely using `LocalPythonExecutor` and `DockerSandboxBackend` (OpenSandbox), host-tool bridging, and stateful multi-turn computations. |

---

## How to Run These Notebooks

Run with `uv` from the repository root:

```bash
# Launch Jupyter Lab
uv run jupyter lab notebooks/
```

Or open directly in **VS Code** with the Python environment configured.

---

## Related Documentation

- [docs/README.md](../docs/README.md) — Master documentation index
- [docs/harness.md](../docs/harness.md) — Unified BaseHarness architecture
- [docs/middleware-pii-and-routing.md](../docs/middleware-pii-and-routing.md) — Middleware, PII detection, and sensitivity routing
- [docs/sandbox_support.md](../docs/sandbox_support.md) — OpenSandbox container configuration and security
- [docs/codeact.md](../docs/codeact.md) — Stateful sandboxed Python execution for agents
