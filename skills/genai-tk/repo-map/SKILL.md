---
name: genai-tk-repo-map
description: Navigate the genai-tk repository by mapping docs to code, config, tests, and existing skills. Use before making cross-cutting changes or when deciding which domain skill to load.
---

# GenAI Toolkit Repository Map

## Start Here

Read the closest doc first, then inspect the matching implementation paths. Do not infer behavior from filenames alone; this repo is configuration-driven and docs are the intended navigation layer.

## Domain Map

| Work area | Read first | Then inspect |
|---|---|---|
| CLI commands | `docs/cli.md`, `docs/scaffolding.md` | `genai_tk/main/cli.py`, `genai_tk/cli/`, `genai_tk/main/scaffolder.py`, `config/app_conf.yaml` |
| Configuration | `docs/configuration.md` | `genai_tk/utils/config_mngr.py`, `genai_tk/utils/config_exceptions.py`, `config/**/*.yaml` |
| LLMs, embeddings, cache | `docs/core.md`, `docs/llm-selection.md` | `genai_tk/core/`, `config/providers/`, `tests/unit_tests/core/` |
| Agents | `docs/agents.md`, `docs/deer-flow.md` | `genai_tk/agents/`, `config/agents/`, `tests/unit_tests/agents/` |
| Agent middleware | `docs/middleware-pii-and-routing.md` | `genai_tk/agents/langchain/middleware/`, `tests/unit_tests/agents/langchain/middleware/` |
| RAG | `docs/rag.md` | `genai_tk/core/retrievers/`, `genai_tk/workflow/rag/`, `config/rag.yaml`, `tests/unit_tests/core/test_retriever_factory.py` |
| Workflows and Prefect | `docs/workflows.md`, `docs/prefect.md` | `genai_tk/workflow/`, `genai_tk/workflow/prefect/`, `config/workflows.yaml` |
| MCP exposure | `docs/mcp-servers.md` | `genai_tk/mcp/`, `config/mcp/servers.yaml`, `tests/unit_tests/mcp/` |
| Browser automation | `docs/browser_control.md`, `docs/sandbox_support.md` | `genai_tk/agents/tools/sandbox_browser/`, `genai_tk/agents/tools/direct_browser/`, `genai_tk/agents/sandbox/` |
| Webapp | `docs/webapp.md` | `genai_tk/webapp/`, `config/webapp.yaml` |
| BAML | `docs/baml.md` | `genai_tk/extra/structured/`, `genai_tk/workflow/prefect/flows/baml_flow.py` |
| Evaluation/testing | `docs/TESTING_GUIDE.md`, `docs/evaluation_testing.md` | `tests/`, `pyproject.toml` dependency groups |

## Implementation Rules

- Use absolute imports from `genai_tk.*`.
- Use Pydantic v2 for structured config, DTOs, and results.
- Keep configuration in YAML where the docs describe YAML-driven behavior.
- Use `uv run ...` for commands and tests.
- Add or update tests under the closest `tests/unit_tests/<domain>/` path first; use integration tests only when behavior requires real services or CLI wiring.

## Verification Shortcuts

```bash
uv run pytest tests/unit_tests/core -q
uv run pytest tests/unit_tests/agents -q
uv run pytest tests/unit_tests/workflow -q
uv run pytest tests/unit_tests/mcp -q
uv run pytest tests/unit_tests/cli -q
```

Run `make lint` before finishing code changes that touch Python.
