---
name: genai-tk-workflow-engine
description: Work on YAML-driven workflows, Prefect flow wrappers, workflow compilation, execution, resolver behavior, cache manifests, and CLI workflow commands.
---

# GenAI Toolkit Workflow Engine

## Read First

- `docs/workflows.md`
- `docs/prefect.md`
- `genai_tk/workflow/models.py`
- `genai_tk/workflow/compiler.py`
- `genai_tk/workflow/executor.py`
- `config/workflows.yaml`

## Code Map

| Concern | Paths |
|---|---|
| DSL models | `genai_tk/workflow/models.py`, `genai_tk/workflow/compiled_models.py` |
| Resolution and compilation | `genai_tk/workflow/resolver.py`, `genai_tk/workflow/compiler.py` |
| Execution | `genai_tk/workflow/executor.py`, `genai_tk/workflow/commands.py` |
| Prefect integration | `genai_tk/workflow/prefect/` |
| Built-in flows | `genai_tk/workflow/prefect/flows/` |
| Cache manifests | `genai_tk/workflow/flow_cache/` |

## DSL Rules

- Use `workflow.defaults` plus profile values plus CLI `--set`; CLI values win.
- Step arguments live in flat `with:` blocks.
- `invoke.kind` is one of `flow`, `callable`, `workflow`, `task`, or `deployment` where supported.
- `wait_for` expresses dependencies; keep DAG behavior explicit.
- `foreach` fans out a step over resolved items.

## Change Workflow

1. Add or update YAML examples in `config/workflows.yaml` for user-facing workflow behavior.
2. Update Pydantic DSL models before compiler/executor logic if new fields are needed.
3. Keep Prefect-specific behavior inside `genai_tk/workflow/prefect/`.
4. Add dry-run tests for compilation and execution tests with lightweight callables.

## Commands

```bash
uv run cli workflow list
uv run cli workflow run <workflow_name> --dry-run
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/workflow -q
```

## Avoid

- Do not bypass the resolver for user-facing workflows.
- Do not split inputs between legacy `inputs:` and `params:` blocks; use `with:`.
- Do not require a live Prefect server for unit tests.
