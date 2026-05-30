---
name: genai-tk-workflow-engine
description: Work on YAML-driven workflows, Prefect flow wrappers, workflow compilation, execution, resolver behavior, cache manifests, and CLI workflow commands.
---

# GenAI Toolkit Workflow Engine

## Read First

- `docs/workflows.md` — DSL reference, built-in workflows, how to create new ones
- `docs/prefect.md` — Running `@flow` functions, `flow_from_yaml`, Prefect server management
- `genai_tk/workflow/models.py`
- `genai_tk/workflow/resolver.py`
- `genai_tk/workflow/executor.py`
- `genai_tk/utils/prefect_server.py`
- `config/workflows.yaml`

## Code Map

| Concern | Paths |
|---|---|
| DSL models | `genai_tk/workflow/models.py` |
| resolver | `genai_tk/workflow/resolver.py` |
| `@workflow` decorator + registry | `genai_tk/workflow/registry.py` |
| Compiled (execution) models | `genai_tk/workflow/compiled_models.py` |
| Compiler (v2 → compiled) | `genai_tk/workflow/compiler.py` |
| Executor + pre-flight checks | `genai_tk/workflow/executor.py` |
| CLI commands (workflow) | `genai_tk/workflow/commands.py` |
| CLI commands (prefect server) | `genai_tk/workflow/prefect_commands.py` |
| Prefect server singleton | `genai_tk/utils/prefect_server.py` |
| Prefect integration | `genai_tk/workflow/prefect/` |
| Built-in flows | `genai_tk/workflow/prefect/flows/` |
| `PrefectFlowFactory` + `flow_from_yaml` | `genai_tk/workflow/prefect/flow_factory.py` |
| Cache manifests | `genai_tk/workflow/flow_cache/` |

## Public API (top-level imports)

```python
from genai_tk.workflow import (
    PrefectFlowFactory,          # compile + run a YAML workflow
    flow_from_yaml,              # parse YAML inline → Prefect @flow
    execute_workflow,            # run a ResolvedWorkflowInvocation
    resolve_workflow_invocation, # resolve "name/preset" → invocation
    load_workflows,              # read all YAML workflow defs from config
    WorkflowCompiler,            # compile WorkflowDefV2 → CompiledWorkflow
    workflow,                    # @workflow decorator
)
```

### `flow_from_yaml` — Inline YAML to Prefect Flow

Accepts a YAML **string**, **`Path`**, or **`dict`** and returns a standard Prefect `@flow`:

```python
from genai_tk.workflow import flow_from_yaml

flow = flow_from_yaml("""
workflows:
  my_wf:
    run: myapp.flows.my_flow
    defaults:
      base_dir: /data/in
      output_dir: /data/out
""")
flow()  # execute immediately
```

This is the preferred entry-point for notebooks, scripts, and testing flows outside the
config directory.

## DSL Rules

- Each workflow has either `run:` (single-step) **or** `pipeline:` (multi-step) — not both.
- `run:` is a **dotted Python path** or another **workflow name**.
- `defaults:` keys are auto-wired as `${values.KEY}` to the step — no explicit `with:` needed for single-step workflows.
- `params:` is documentation + validation metadata; `required: true` params without a `defaults` entry must be supplied via preset or `--set`.
- `presets:` live **inside** the workflow (not a separate top-level section).
- Select a preset with `workflow_name/preset_name` on the CLI.
- Pipeline steps use `after:` (alias `wait_for:`) for dependencies.
- `${values.key}` in `with:` blocks resolves against `defaults → preset → CLI --set`.
- `${paths.*}` interpolation in preset/default values is resolved against the global config.
- The `@workflow` decorator registers a callable by name; it appears in `cli workflow list` and can be referenced as `run: name` in YAML.

## Key v2 Resolver Behaviour

- `load_workflows(config)` reads `workflows:` directly; skips `step_templates`, `definitions`, `profiles` (v1 reserved keys).
- `resolve_workflow_invocation("name/preset", cli_overrides={...})` returns a `ResolvedWorkflowInvocation`.
- `_v2_to_workflow_spec` auto-wires `{k: "${values.k}"}` for all `defaults` keys in single-step workflows.
- `_validate_required_params` checks required params before execution; raises `WorkflowResolutionError` with CLI hints.
- `_expand_pipeline` recursively expands sub-workflow `run:` references; parent `with:` overrides sub-workflow auto-wired defaults.
- `_merge_dicts` uses `OmegaConf.merge` with the global config root as context so `${paths.*}` resolves in preset values.
- `executor.py` runs `_preflight_check_signatures` after compilation to catch unexpected kwargs before Prefect starts.

## Change Workflow

1. Modify `models.py` if new DSL fields are needed.
2. Update `resolver.py` to handle the new field in `_v2_to_workflow_spec` or `_expand_pipeline`.
3. Update `commands.py` if the CLI surface changes.
4. Update `config/workflows.yaml` with user-facing examples.
5. Run `uv run cli workflow validate` to confirm no regressions.

## Commands

```bash
uv run cli workflow list
uv run cli workflow show <name>
uv run cli workflow run <name>[/<preset>] --dry-run
uv run cli workflow serve <name>[/<preset>]   # register as Prefect deployment
uv run cli workflow validate
uv run cli prefect start      # start local server (auto-starts on workflow run)
uv run cli prefect stop
uv run cli prefect status
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/workflow -q
```

## Avoid

- Do not bypass the resolver for user-facing workflows.
- Do not split arguments between v1 `inputs:` and `params:` blocks — use `with:`.
- Do not use `step_templates:`, `definitions:`, or `workflow_profiles:` — these are legacy DSL keys (no longer supported).
- Do not require a live Prefect server for unit tests.
- Do not call `run_flow_ephemeral` or use `ephemeral_prefect_settings` — both are deleted.  Call flows directly after `prefect_server().ensure_running()`.


- Each workflow has either `run:` (single-step) **or** `pipeline:` (multi-step) — not both.
- `run:` is a **dotted Python path** or another **workflow name**.
- `defaults:` keys are auto-wired as `${values.KEY}` to the step — no explicit `with:` needed for single-step workflows.
- `params:` is documentation + validation metadata; `required: true` params without a `defaults` entry must be supplied via preset or `--set`.
- `presets:` live **inside** the workflow (not a separate top-level section).
- Select a preset with `workflow_name/preset_name` on the CLI.
- Pipeline steps use `after:` (alias `wait_for:`) for dependencies.
- `${values.key}` in `with:` blocks resolves against `defaults → preset → CLI --set`.
- `${paths.*}` interpolation in preset/default values is resolved against the global config.
- The `@workflow` decorator registers a callable by name; it appears in `cli workflow list` and can be referenced as `run: name` in YAML.

## Key v2 Resolver Behaviour

- `load_workflows(config)` reads `workflows:` directly; skips `step_templates`, `definitions`, `profiles` (v1 reserved keys).
- `resolve_workflow_invocation("name/preset", cli_overrides={...})` returns a `ResolvedWorkflowInvocation`.
- `_v2_to_workflow_spec` auto-wires `{k: "${values.k}"}` for all `defaults` keys in single-step workflows.
- `_validate_required_params` checks required params before execution; raises `WorkflowResolutionError` with CLI hints.
- `_expand_pipeline` recursively expands sub-workflow `run:` references; parent `with:` overrides sub-workflow auto-wired defaults.
- `_merge_dicts` uses `OmegaConf.merge` with the global config root as context so `${paths.*}` resolves in preset values.
- `executor.py` runs `_preflight_check_signatures` after compilation to catch unexpected kwargs before Prefect starts.

## Change Workflow

1. Modify `models.py` if new DSL fields are needed.
2. Update `resolver.py` to handle the new field in `_v2_to_workflow_spec` or `_expand_pipeline`.
3. Update `commands.py` if the CLI surface changes.
4. Update `config/workflows.yaml` with user-facing examples.
5. Run `uv run cli workflow validate` to confirm no regressions.

## Commands

```bash
uv run cli workflow list
uv run cli workflow show <name>
uv run cli workflow run <name>[/<preset>] --dry-run
uv run cli workflow serve <name>[/<preset>]   # register as Prefect deployment
uv run cli workflow validate
uv run cli prefect start      # start local server (auto-starts on workflow run)
uv run cli prefect stop
uv run cli prefect status
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/workflow -q
```

## Avoid

- Do not bypass the resolver for user-facing workflows.
- Do not split arguments between v1 `inputs:` and `params:` blocks — use `with:`.
- Do not use `step_templates:`, `definitions:`, or `workflow_profiles:` — these are legacy DSL keys (no longer supported).
- Do not require a live Prefect server for unit tests.
- Do not call `run_flow_ephemeral` or use `ephemeral_prefect_settings` — both are deleted.  Call flows directly after `prefect_server().ensure_running()`.
