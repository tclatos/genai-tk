"""Workflow execution engine.

This module is the public entry-point for executing a resolved workflow.
The heavy lifting is delegated to:

- :class:`~genai_tk.workflow.compiler.WorkflowCompiler` — YAML → compiled graph
- :class:`~genai_tk.workflow.prefect.flow_factory.PrefectFlowFactory` — compiled graph → Prefect flow → execution
"""

from __future__ import annotations

from typing import Any

from genai_tk.workflow.models import ResolvedWorkflowInvocation


class WorkflowExecutionError(RuntimeError):
    """Raised when a workflow step fails and ``on_failure`` is ``abort``."""


def _preflight_check_signatures(compiled: Any) -> None:
    """Validate each step's kwargs against its callable's signature.

    Catches unexpected keyword arguments (typos, renamed params, etc.) before
    the Prefect flow starts — giving a concise error instead of a verbose
    Prefect traceback.
    """
    import importlib
    import inspect

    from genai_tk.workflow.compiled_models import StepKind
    from genai_tk.workflow.prefect.flow_factory import _prepare_inputs

    for step in compiled.steps:
        if step.invoke.kind != StepKind.callable or not step.invoke.target:
            continue

        try:
            module_path, fn_name = step.invoke.target.rsplit(".", 1)
            fn = getattr(importlib.import_module(module_path), fn_name)
        except (ImportError, AttributeError, ValueError) as exc:
            raise WorkflowExecutionError(f"Step '{step.id}': cannot import '{step.invoke.target}': {exc}") from exc

        sig = inspect.signature(fn)
        # If the function accepts **kwargs it will handle any key
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            continue

        step_inputs = _prepare_inputs(step.with_, {})
        valid_params = set(sig.parameters)
        unexpected = sorted(set(step_inputs) - valid_params)
        if unexpected:
            raise WorkflowExecutionError(
                f"Step '{step.id}': unexpected argument(s) for '{step.invoke.target}': "
                f"{unexpected}.\n"
                f"Function accepts: {sorted(valid_params)}.\n"
                f"Hint: check the workflow 'defaults' / 'with:' keys match the function parameters."
            )


def execute_workflow(invocation: ResolvedWorkflowInvocation) -> dict[str, Any]:
    """Execute a resolved workflow invocation.

    Compiles the workflow spec into a :class:`~genai_tk.workflow.compiled_models.CompiledWorkflow`
    and runs it via :class:`~genai_tk.workflow.prefect.flow_factory.PrefectFlowFactory`.
    Independent steps execute concurrently; ordered steps wait for their
    dependencies via Prefect futures.

    Args:
        invocation: A resolved invocation produced by
            :func:`~genai_tk.workflow.resolver.resolve_workflow_invocation`.

    Returns:
        Mapping of ``step_id → result`` for all executed steps.
    """
    from genai_tk.workflow.compiler import WorkflowCompiler
    from genai_tk.workflow.prefect.flow_factory import PrefectFlowFactory
    from genai_tk.workflow.prefect.flow_factory import WorkflowExecutionError as _FlowError

    # Propagate --force flag as values so YAML steps can reference ${values.force}
    values = dict(invocation.values)
    if invocation.force:
        values.setdefault("force", True)
        values.setdefault("force_rebuild", True)

    compiled = WorkflowCompiler().compile(invocation.workflow, values)
    _preflight_check_signatures(compiled)
    try:
        return PrefectFlowFactory(compiled=compiled).run()
    except _FlowError as exc:
        raise WorkflowExecutionError(str(exc)) from exc
