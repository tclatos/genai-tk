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
    try:
        return PrefectFlowFactory(compiled=compiled).run()
    except _FlowError as exc:
        raise WorkflowExecutionError(str(exc)) from exc
