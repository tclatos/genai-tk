"""Workflow execution delegating to standalone prefect_yaml package."""

from __future__ import annotations

from typing import Any

from prefect_yaml.models.authoring import ResolvedWorkflowInvocation
from prefect_yaml.runtime import WorkflowExecutionError
from prefect_yaml.runtime import execute_workflow as _py_execute_workflow


def execute_workflow(invocation: ResolvedWorkflowInvocation) -> dict[str, Any]:
    """Execute a resolved workflow invocation."""
    if invocation.force:
        invocation.values.setdefault("force", True)
        invocation.values.setdefault("force_rebuild", True)
    return _py_execute_workflow(invocation)


__all__ = [
    "WorkflowExecutionError",
    "execute_workflow",
]
