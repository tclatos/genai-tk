"""Workflow execution engine.

Resolves step implementations from dotted paths and orchestrates execution
via Prefect flows/tasks, respecting dependency ordering.
"""

from __future__ import annotations

import importlib
from typing import Any

from loguru import logger

from genai_tk.workflow.models import ResolvedWorkflowInvocation, StepSpec


class WorkflowExecutionError(RuntimeError):
    """Raised when workflow execution fails."""


def _import_callable(dotted_path: str) -> Any:
    """Import a callable from a dotted Python path."""
    module_path, _, attr_name = dotted_path.rpartition(".")
    if not module_path:
        raise WorkflowExecutionError(f"Invalid step path '{dotted_path}': must be a dotted module.attribute path.")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise WorkflowExecutionError(f"Cannot import module '{module_path}': {exc}") from exc
    if not hasattr(module, attr_name):
        raise WorkflowExecutionError(f"Module '{module_path}' has no attribute '{attr_name}'.")
    return getattr(module, attr_name)


def _resolve_step_inputs(step: StepSpec, values: dict[str, Any]) -> dict[str, Any]:
    """Substitute ${profile.*} placeholders in step inputs/params with resolved values."""
    resolved: dict[str, Any] = {}

    for key, val in {**step.inputs, **step.params}.items():
        if isinstance(val, str) and val.startswith("${profile."):
            lookup_key = val[len("${profile.") : -1]
            resolved[key] = values.get(lookup_key)
        else:
            resolved[key] = val

    return resolved


def _topological_sort(steps: list[StepSpec]) -> list[StepSpec]:
    """Sort steps by dependency order (Kahn's algorithm)."""
    step_map = {s.id: s for s in steps}
    in_degree: dict[str, int] = {s.id: 0 for s in steps}
    for s in steps:
        for dep in s.needs:
            if dep not in step_map:
                raise WorkflowExecutionError(f"Step '{s.id}' depends on unknown step '{dep}'.")
            in_degree[s.id] += 1

    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    ordered: list[StepSpec] = []

    while queue:
        current = queue.pop(0)
        ordered.append(step_map[current])
        for s in steps:
            if current in s.needs:
                in_degree[s.id] -= 1
                if in_degree[s.id] == 0:
                    queue.append(s.id)

    if len(ordered) != len(steps):
        raise WorkflowExecutionError("Cycle detected in workflow step dependencies.")
    return ordered


def execute_workflow(invocation: ResolvedWorkflowInvocation) -> dict[str, Any]:
    """Execute a resolved workflow invocation.

    Runs each step in topological order using Prefect ephemeral mode.
    Returns a mapping of step_id -> result.
    """
    from genai_tk.utils.prefect_run import run_flow_ephemeral

    workflow = invocation.workflow
    values = invocation.values
    results: dict[str, Any] = {}

    ordered_steps = _topological_sort(workflow.steps)
    logger.info("Executing workflow '{}' ({} steps)", workflow.name, len(ordered_steps))

    for step in ordered_steps:
        step_kwargs = _resolve_step_inputs(step, values)
        # Remove None values - let the flow use its defaults
        step_kwargs = {k: v for k, v in step_kwargs.items() if v is not None}

        if invocation.force:
            step_kwargs["force"] = True

        logger.info("Running step '{}' ({})", step.id, step.uses)
        logger.debug("Step kwargs: {}", step_kwargs)

        callable_obj = _import_callable(step.uses)

        try:
            result = run_flow_ephemeral(callable_obj, **step_kwargs)
            results[step.id] = result
            logger.success("Step '{}' completed", step.id)
        except Exception as exc:
            if step.on_failure == "abort":
                raise WorkflowExecutionError(f"Step '{step.id}' failed: {exc}") from exc
            elif step.on_failure == "skip":
                logger.warning("Step '{}' failed (skipping): {}", step.id, exc)
                results[step.id] = None
            else:  # continue
                logger.warning("Step '{}' failed (continuing): {}", step.id, exc)
                results[step.id] = None

    logger.success("Workflow '{}' completed ({} steps)", workflow.name, len(ordered_steps))
    return results
