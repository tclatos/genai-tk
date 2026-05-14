"""PrefectFlowFactory: builds a real Prefect flow from a CompiledWorkflow.

This is the central object in the new workflow engine.  It follows the same
factory pattern used elsewhere in the toolkit: configure via Pydantic fields,
then call ``get()`` to obtain the runtime object.

Usage — from a resolved invocation (executor path)::

    from genai_tk.workflow.compiler import WorkflowCompiler
    from genai_tk.workflow.prefect.flow_factory import PrefectFlowFactory

    compiled = WorkflowCompiler().compile(invocation.workflow, invocation.values)
    factory = PrefectFlowFactory(compiled=compiled)

    # Inspect the flow (dry-run / DAG inspection)
    flow_fn = factory.get()

    # Execute inside ephemeral Prefect context
    results = factory.run()

Usage — directly from a profile name::

    factory = PrefectFlowFactory.from_profile("markdownize_docs")
    results = factory.run()

DAG parallelism
---------------
Independent steps (no shared ``wait_for`` edges) are submitted concurrently
via Prefect futures to the flow's ``ThreadPoolTaskRunner``.  Steps that declare
``wait_for`` are submitted with those futures as dependencies, so Prefect's
runtime enforces ordering while maximising concurrency.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger
from pydantic import BaseModel

from genai_tk.workflow.compiled_models import CompiledStep, CompiledWorkflow


class WorkflowExecutionError(RuntimeError):
    """Raised when a workflow step fails and ``on_failure`` is ``abort``."""


class PrefectFlowFactory(BaseModel):
    """Build and run a Prefect flow from a :class:`~genai_tk.workflow.compiled_models.CompiledWorkflow`.

    Attributes:
        compiled: The normalized workflow graph produced by :class:`WorkflowCompiler`.
        max_workers: Maximum parallel tasks for the ``ThreadPoolTaskRunner``.
    """

    compiled: CompiledWorkflow
    max_workers: int = 4

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self):
        """Build and return a Prefect ``@flow`` function without executing it.

        Returns:
            A Prefect ``Flow`` object whose name matches the compiled workflow name.
        """
        return _build_prefect_flow(self.compiled, max_workers=self.max_workers)

    def run(self) -> dict[str, Any]:
        """Execute the workflow inside an ephemeral (or configured) Prefect context.

        Returns:
            Mapping of ``step_id → result`` for all executed steps.
        """
        from genai_tk.workflow.prefect.run import ephemeral_prefect_settings

        flow_fn = self.get()
        with ephemeral_prefect_settings():
            return flow_fn()

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_profile(
        cls,
        workflow_or_profile: str,
        *,
        values: dict[str, Any] | None = None,
        max_workers: int = 4,
    ) -> PrefectFlowFactory:
        """Create a factory by resolving a workflow profile name.

        Args:
            workflow_or_profile: Workflow name or profile name (same rules as
                ``cli workflow run``).
            values: Optional CLI-style overrides merged on top of profile values.
            max_workers: Thread pool size for the generated flow.

        Returns:
            A configured :class:`PrefectFlowFactory`.
        """
        from genai_tk.workflow.compiler import WorkflowCompiler
        from genai_tk.workflow.resolver import resolve_workflow_invocation

        invocation = resolve_workflow_invocation(workflow_or_profile, cli_overrides=values or {})
        compiled = WorkflowCompiler().compile(invocation.workflow, invocation.values)
        return cls(compiled=compiled, max_workers=max_workers)


# ---------------------------------------------------------------------------
# Internal flow-building logic
# ---------------------------------------------------------------------------


def _build_prefect_flow(compiled: CompiledWorkflow, *, max_workers: int = 4):
    """Dynamically create a ``@flow`` function for the given compiled workflow."""
    from prefect import flow as prefect_flow
    from prefect.task_runners import ThreadPoolTaskRunner

    from genai_tk.workflow.compiler import topological_sort
    from genai_tk.workflow.prefect.step_factory import PrefectStepFactory

    sorted_steps = topological_sort(compiled.steps)
    step_map: dict[str, CompiledStep] = {s.id: s for s in compiled.steps}

    # Pre-build all step tasks outside the flow function so they are defined
    # at module scope relative to the flow (important for Prefect serialisation).
    step_factory = PrefectStepFactory()
    step_tasks = {step.id: step_factory.create(step) for step in compiled.steps}

    workflow_name = compiled.name
    workflow_desc = compiled.description or ""

    @prefect_flow(
        name=workflow_name,
        description=workflow_desc,
        task_runner=ThreadPoolTaskRunner(max_workers=max_workers),
    )
    def _workflow_flow() -> dict[str, Any]:
        futures: dict[str, Any] = {}
        results: dict[str, Any] = {}

        # Submit phase — wire futures as Prefect wait_for dependencies so
        # independent branches run concurrently while ordered steps wait.
        for step in sorted_steps:
            step_inputs = _prepare_inputs(step.with_, results)

            wait_futures = [futures[dep] for dep in step.wait_for if dep in futures]

            step_task = step_tasks[step.id]

            if wait_futures:
                future = step_task.submit(**step_inputs, wait_for=wait_futures)
            else:
                future = step_task.submit(**step_inputs)

            futures[step.id] = future
            logger.debug("Submitted step '{}'", step.id)

        # Collect phase — resolve futures in topological order so that
        # on_failure logic can be applied per-step.
        for step in sorted_steps:
            step_id = step.id
            on_failure = step_map[step_id].execution.on_failure
            try:
                results[step_id] = futures[step_id].result()
                logger.info("Step '{}' completed", step_id)
            except Exception as exc:
                if on_failure == "abort":
                    raise WorkflowExecutionError(f"Step '{step_id}' failed: {exc}") from exc
                logger.warning("Step '{}' failed ({}): {}", step_id, on_failure, exc)
                results[step_id] = None

        return results

    return _workflow_flow


def _prepare_inputs(with_: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``${steps.*}`` references and strip None values from step inputs.

    ``${steps.<id>.result.<field>}`` references are resolved against the
    already-collected results dict.  Other values pass through unchanged.
    """
    resolved: dict[str, Any] = {}
    for key, val in with_.items():
        resolved[key] = _resolve_step_ref(val, results)

    # Remove None values so steps can use their own parameter defaults.
    return {k: v for k, v in resolved.items() if v is not None}


def _resolve_step_ref(val: Any, results: dict[str, Any]) -> Any:
    """Replace ``${steps.<id>.result.<field>}`` with the actual result value."""
    if not isinstance(val, str):
        return val

    m = re.fullmatch(r"\$\{steps\.([^.}]+)\.result\.([^}]+)\}", val)
    if m:
        step_id, field = m.group(1), m.group(2)
        step_result = results.get(step_id)
        if step_result is None:
            return None
        if isinstance(step_result, dict):
            return step_result.get(field)
        return getattr(step_result, field, None)

    return val
