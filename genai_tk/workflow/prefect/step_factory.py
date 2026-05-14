"""PrefectStepFactory: resolves a CompiledStep into an executable Prefect task.

Each step kind maps to a Prefect execution primitive:

- ``flow``       → wrapped as a ``@task`` that calls the flow as a subflow
- ``task``       → wrapped as a ``@task`` that calls the underlying function
- ``callable``   → wrapped as a ``@task`` that calls the plain callable
- ``deployment`` → wrapped as a ``@task`` that triggers a Prefect deployment
- ``factory``    → wrapped as a ``@task`` that calls ``factory.create()(**kwargs)``
"""

from __future__ import annotations

import importlib
from typing import Any

from loguru import logger
from pydantic import BaseModel

from genai_tk.workflow.compiled_models import CompiledStep, StepKind


class PrefectStepFactory(BaseModel):
    """Build a Prefect task for a single :class:`CompiledStep`.

    Example:
        ```python
        factory = PrefectStepFactory()
        step_task = factory.create(compiled_step)
        future = step_task.submit(**step.with_)
        ```
    """

    def create(self, step: CompiledStep):
        """Return a Prefect ``@task``-decorated callable for the given step.

        Args:
            step: Fully compiled step with resolved invoke spec and execution policy.

        Returns:
            A Prefect task object that can be called or submitted.
        """
        return _make_step_task(step)


# ---------------------------------------------------------------------------
# Internal factory helpers
# ---------------------------------------------------------------------------


def _import_target(dotted_path: str) -> Any:
    module_path, _, attr_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid dotted path '{dotted_path}'")
    module = importlib.import_module(module_path)
    if not hasattr(module, attr_name):
        raise AttributeError(f"Module '{module_path}' has no attribute '{attr_name}'")
    return getattr(module, attr_name)


def _make_step_task(step: CompiledStep):
    """Create a ``@task``-wrapped callable for a compiled step."""
    from prefect import task as prefect_task

    step_id = step.id
    target = step.invoke.target
    kind = step.invoke.kind
    retries = step.execution.retries
    retry_delay = step.execution.retry_delay_seconds
    tags = list(step.execution.tags or [])

    if kind == StepKind.deployment:
        def _fn(**kwargs: Any) -> Any:
            from prefect.deployments import run_deployment

            logger.info("Triggering deployment '{}' for step '{}'", target, step_id)
            return run_deployment(target, parameters=kwargs)

    elif kind == StepKind.factory:
        def _fn(**kwargs: Any) -> Any:
            factory_cls = _import_target(target)
            instance = factory_cls(**kwargs)
            return instance.get() if hasattr(instance, "get") else instance.create()

    else:
        # flow / task / callable — all resolved the same way at runtime.
        callable_obj = _import_target(target)

        def _fn(**kwargs: Any) -> Any:
            return callable_obj(**kwargs)

    # Give each task a unique, human-readable name and qualname so Prefect can
    # distinguish them in the dashboard and task-runner hash.
    _fn.__name__ = step_id
    _fn.__qualname__ = f"workflow.{step_id}"
    _fn.__module__ = __name__

    return prefect_task(
        _fn,
        name=step_id,
        retries=retries,
        retry_delay_seconds=retry_delay,
        tags=tags or None,
    )
