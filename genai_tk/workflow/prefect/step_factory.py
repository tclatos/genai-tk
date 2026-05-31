"""PrefectStepFactory: resolves a CompiledStep into an executable Prefect task.

Each step kind maps to a Prefect execution primitive:

- ``flow``       → wrapped as a ``@task`` that calls the flow as a subflow
- ``task``       → wrapped as a ``@task`` that calls the underlying function
- ``callable``   → wrapped as a ``@task`` that calls the plain callable
- ``deployment`` → wrapped as a ``@task`` that triggers a Prefect deployment
- ``factory``    → wrapped as a ``@task`` that calls ``factory.create()(**kwargs)``

Note: ``flow``, ``task``, and ``callable`` kinds are all executed identically
at runtime (import the dotted target, call it).  The ``kind`` field is kept
for documentation purposes and to allow future differentiation.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel

from genai_tk.workflow.compiled_models import CompiledStep, StepKind

if TYPE_CHECKING:
    from prefect import Task


class PrefectStepFactory(BaseModel):
    """Build a Prefect task for a single :class:`CompiledStep`.

    Example:
        ```python
        factory = PrefectStepFactory()
        step_task = factory.create(compiled_step)
        future = step_task.submit(**step.with_)
        ```
    """

    def create(self, step: CompiledStep) -> Task:
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


def _make_step_task(step: CompiledStep) -> Task:
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

        # When inline=True and the target is a @flow-decorated function,
        # call .fn to bypass subflow creation — internal tasks run directly
        # under the parent workflow flow.
        if step.inline:
            try:
                from prefect.flows import Flow as PrefectFlow

                if isinstance(callable_obj, PrefectFlow):
                    callable_obj = callable_obj.fn
            except ImportError:
                pass

        def _fn(**kwargs: Any) -> Any:
            return callable_obj(**kwargs)

    # Give each task a unique, human-readable name and qualname so Prefect can
    # distinguish them in the dashboard and task-runner hash.
    _fn.__name__ = step_id
    _fn.__qualname__ = f"workflow.{step_id}"
    _fn.__module__ = __name__

    cache_policy = _cache_policy_for_step(step)

    return prefect_task(
        _fn,
        name=step_id,
        retries=retries,
        retry_delay_seconds=retry_delay,
        tags=tags or None,
        cache_policy=cache_policy,
    )


def _cache_policy_for_step(step: CompiledStep) -> Any:
    """Return a Prefect ``cache_policy`` for the given step spec.

    ``manifest`` and ``hybrid`` backends are handled at the engine level
    (in ``flow_factory._build_prefect_flow``), so no Prefect-level policy is
    needed for them.  ``prefect_result`` uses Prefect's ``INPUTS`` policy so
    the task result is cached when inputs hash matches a previous run.
    """
    backend = step.cache.backend
    if backend == "prefect_result" or backend == "hybrid":
        from prefect.cache_policies import INPUTS

        return INPUTS
    # none / manifest: no Prefect-level cache (manifest handled by the engine)
    from prefect.cache_policies import NO_CACHE

    return NO_CACHE
