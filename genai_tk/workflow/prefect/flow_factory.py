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

_PrefectFlowBuilder
-------------------
Dynamic ``@flow``-decorated functions created inside another function can cause
Prefect serialisation / discovery issues because the resulting function is not
importable by dotted path.  ``_PrefectFlowBuilder`` assigns a stable
``__name__`` and registers the wrapper as an attribute of this module before
applying ``@flow``, making every generated flow properly discoverable.

Workflow-level manifest caching
--------------------------------
When a step declares ``cache: {backend: manifest}`` (or ``hybrid``) in its
YAML spec, the engine computes a deterministic fingerprint of the step's
resolved inputs and checks :class:`~genai_tk.workflow.flow_cache.manifest.ManifestCache`
before submitting the step task.  Fresh steps are skipped entirely; stale or
new steps are executed and their results recorded in the manifest.
The ``force`` / ``force_rebuild`` values flag bypasses this check.
"""

from __future__ import annotations

import json
import re
import sys
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger
from pydantic import BaseModel
from upath import UPath

from genai_tk.workflow.compiled_models import CompiledStep, CompiledWorkflow

if TYPE_CHECKING:
    from prefect import Flow


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

    def get(self) -> Flow[[], dict[str, Any]]:
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
# _PrefectFlowBuilder: register wrapper in module namespace before @flow
# ---------------------------------------------------------------------------


class _PrefectFlowBuilder:
    """Create properly-named, module-registered Prefect flows.

    Prefect requires that flow functions be importable by dotted path for
    deployment and serialisation.  Dynamically-created functions defined inside
    another function body are not discoverable.  This class assigns a stable
    ``__name__`` / ``__qualname__`` to the wrapper and registers it as an
    attribute of this module before applying ``@flow``.
    """

    def build(
        self,
        name: str,
        fn: Callable[[], dict[str, Any]],
        **flow_kwargs: Any,
    ) -> Flow[[], dict[str, Any]]:
        """Wrap *fn* as a named, module-registered Prefect flow.

        Args:
            name: Human-readable flow name (also used as Python identifier).
            fn: The inner callable implementing the flow logic.
            **flow_kwargs: Extra keyword arguments forwarded to ``@flow``.

        Returns:
            A Prefect ``Flow`` object.
        """
        from prefect import flow as prefect_flow

        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

        def wrapper() -> dict[str, Any]:
            return fn()

        wrapper.__name__ = safe_name
        wrapper.__qualname__ = safe_name
        wrapper.__module__ = __name__

        # Register in this module so Prefect can resolve it by import path.
        setattr(sys.modules[__name__], safe_name, wrapper)

        return prefect_flow(wrapper, name=name, **flow_kwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]


# ---------------------------------------------------------------------------
# Manifest path & fingerprint helpers (also used by --dry-run in commands.py)
# ---------------------------------------------------------------------------


def workflow_manifest_path(workflow_name: str) -> UPath:
    """Return the path to the workflow-level step-cache manifest.

    Uses ``paths.data_root`` from global config when available, falling back
    to ``~/.cache/genai_tk``.

    Args:
        workflow_name: Name of the compiled workflow.

    Returns:
        Path to the manifest JSON file.
    """
    try:
        from genai_tk.utils.config_mngr import global_config

        data_root = UPath(str(global_config().paths.data_root))
    except Exception:
        data_root = UPath.home() / ".cache" / "genai_tk"
    return data_root / ".workflow_manifests" / workflow_name / "manifest.json"


def compute_step_fingerprint(step_id: str, step_inputs: dict[str, Any]) -> str:
    """Compute a deterministic fingerprint for a step invocation.

    The fingerprint is an xxh3_64 hash of the step ID and its resolved
    input values.  Any change in inputs invalidates the cache for that step.

    Args:
        step_id: Unique step identifier.
        step_inputs: Fully-resolved input dict passed to the step callable.
            Keys listed in :data:`FINGERPRINT_EXCLUDE_KEYS` are ignored so that
            control flags (``force_rebuild``, ``delete_first`` …) do not
            produce a different fingerprint between force and normal runs.

    Returns:
        A 16-character hex string.
    """
    from genai_tk.utils.hashing import buffer_digest

    stable_inputs = {k: v for k, v in step_inputs.items() if k not in FINGERPRINT_EXCLUDE_KEYS}
    payload = json.dumps({"id": step_id, "inputs": stable_inputs}, sort_keys=True, default=str)
    return buffer_digest(payload.encode())


#: Input keys that control *how* a step runs rather than *what* it produces.
#: Excluded from the manifest fingerprint so that a force-rebuilt result
#: remains usable on subsequent non-force runs.
FINGERPRINT_EXCLUDE_KEYS: frozenset[str] = frozenset(
    {
        "force",
        "force_rebuild",
        "delete_first",
    }
)


# ---------------------------------------------------------------------------
# Internal flow-building logic
# ---------------------------------------------------------------------------


def _build_prefect_flow(
    compiled: CompiledWorkflow,
    *,
    max_workers: int = 4,
) -> Flow[[], dict[str, Any]]:
    """Dynamically create a ``@flow`` function for the given compiled workflow."""
    from prefect.task_runners import ThreadPoolTaskRunner

    from genai_tk.workflow.compiler import topological_sort
    from genai_tk.workflow.flow_cache.manifest import ManifestCache
    from genai_tk.workflow.prefect.step_factory import PrefectStepFactory

    sorted_steps = topological_sort(compiled.steps)
    step_map: dict[str, CompiledStep] = {s.id: s for s in compiled.steps}
    force: bool = bool(compiled.values.get("force") or compiled.values.get("force_rebuild"))

    # Pre-build all step tasks outside the flow function so they are defined
    # at module scope relative to the flow (important for Prefect serialisation).
    step_factory = PrefectStepFactory()
    step_tasks = {step.id: step_factory.create(step) for step in compiled.steps}

    manifest_path = workflow_manifest_path(compiled.name)

    def _flow_body() -> dict[str, Any]:
        manifest = ManifestCache.load(manifest_path)
        futures: dict[str, Any] = {}
        results: dict[str, Any] = {}
        step_fingerprints: dict[str, str] = {}

        # Submit phase — skip cached steps immediately; submit live steps
        # with their Prefect wait_for dependencies so independent branches
        # run concurrently while ordered steps wait.
        for step in sorted_steps:
            step_inputs = _prepare_inputs(step.with_, results)
            fp = compute_step_fingerprint(step.id, step_inputs)
            step_fingerprints[step.id] = fp

            uses_manifest = step.cache.backend in ("manifest", "hybrid")
            if uses_manifest and not force and manifest.is_fresh(step.id, fingerprint=fp):
                cached_result = manifest.get_output(step.id, "result")
                logger.info("Step '{}' SKIPPED (cached, fingerprint={})", step.id, fp[:8])
                results[step.id] = cached_result
                futures[step.id] = None  # sentinel: already resolved
                continue

            wait_futures = [futures[dep] for dep in step.wait_for if dep in futures and futures[dep] is not None]
            step_task = step_tasks[step.id]
            future = (
                step_task.submit(**step_inputs, wait_for=wait_futures)
                if wait_futures
                else step_task.submit(**step_inputs)
            )
            futures[step.id] = future
            logger.debug("Submitted step '{}'", step.id)

        # Collect phase — resolve futures in topological order so that
        # on_failure logic can be applied per-step; record fresh results.
        manifest_dirty = False
        for step in sorted_steps:
            step_id = step.id
            if futures.get(step_id) is None and step_id in results:
                # Already resolved from cache in the submit phase.
                continue

            on_failure = step_map[step_id].execution.on_failure
            try:
                result = futures[step_id].result()
                results[step_id] = result
                logger.info("Step '{}' completed", step_id)

                uses_manifest = step.cache.backend in ("manifest", "hybrid")
                if uses_manifest:
                    manifest.record_success(
                        step_id,
                        step_fingerprints[step_id],
                        outputs={"result": result},
                    )
                    manifest_dirty = True

            except Exception as exc:
                if on_failure == "abort":
                    raise WorkflowExecutionError(f"Step '{step_id}' failed: {exc}") from exc
                logger.warning("Step '{}' failed ({}): {}", step_id, on_failure, exc)
                results[step_id] = None

        if manifest_dirty:
            manifest.save(manifest_path)
            logger.debug("Manifest saved to {}", manifest_path)

        return results

    return _PrefectFlowBuilder().build(
        name=compiled.name,
        fn=_flow_body,
        task_runner=ThreadPoolTaskRunner(max_workers=max_workers),
        description=compiled.description or "",
    )


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
