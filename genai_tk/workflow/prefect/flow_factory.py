"""PrefectFlowFactory: builds a real Prefect flow from a CompiledWorkflow.

This is the central object in the workflow engine.  It follows the same
factory pattern used elsewhere in the toolkit: configure via Pydantic fields,
then call ``get()`` to obtain the runtime object.

Usage — from a resolved invocation (executor path)::

    from genai_tk.workflow.compiler import WorkflowCompiler
    from genai_tk.workflow.prefect.flow_factory import PrefectFlowFactory

    compiled = WorkflowCompiler().compile(invocation.workflow, invocation.values)
    factory = PrefectFlowFactory(compiled=compiled)

    # Inspect the flow (dry-run / DAG inspection)
    flow_fn = factory.get()

    # Execute against the configured Prefect server
    results = factory.run()

    # Start a long-running deployment listener
    factory.serve(name="my-deployment", cron="0 2 * * *")

Usage — directly from a profile name::

    factory = PrefectFlowFactory.from_profile("markdownize_docs")
    results = factory.run()

DAG parallelism
---------------
Independent steps (no shared ``wait_for`` edges) are submitted concurrently
via Prefect futures to the flow's ``ThreadPoolTaskRunner``.  Steps that declare
``wait_for`` are submitted with those futures as dependencies, so Prefect's
runtime enforces ordering while maximising concurrency.

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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel

from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.workflow.compiled_models import CompiledStep, CompiledWorkflow

if TYPE_CHECKING:
    from prefect import Flow


class WorkflowExecutionError(RuntimeError):
    """Raised when a workflow step fails and ``on_failure`` is ``abort``."""


def _root_cause_message(exc: BaseException) -> str:
    """Walk the exception chain and return the most informative message.

    Prefect wraps the original exception through several async layers; this
    traverses ``__cause__`` / ``__context__`` to find the leaf message so that
    ``WorkflowExecutionError`` surfaces the real problem rather than a Prefect
    wrapper message.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    deepest = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        deepest = current
        current = current.__cause__ or (current.__context__ if not current.__suppress_context__ else None)
    return str(deepest)


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
        """Execute the workflow against the configured Prefect server.

        Calls :func:`~genai_tk.utils.prefect_server.prefect_server` to ensure
        the server is running (auto-starts if ``prefect.auto_start`` is true),
        then runs the flow in the current process.

        Returns:
            Mapping of ``step_id → result`` for all executed steps.
        """
        from genai_tk.utils.prefect_server import prefect_server

        server = prefect_server()
        server.ensure_running()
        server.configure_api_url()

        flow_fn = self.get()
        return flow_fn()

    def serve(self, *, name: str | None = None, **serve_kwargs: Any) -> None:
        """Start a long-running Prefect deployment listener for this workflow.

        Registers the flow as a deployment with the Prefect server and blocks,
        waiting for run requests from the UI, API, or schedules.

        Args:
            name: Deployment name (defaults to the workflow name).
            **serve_kwargs: Extra keyword arguments forwarded to ``flow.serve()``
                (e.g. ``cron``, ``interval``, ``tags``, ``pause_on_shutdown``).
        """
        from genai_tk.utils.prefect_server import prefect_server

        server = prefect_server()
        server.ensure_running()
        server.configure_api_url()

        flow_fn = self.get()
        deploy_name = name or self.compiled.name
        flow_fn.serve(
            name=deploy_name,
            parameters=self.compiled.values,
            **serve_kwargs,
        )

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
# Standalone convenience helpers
# ---------------------------------------------------------------------------


def flow_from_yaml(
    source: str | Path | dict,
    *,
    workflow_name: str | None = None,
    values: dict[str, Any] | None = None,
    max_workers: int = 4,
) -> "Flow[[], dict[str, Any]]":
    """Parse a workflow YAML definition and return a ready-to-call Prefect ``@flow``.

    This is the simplest entry-point for programmatic or notebook use: define a
    workflow inline as YAML (or load one from a file), pass optional runtime values,
    and get back a standard Prefect ``Flow`` object that you can call, inspect, or
    hand to ``flow.serve()``.

    The returned flow uses the toolkit's Prefect server singleton — call
    :func:`~genai_tk.utils.prefect_server.prefect_server` (or ``cli prefect start``)
    before invoking it if ``prefect.auto_start`` is ``false``.

    Args:
        source: One of:
            - A YAML **string** containing a ``workflows:`` block.
            - A :class:`~pathlib.Path` to a YAML file.
            - A plain ``dict`` already parsed from YAML.
        workflow_name: Which workflow to extract when the YAML defines more than
            one.  If there is only one workflow in the YAML this argument can be
            omitted.
        values: Optional parameter overrides (same as ``--set KEY=VALUE`` on the
            CLI).  Merged on top of the workflow ``defaults`` and any ``presets``.
        max_workers: Thread pool size for parallel step execution.

    Returns:
        A Prefect ``Flow`` function.

    Raises:
        ValueError: When ``workflow_name`` is required but not given, or when the
            named workflow does not exist in ``source``.

    Example:
        ```python
        from genai_tk.workflow.prefect.flow_factory import flow_from_yaml

        YAML = '''
        workflows:
          greet:
            run: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
            defaults:
              base_dir: /data/pdfs
              output_dir: /data/md
        '''

        flow = flow_from_yaml(YAML)
        flow()  # execute immediately
        ```

        With a file and runtime overrides:
        ```python
        flow = flow_from_yaml(
            Path("config/workflows/my_pipeline.yaml"),
            workflow_name="my_pipeline",
            values={"batch_size": 10},
        )
        flow()
        ```
    """
    from omegaconf import OmegaConf

    from genai_tk.workflow.compiler import WorkflowCompiler
    from genai_tk.workflow.resolver import (
        WorkflowResolutionError,
        _merge_dicts,  # noqa: PLC2701
        _to_workflow_spec,  # noqa: PLC2701
        parse_workflows_from_dict,
    )

    # --- parse source ---
    if isinstance(source, Path):
        raw_dict: dict = OmegaConf.to_container(OmegaConf.load(source), resolve=False)  # type: ignore[assignment]
    elif isinstance(source, str):
        import yaml as _yaml

        raw_dict = _yaml.safe_load(source)  # type: ignore[assignment]
    elif isinstance(source, dict):
        raw_dict = source
    else:
        raise TypeError(f"source must be a str, Path, or dict, got {type(source).__name__}")

    raw_workflows: dict = raw_dict.get("workflows", raw_dict)  # bare dict also accepted

    # --- validate workflow entries (shared logic with load_workflows) ---
    candidates = parse_workflows_from_dict(raw_workflows)

    if not candidates:
        raise ValueError("No valid workflow definitions found in source.")

    if workflow_name is None:
        if len(candidates) > 1:
            raise ValueError(f"Multiple workflows found ({', '.join(candidates)}). Pass workflow_name= to select one.")
        workflow_name = next(iter(candidates))

    if workflow_name not in candidates:
        raise WorkflowResolutionError(f"Workflow '{workflow_name}' not found. Available: {', '.join(candidates)}")

    wf = candidates[workflow_name]

    # --- resolve values (defaults → overrides) ---
    resolved_values = _merge_dicts(wf.defaults, values or {})

    # --- compile & build flow ---
    from genai_tk.workflow.registry import registry as _reg

    all_workflows = candidates
    workflow_spec = _to_workflow_spec(wf, all_workflows, _reg, extra_values=resolved_values)
    compiled = WorkflowCompiler().compile(workflow_spec, resolved_values)
    factory = PrefectFlowFactory(compiled=compiled, max_workers=max_workers)
    return factory.get()


# ---------------------------------------------------------------------------
# Manifest path & fingerprint helpers (also used by --dry-run in commands.py)
# ---------------------------------------------------------------------------


def workflow_manifest_path(workflow_name: str) -> Path:
    """Return the path to the workflow-level step-cache manifest.

    Uses ``paths.data_root`` from global config when available, falling back
    to ``~/.cache/genai_tk``.

    Args:
        workflow_name: Name of the compiled workflow.

    Returns:
        Path to the manifest JSON file.
    """
    try:
        data_root = global_config().get_dir_path("paths.data_root")
    except Exception:
        data_root = Path.home() / ".cache" / "genai_tk"
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
        from genai_tk.utils.prefect_logging import install_loguru_prefect_bridge

        install_loguru_prefect_bridge()

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

            # --- foreach fan-out ---
            if step.foreach is not None:
                # Eagerly collect dependency results so foreach.from_ref can be resolved.
                # These steps have already been submitted; waiting here is safe because
                # topological order guarantees all wait_for steps are already submitted.
                for dep in step.wait_for:
                    if dep not in results and dep in futures and futures[dep] is not None:
                        results[dep] = futures[dep].result()
                collection = _resolve_step_ref(step.foreach.from_ref, results)
                if not isinstance(collection, (list, tuple)):
                    raise WorkflowExecutionError(
                        f"Step '{step.id}' foreach.from resolved to {type(collection).__name__}, expected list."
                    )
                fan_futures = []
                for item in collection:
                    item_inputs = _prepare_inputs(step.with_, results, item_var=item)
                    fan_future = (
                        step_task.submit(**item_inputs, wait_for=wait_futures)
                        if wait_futures
                        else step_task.submit(**item_inputs)
                    )
                    fan_futures.append(fan_future)
                futures[step.id] = fan_futures  # list of futures for fan-out
                logger.debug("Submitted step '{}' (foreach × {})", step.id, len(fan_futures))
                continue

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
                # foreach fan-out: collect a list of results
                if isinstance(futures.get(step_id), list):
                    result = [f.result() for f in futures[step_id]]
                else:
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
                    root_msg = _root_cause_message(exc)
                    raise WorkflowExecutionError(f"Step '{step_id}' failed: {root_msg}") from exc
                logger.warning("Step '{}' failed ({}): {}", step_id, on_failure, exc)
                results[step_id] = None

        if manifest_dirty:
            manifest.save(manifest_path)
            logger.debug("Manifest saved to {}", manifest_path)

        # Workflow-level summary artifact (opt-in via artifacts.publish_result)
        _publish_workflow_artifact(compiled.name, sorted_steps, results)

        return results

    # With an external Prefect server, there is no need to register the flow
    # wrapper in the module namespace for discovery.  A plain @flow-decorated
    # function is sufficient for both immediate execution and flow.serve().
    from prefect import flow as prefect_flow

    return prefect_flow(
        _flow_body,
        name=compiled.name,
        task_runner=ThreadPoolTaskRunner(max_workers=max_workers),
        description=compiled.description or "",
    )


def _prepare_inputs(
    with_: dict[str, Any],
    results: dict[str, Any],
    *,
    item_var: Any = None,
) -> dict[str, Any]:
    """Resolve ``${steps.*}`` and ``${item}`` references, strip None values.

    ``${steps.<id>.result.<field>}`` references are resolved against the
    already-collected results dict.  ``${item}`` resolves to *item_var* when
    the caller is processing a ``foreach`` fan-out iteration.  Other values
    pass through unchanged.
    """
    resolved: dict[str, Any] = {}
    for key, val in with_.items():
        resolved[key] = _resolve_step_ref(val, results, item_var=item_var)

    # Remove None values so steps can use their own parameter defaults.
    return {k: v for k, v in resolved.items() if v is not None}


def _resolve_step_ref(val: Any, results: dict[str, Any], *, item_var: Any = None) -> Any:
    """Replace ``${steps.<id>.result.<field>}`` and ``${item}`` with actual values."""
    if not isinstance(val, str):
        return val

    # ${item} — current foreach iteration value
    if val == "${item}":
        return item_var

    m = re.fullmatch(r"\$\{steps\.([^.}]+)\.result\.([^}]*)\}", val)
    if m:
        step_id, field = m.group(1), m.group(2)
        step_result = results.get(step_id)
        if step_result is None:
            return None
        if not field:
            return step_result  # empty field → whole result
        if isinstance(step_result, dict):
            return step_result.get(field)
        return getattr(step_result, field, None)

    return val


def _publish_workflow_artifact(
    workflow_name: str,
    steps: list[CompiledStep],
    results: dict[str, Any],
) -> None:
    """Create a markdown summary artifact for the workflow run (best-effort)."""
    try:
        from prefect.artifacts import create_markdown_artifact
    except ImportError:
        return

    lines = [f"# Workflow: {workflow_name}", "", "| Step | Result |", "|------|--------|"]
    for step in steps:
        res = results.get(step.id)
        if res is None:
            summary = "—"
        elif isinstance(res, dict):
            parts = [f"{k}={v}" for k, v in res.items() if k != "config_name"]
            summary = ", ".join(parts[:4]) or "ok"
        else:
            summary = str(res)[:80]
        lines.append(f"| {step.id} | {summary} |")

    try:
        safe_key = re.sub(r"[^a-z0-9-]", "-", workflow_name.lower())
        create_markdown_artifact("\n".join(lines), key=f"workflow-{safe_key}")
    except Exception:
        logger.debug("Failed to create workflow summary artifact")
