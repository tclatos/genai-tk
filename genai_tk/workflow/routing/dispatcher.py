"""Dispatcher flow: route source items to workflows and run them in parallel.

The dispatcher is the runtime counterpart of an :class:`IngestRouteTable`:

1. Split *sources* into URL items (passed through as-is) and filesystem specs
   (walked by ``resolve_sources``).
2. Classify each item against the route table — first matching rule wins —
   and bucket items by ``(workflow, params)``.
3. Fan out: one Prefect task per bucket, all submitted concurrently. Each task
   resolves its workflow through the standard engine (``PrefectFlowFactory``),
   so YAML workflows, presets and ``@workflow``-registered callables are all
   valid routing targets, and runs it as a subflow.
4. Fan in: merge bucket results into a single summary.

Routed workflows must accept ``sources: list[str]`` and ``md_output_dir: str``
(the shared ingestion contract) and may accept ``force_stage``. Avoid routing
to the same workflow name in multiple concurrent buckets when that workflow
uses workflow-level ``cache: manifest`` — its manifest file is shared per name.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from typing import Any

from loguru import logger
from prefect import flow, task
from prefect.task_runners import ThreadPoolTaskRunner

from genai_tk.workflow.force import ForceStage
from genai_tk.workflow.prefect.flow_factory import PrefectFlowFactory
from genai_tk.workflow.registry import workflow
from genai_tk.workflow.resolver import WorkflowResolutionError
from genai_tk.workflow.routing.loader import get_ingest_routes
from genai_tk.workflow.routing.models import IngestRouteTable, is_url
from genai_tk.workflow.sources import resolve_sources

_DEFAULT_MAX_WORKERS = 8


@task
def _run_workflow_task(bucket_workflow: str, bucket_values: dict[str, Any]) -> dict[str, Any]:
    """Run one routed workflow with the bucket's items as a subflow."""
    # Server lifecycle is owned by the outer execution (execute_workflow /
    # PrefectFlowFactory.run) — same convention as every other flow.
    try:
        # Full DSL semantics: YAML workflow names, presets, ${values.*} defaults.
        factory = PrefectFlowFactory.from_profile(bucket_workflow, values=bucket_values)
        result = factory.get()()
    except WorkflowResolutionError:
        # Plain dotted path — import and call directly.
        result = _import_target(bucket_workflow)(**bucket_values)
    if not isinstance(result, dict):
        return {"result": result}
    # Single-step DSL workflows return {step_id: result} — unwrap for a uniform
    # per-workflow summary contract.
    if set(result) == {"run"}:
        unwrapped = result["run"]
        return unwrapped if isinstance(unwrapped, dict) else {"result": unwrapped}
    return result


def _import_target(dotted_path: str) -> Callable[..., Any]:
    """Import and return the callable at *dotted_path*."""
    module_path, _, attr_name = dotted_path.rpartition(".")
    if not module_path:
        raise WorkflowResolutionError(f"'{dotted_path}' is neither a known workflow nor a dotted path")
    module = importlib.import_module(module_path)
    if not hasattr(module, attr_name):
        raise WorkflowResolutionError(f"Module '{module_path}' has no attribute '{attr_name}'")
    return getattr(module, attr_name)


@flow(name="ingest_dispatch", task_runner=ThreadPoolTaskRunner(max_workers=_DEFAULT_MAX_WORKERS))  # type: ignore[call-overload]
def ingest_dispatch_flow(
    sources: list[str],
    md_output_dir: str,
    *,
    routes: str | IngestRouteTable = "default",
    force_stage: str | ForceStage | None = None,
) -> dict[str, Any]:
    """Route *sources* through a route table and run the selected workflows in parallel.

    Args:
        sources: Directories, ``.zip`` archives, files, or http(s) URLs.
        md_output_dir: Common Markdown staging directory passed to every routed workflow.
        routes: Route table instance, or a table name resolvable from configuration
            (built-ins and the ``ingest_routes`` config section).
        force_stage: Optional cache-invalidation stage forwarded to routed workflows.

    Returns:
        Summary dict with per-bucket results under ``"buckets"`` and failures under
        ``"failures"``.
    """
    table = routes if isinstance(routes, IngestRouteTable) else get_ingest_routes(routes)
    logger.info("Routing {} source(s) with route table '{}' ({})", len(sources), table.name, table.fingerprint())

    items = _resolve_items(sources, md_output_dir)
    if not items:
        return {"routes": table.name, "buckets": {}, "failures": ["no source items resolved"]}

    # --- classify: first matching rule wins -------------------------------
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        workflow_name, params = table.select(item)
        key = (workflow_name, json.dumps(params, sort_keys=True))
        bucket = buckets.setdefault(key, {"workflow": workflow_name, "params": params, "items": []})
        bucket["items"].append(item)

    for (workflow_name, _params), bucket in buckets.items():
        logger.info("Bucket '{}': {} item(s) (e.g. {})", workflow_name, len(bucket["items"]), bucket["items"][:3])

    # --- fan out: one task per bucket, all concurrent ---------------------
    futures: dict[tuple[str, str], Any] = {}
    for bucket_key, bucket in buckets.items():
        values: dict[str, Any] = {
            **bucket["params"],
            "sources": bucket["items"],
            "md_output_dir": md_output_dir,
        }
        if force_stage is not None:
            values["force_stage"] = str(force_stage)
        futures[bucket_key] = _run_workflow_task.submit(bucket["workflow"], values)

    # --- fan in: merge results --------------------------------------------
    results: dict[str, Any] = {}
    failures: list[str] = []
    for (workflow_name, _params), future in futures.items():
        try:
            results[workflow_name] = future.result()
        except Exception as exc:
            logger.error("Routed workflow '{}' failed: {}", workflow_name, exc)
            failures.append(f"{workflow_name}: {exc}")

    return {
        "routes": table.name,
        "items": len(items),
        "buckets": results,
        "failures": failures,
    }


def _resolve_items(sources: list[str], md_output_dir: str) -> list[str]:
    """Split URL specs (routed as-is) from filesystem specs (walked normally)."""
    from pathlib import Path

    from genai_tk.config_mgmt.file_patterns import resolve_config_path

    specs = [sources] if isinstance(sources, str) else list(sources)
    url_items = [s for s in specs if is_url(s)]
    file_specs = [s for s in specs if not is_url(s)]

    items = list(url_items)
    if file_specs:
        cache_root = Path(resolve_config_path(md_output_dir)) / ".cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        resolved = resolve_sources(file_specs, cache_dir=cache_root, pathspecs=None)
        items.extend(str(rf.path) for rf in resolved)
    return items


@workflow(name="ingest_dispatch", description="Route sources through ingest-route rules to workflows (parallel)")
def ingest_dispatch_step(
    *,
    sources: list[str],
    md_output_dir: str,
    routes: str = "default",
    force_stage: str | None = None,
) -> dict[str, Any]:
    """Workflow-engine wrapper around :func:`ingest_dispatch_flow`."""
    return ingest_dispatch_flow(sources=sources, md_output_dir=md_output_dir, routes=routes, force_stage=force_stage)
