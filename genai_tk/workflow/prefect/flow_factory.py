"""PrefectFlowFactory re-exported and adapted from standalone prefect_yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prefect_yaml.cache.fingerprint import compute_step_fingerprint
from prefect_yaml.cache.manifest import ManifestCache, default_manifest_path
from prefect_yaml.models.compiled import CompiledWorkflow
from prefect_yaml.runtime.flow_factory import (
    PrefectFlowFactory,
    WorkflowExecutionError,
    _prepare_inputs,
    _resolve_step_ref,
    flow_from_yaml,
)

from genai_tk.config_mgmt.config_mngr import global_config


def _build_prefect_flow(workflow: CompiledWorkflow, max_workers: int = 4) -> Any:
    return PrefectFlowFactory(
        compiled=workflow,
        max_workers=max_workers,
        manifest_path=workflow_manifest_path(workflow.name),
    ).get()


def workflow_manifest_path(workflow_name: str) -> Path:
    """Return the path to the workflow-level step-cache manifest."""
    try:
        cfg = global_config()
        if hasattr(cfg, "get_dir_path"):
            data_root = cfg.get_dir_path("paths.data_root")
        else:
            data_root = Path(str(cfg.paths.data_root))
    except Exception:
        data_root = Path.home() / ".cache" / "genai_tk"
    return data_root / ".workflow_manifests" / workflow_name / "manifest.json"


__all__ = [
    "ManifestCache",
    "PrefectFlowFactory",
    "WorkflowExecutionError",
    "_build_prefect_flow",
    "_prepare_inputs",
    "_resolve_step_ref",
    "compute_step_fingerprint",
    "default_manifest_path",
    "flow_from_yaml",
    "global_config",
    "workflow_manifest_path",
]
