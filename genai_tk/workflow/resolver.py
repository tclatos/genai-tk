"""Workflow resolver backed by standalone prefect_yaml package."""

from __future__ import annotations

from typing import Any

from prefect_yaml.models.authoring import ResolvedWorkflowInvocation, WorkflowDef
from prefect_yaml.resolver import (
    WorkflowResolutionError,
    expand_pipeline,
    load_workflows as _py_load_workflows,
    parse_cli_overrides,
    parse_workflows_from_dict,
    resolve_workflow_invocation as _py_resolve_workflow_invocation,
)


def _get_global_config_raw() -> Any:
    try:
        from genai_tk.config_mgmt.config_mngr import global_config

        return global_config().root
    except Exception:
        return None


def load_workflows(config: Any = None) -> dict[str, WorkflowDef]:
    """Load workflow definitions from global config or provided source."""
    src = config if config is not None else _get_global_config_raw()
    return _py_load_workflows(source=src)


def list_workflow_names(config: Any = None) -> list[str]:
    """Return sorted workflow names available."""
    return sorted(load_workflows(config).keys())


def list_preset_names(workflow_name: str, config: Any = None) -> list[str]:
    """Return sorted preset names for a given workflow."""
    wfs = load_workflows(config)
    if workflow_name not in wfs:
        return []
    return sorted(wfs[workflow_name].presets.keys())


def resolve_workflow_invocation(
    name_or_preset: str,
    *,
    cli_overrides: dict[str, Any] | None = None,
    config: Any = None,
    force: bool = False,
) -> ResolvedWorkflowInvocation:
    """Resolve workflow invocation using prefect_yaml resolver with global config context."""
    cfg = config if config is not None else _get_global_config_raw()
    return _py_resolve_workflow_invocation(
        name_or_preset,
        cli_overrides=cli_overrides,
        source=cfg,
        config_context=cfg,
        force=force,
    )


__all__ = [
    "WorkflowResolutionError",
    "expand_pipeline",
    "list_preset_names",
    "list_workflow_names",
    "load_workflows",
    "parse_cli_overrides",
    "parse_workflows_from_dict",
    "resolve_workflow_invocation",
]
