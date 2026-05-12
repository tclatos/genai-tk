"""Workflow and workflow-profile resolution helpers."""

from __future__ import annotations

import re
from typing import Any

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError

from genai_tk.utils.config_mngr import OmegaConfig, global_config
from genai_tk.workflow.models import (
    ResolvedWorkflowInvocation,
    WorkflowProfileSpec,
    WorkflowSpec,
)


class WorkflowResolutionError(ValueError):
    """Raised when a workflow or workflow profile cannot be resolved."""


def _extract_interpolation_context(error: Exception) -> str | None:
    """Extract interpolation key from OmegaConf error message."""
    error_str = str(error)
    match = re.search(r"Interpolation key '([^']+)' not found", error_str)
    if match:
        return match.group(1)
    return None


def _suggest_available_paths(config: OmegaConfig) -> str:
    """Suggest commonly used paths from configuration."""
    try:
        paths = config.get("paths", default={})
        if isinstance(paths, DictConfig):
            path_keys = sorted(paths.keys())
            if path_keys:
                suggestions = ", ".join(f"paths.{k}" for k in path_keys[:5])
                return f"Available paths: {suggestions}..."
    except Exception:
        pass
    return "Check your config for available paths under 'paths:' section."


def _safe_resolve_omegaconf(value: Any, config: OmegaConfig | None = None) -> Any:
    """Safely resolve OmegaConf values with better error messages."""
    try:
        if isinstance(value, DictConfig):
            return OmegaConf.to_container(value, resolve=True)
        return value
    except InterpolationKeyError as exc:
        missing_key = _extract_interpolation_context(exc)
        cfg = config or global_config()
        hint = _suggest_available_paths(cfg)
        raise WorkflowResolutionError(
            f"Configuration interpolation error: key '{missing_key}' not found. {hint}"
        ) from exc
    except Exception as exc:
        raise WorkflowResolutionError(f"Configuration resolution error: {exc}") from exc


def _config_or_global(config: OmegaConfig | None) -> OmegaConfig:
    return config if config is not None else global_config()


def _section_dict(config: OmegaConfig, key: str, *, resolve: bool) -> dict[str, Any]:
    raw = config.get(key, default={})
    if raw is None:
        return {}
    if isinstance(raw, DictConfig):
        try:
            resolved = OmegaConf.to_container(raw, resolve=resolve)
        except InterpolationKeyError as exc:
            missing_key = _extract_interpolation_context(exc)
            hint = _suggest_available_paths(config)
            raise WorkflowResolutionError(
                f"Configuration interpolation error in '{key}': key '{missing_key}' not found. {hint}"
            ) from exc
    elif isinstance(raw, dict):
        resolved = raw
    else:
        raise WorkflowResolutionError(f"Configuration section '{key}' must be a mapping, got {type(raw).__name__}")
    if not isinstance(resolved, dict):
        raise WorkflowResolutionError(f"Configuration section '{key}' must resolve to a mapping")
    return resolved


def _expand_step_templates(steps_data: list[dict], templates: dict[str, dict]) -> list[dict]:
    """Expand `ref:` fields in step definitions by merging from step templates.

    For each step with a `ref:` key, the template's fields are used as defaults;
    any field explicitly set on the step overrides the template value.
    """
    expanded = []
    for step in steps_data:
        ref = step.get("ref")
        if ref is None:
            expanded.append(step)
            continue
        if ref not in templates:
            available = ", ".join(sorted(templates)) or "<none>"
            raise WorkflowResolutionError(
                f"Step '{step.get('id', '?')}' references unknown step template '{ref}'. "
                f"Available templates: {available}"
            )
        template = templates[ref]
        # Template fields are defaults; step-level fields win.
        # For dict fields (inputs, params, outputs) merge so step overrides at key level.
        merged: dict = {**template}
        for key, value in step.items():
            if key == "ref":
                continue
            if key in ("inputs", "params", "outputs") and isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        expanded.append(merged)
    return expanded


def list_workflow_names(config: OmegaConfig | None = None) -> list[str]:
    """Return configured workflow names."""
    cfg = _config_or_global(config)
    return sorted(_section_dict(cfg, "workflows", resolve=False).keys())


def list_workflow_profile_names(config: OmegaConfig | None = None) -> list[str]:
    """Return configured workflow profile names."""
    cfg = _config_or_global(config)
    return sorted(_section_dict(cfg, "workflow_profiles", resolve=False).keys())


def list_step_template_names(config: OmegaConfig | None = None) -> list[str]:
    """Return configured step template names."""
    cfg = _config_or_global(config)
    return sorted(_section_dict(cfg, "step_templates", resolve=False).keys())


def load_workflow_spec(name: str, config: OmegaConfig | None = None) -> WorkflowSpec:
    """Load a workflow definition by name, expanding any step template references."""
    cfg = _config_or_global(config)
    workflows = _section_dict(cfg, "workflows", resolve=False)
    if name not in workflows:
        available = ", ".join(sorted(workflows)) or "<none>"
        raise WorkflowResolutionError(f"Workflow '{name}' not found. Available workflows: {available}")
    data = workflows[name]
    if not isinstance(data, dict):
        raise WorkflowResolutionError(f"Workflow '{name}' must be a mapping")
    templates = _section_dict(cfg, "step_templates", resolve=False)
    steps_data = data.get("steps", [])
    if steps_data:
        steps_data = _expand_step_templates(list(steps_data), templates)
    return WorkflowSpec.model_validate({"name": name, **data, "steps": steps_data})


def load_workflow_profile(name: str, config: OmegaConfig | None = None) -> WorkflowProfileSpec:
    """Load a workflow profile by name."""
    cfg = _config_or_global(config)
    profiles = _section_dict(cfg, "workflow_profiles", resolve=True)
    if name not in profiles:
        available = ", ".join(sorted(profiles)) or "<none>"
        raise WorkflowResolutionError(f"Workflow profile '{name}' not found. Available profiles: {available}")
    data = profiles[name]
    if not isinstance(data, dict):
        raise WorkflowResolutionError(f"Workflow profile '{name}' must be a mapping")
    return WorkflowProfileSpec.model_validate(data)


def parse_cli_overrides(values: list[str] | None) -> dict[str, Any]:
    """Parse ``--set key=value`` CLI overrides into a nested dict."""
    if not values:
        return {}

    overrides = OmegaConf.create({})
    for item in values:
        if "=" not in item:
            raise WorkflowResolutionError(f"Invalid override '{item}'. Expected KEY=VALUE format.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise WorkflowResolutionError(f"Invalid override '{item}'. Missing key before '='.")
        try:
            parsed_holder = OmegaConf.create(f"value: {raw_value}")
            parsed_value = parsed_holder.get("value")
        except Exception:
            parsed_value = raw_value
        OmegaConf.update(overrides, key, parsed_value, merge=True)

    try:
        resolved = OmegaConf.to_container(overrides, resolve=True)
    except InterpolationKeyError as exc:
        missing_key = _extract_interpolation_context(exc)
        cfg = global_config()
        hint = _suggest_available_paths(cfg)
        raise WorkflowResolutionError(
            f"Configuration interpolation error in CLI overrides: key '{missing_key}' not found. {hint}"
        ) from exc
    if not isinstance(resolved, dict):
        raise WorkflowResolutionError("CLI overrides must resolve to a mapping")
    return resolved


def _merge_dicts(*parts: dict[str, Any]) -> dict[str, Any]:
    merged = OmegaConf.create({})
    for part in parts:
        merged = OmegaConf.merge(merged, OmegaConf.create(part))
    try:
        resolved = OmegaConf.to_container(merged, resolve=True)
    except InterpolationKeyError as exc:
        missing_key = _extract_interpolation_context(exc)
        cfg = global_config()
        hint = _suggest_available_paths(cfg)
        raise WorkflowResolutionError(
            f"Configuration interpolation error while merging values: key '{missing_key}' not found. {hint}"
        ) from exc
    return resolved if isinstance(resolved, dict) else {}


def resolve_workflow_invocation(
    workflow_or_profile: str,
    *,
    profile_name: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    config: OmegaConfig | None = None,
    force: bool = False,
) -> ResolvedWorkflowInvocation:
    """Resolve a workflow invocation from a workflow name or profile name."""
    cfg = _config_or_global(config)
    workflows = set(list_workflow_names(cfg))
    profiles = set(list_workflow_profile_names(cfg))
    cli_values = cli_overrides or {}

    if profile_name is not None:
        if workflow_or_profile not in workflows:
            raise WorkflowResolutionError(
                f"Workflow '{workflow_or_profile}' not found. Use a workflow name with --profile, not a profile name."
            )
        profile = load_workflow_profile(profile_name, cfg)
        if profile.workflow != workflow_or_profile:
            raise WorkflowResolutionError(
                f"Profile '{profile_name}' targets workflow '{profile.workflow}', not '{workflow_or_profile}'."
            )
        workflow = load_workflow_spec(workflow_or_profile, cfg)
        values = _merge_dicts(workflow.defaults, profile.values, cli_values)
        return ResolvedWorkflowInvocation(
            requested_name=workflow_or_profile,
            workflow_name=workflow.name,
            workflow=workflow,
            profile_name=profile_name,
            values=values,
            step_overrides=profile.overrides,
            cli_overrides=cli_values,
            force=force,
        )

    if workflow_or_profile in workflows and workflow_or_profile in profiles:
        raise WorkflowResolutionError(
            f"'{workflow_or_profile}' matches both a workflow and a profile. Use --profile to disambiguate."
        )

    if workflow_or_profile in profiles:
        profile = load_workflow_profile(workflow_or_profile, cfg)
        workflow = load_workflow_spec(profile.workflow, cfg)
        values = _merge_dicts(workflow.defaults, profile.values, cli_values)
        return ResolvedWorkflowInvocation(
            requested_name=workflow_or_profile,
            workflow_name=workflow.name,
            workflow=workflow,
            profile_name=workflow_or_profile,
            values=values,
            step_overrides=profile.overrides,
            cli_overrides=cli_values,
            force=force,
        )

    if workflow_or_profile in workflows:
        workflow = load_workflow_spec(workflow_or_profile, cfg)
        return ResolvedWorkflowInvocation(
            requested_name=workflow_or_profile,
            workflow_name=workflow.name,
            workflow=workflow,
            values=_merge_dicts(workflow.defaults, cli_values),
            cli_overrides=cli_values,
            force=force,
        )

    available_workflows = ", ".join(sorted(workflows)) or "<none>"
    available_profiles = ", ".join(sorted(profiles)) or "<none>"
    raise WorkflowResolutionError(
        f"'{workflow_or_profile}' is neither a workflow nor a workflow profile. "
        f"Available workflows: {available_workflows}. Available profiles: {available_profiles}."
    )
