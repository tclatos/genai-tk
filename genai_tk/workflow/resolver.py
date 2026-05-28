"""v2 Workflow resolver — the new single entry-point for the redesigned DSL.

This resolver handles the simplified v2 YAML format:

- Workflows live directly under ``workflows:`` (no ``definitions:`` sub-key).
- Presets are nested inside each workflow under ``presets:``.
- ``run:`` in a pipeline step auto-resolves to: a sub-workflow, a
  ``@workflow``-registered name, or a dotted Python import path.
- ``after:`` is a natural-English alias for ``wait_for:``.

The resolver produces the same :class:`~genai_tk.workflow.models.ResolvedWorkflowInvocation`
that ``execute_workflow()`` and the CLI already consume — so the compiler,
executor, and Prefect layer remain completely unchanged.

CLI usage examples::

    cli workflow run markdownize               # no preset, uses defaults
    cli workflow run markdownize/rainbow       # named preset
    cli workflow run markdownize --set base_dir=/x
    cli workflow list
    cli workflow show markdownize
"""

from __future__ import annotations

import re
from typing import Any

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError

from genai_tk.utils.config_mngr import OmegaConfig, global_config
from genai_tk.workflow.compiled_models import ArtifactSpec, ExecutionSpec, InvokeSpec, StepKind
from genai_tk.workflow.models import PipelineStep, ResolvedWorkflowInvocation, StepSpec, WorkflowDefV2, WorkflowSpec
from genai_tk.workflow.registry import WorkflowRegistry
from genai_tk.workflow.registry import registry as _global_registry


class WorkflowResolutionError(ValueError):
    """Raised when a v2 workflow or preset cannot be resolved."""


_LEGACY_KEYS = frozenset({"step_templates", "definitions", "profiles"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_workflow_invocation(
    name_or_preset: str,
    *,
    cli_overrides: dict[str, Any] | None = None,
    config: OmegaConfig | None = None,
    force: bool = False,
) -> ResolvedWorkflowInvocation:
    """Resolve a v2 ``workflow_name[/preset_name]`` string to a ``ResolvedWorkflowInvocation``.

    Args:
        name_or_preset: Either ``"workflow_name"`` or ``"workflow_name/preset_name"``.
        cli_overrides: Extra key-value overrides (highest priority).
        config: Optional config to use; defaults to :func:`global_config`.
        force: When ``True`` the ``force`` / ``force_rebuild`` flags are injected.

    Returns:
        A :class:`~genai_tk.workflow.models.ResolvedWorkflowInvocation` ready for
        :func:`~genai_tk.workflow.executor.execute_workflow`.
    """
    cfg = config if config is not None else global_config()

    workflow_name, preset_name = _parse_name(name_or_preset)
    all_workflows = load_workflows(cfg)
    reg = _global_registry

    if workflow_name not in all_workflows:
        available = ", ".join(sorted(all_workflows)) or "<none>"
        raise WorkflowResolutionError(f"Workflow '{workflow_name}' not found. Available: {available}")

    wf = all_workflows[workflow_name]

    if preset_name is not None and preset_name not in wf.presets:
        available_presets = ", ".join(sorted(wf.presets)) or "<none>"
        raise WorkflowResolutionError(
            f"Preset '{preset_name}' not found in workflow '{workflow_name}'. Available presets: {available_presets}"
        )

    preset_values = wf.presets.get(preset_name, {}) if preset_name else {}
    cli_values = cli_overrides or {}
    values = _merge_dicts(wf.defaults, preset_values, cli_values)
    if force:
        values["force"] = True

    _validate_required_params(wf, values, name_or_preset)

    workflow_spec = _v2_to_workflow_spec(wf, all_workflows, reg, extra_values=values)

    return ResolvedWorkflowInvocation(
        requested_name=name_or_preset,
        workflow_name=workflow_name,
        workflow=workflow_spec,
        profile_name=preset_name,
        values=values,
        cli_overrides=cli_values,
        force=force,
    )


def load_workflows(config: OmegaConfig | None = None) -> dict[str, WorkflowDefV2]:
    """Load all v2 workflow definitions from config.

    Reads the ``workflows:`` section directly (not ``workflows.definitions:``).
    Keys that belong to the v1 format (``step_templates``, ``definitions``,
    ``profiles``) are silently skipped so old and new configs can coexist during
    migration.

    Args:
        config: Optional config; defaults to :func:`global_config`.

    Returns:
        Mapping of workflow name → :class:`WorkflowDefV2`.
    """
    cfg = config if config is not None else global_config()
    raw = cfg.get("workflows", default={})
    if raw is None:
        return {}
    if isinstance(raw, DictConfig):
        try:
            raw = OmegaConf.to_container(raw, resolve=False)
        except InterpolationKeyError:
            raw = OmegaConf.to_container(raw, resolve=False, throw_on_missing=False)

    if not isinstance(raw, dict):
        return {}

    workflows: dict[str, WorkflowDefV2] = {}
    for name, data in raw.items():
        if name.startswith("_") or name in _LEGACY_KEYS:
            continue
        if not isinstance(data, dict):
            continue
        if "run" not in data and "pipeline" not in data:
            continue
        try:
            workflows[name] = WorkflowDefV2.model_validate({"name": name, **data})
        except Exception as exc:
            raise WorkflowResolutionError(f"Invalid workflow definition '{name}': {exc}") from exc

    # Also include Python-registered workflows (from @workflow decorator)
    for reg_entry in _global_registry.list_all():
        if reg_entry.name not in workflows:
            # Synthesise a single-step WorkflowDefV2 from the registry entry
            workflows[reg_entry.name] = WorkflowDefV2(
                name=reg_entry.name,
                description=reg_entry.description,
                run=reg_entry.dotted_path,
            )

    return workflows


def list_workflow_names(config: OmegaConfig | None = None) -> list[str]:
    """Return sorted workflow names available in the v2 format."""
    return sorted(load_workflows(config).keys())


def list_preset_names(workflow_name: str, config: OmegaConfig | None = None) -> list[str]:
    """Return sorted preset names for a given workflow."""
    workflows = load_workflows(config)
    if workflow_name not in workflows:
        return []
    return sorted(workflows[workflow_name].presets.keys())


# ---------------------------------------------------------------------------
# Public utilities
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Internal: required-param validation
# ---------------------------------------------------------------------------


def _validate_required_params(
    wf: WorkflowDefV2,
    values: dict[str, Any],
    requested_name: str,
) -> None:
    """Check that all required params declared in ``params:`` have a resolved value.

    A param is considered required when its schema entry has ``required: true``
    AND the key is not present in ``defaults`` (which would already supply a
    fallback).  Missing required params raise :class:`WorkflowResolutionError`
    with a clear message and hint before execution starts.
    """
    if not wf.params:
        return

    missing: list[str] = []
    for param_name, spec in wf.params.items():
        # spec may be a ParamSpec instance, a plain dict, or a primitive (e.g. {})
        if isinstance(spec, dict):
            required = spec.get("required", False)
        else:
            required = getattr(spec, "required", False)

        if required and param_name not in values:
            missing.append(param_name)

    if missing:
        hint = "  ".join(f"--set {p}=<value>" for p in missing)
        raise WorkflowResolutionError(
            f"Workflow '{requested_name}' is missing required parameter(s): {', '.join(missing)}.\n"
            f"Provide them via CLI:  cli workflow run {requested_name} {hint}\n"
            f"Or select a preset:   cli workflow run {requested_name}/<preset_name>\n"
            f"Available presets: {', '.join(sorted(wf.presets)) or '<none>'}"
        )


# ---------------------------------------------------------------------------
# Internal: name parsing
# ---------------------------------------------------------------------------


def _parse_name(name_or_preset: str) -> tuple[str, str | None]:
    """Split ``"workflow/preset"`` into ``("workflow", "preset")`` or ``("workflow", None)``."""
    if "/" in name_or_preset:
        parts = name_or_preset.split("/", 1)
        return parts[0].strip(), parts[1].strip()
    return name_or_preset.strip(), None


# ---------------------------------------------------------------------------
# Internal: v2 → WorkflowSpec conversion
# ---------------------------------------------------------------------------


def _v2_to_workflow_spec(
    wf: WorkflowDefV2,
    all_workflows: dict[str, WorkflowDefV2],
    reg: WorkflowRegistry,
    *,
    extra_values: dict[str, Any] | None = None,
) -> WorkflowSpec:
    """Convert a :class:`WorkflowDefV2` to the v1 :class:`WorkflowSpec` for the compiler."""
    if wf.run:
        target = _resolve_run_to_path(wf.run, all_workflows, reg, step_id="<root>")
        # Auto-wire all declared default keys and required params as ${values.KEY}
        # references so the compiler resolves them against the effective values dict.
        # This removes the need to write explicit 'with:' mappings for single-step workflows.
        auto_with = {k: f"${{values.{k}}}" for k in wf.defaults}
        # Include required params that may not have defaults
        for param_name in wf.params:
            if param_name not in auto_with:
                auto_with[param_name] = f"${{values.{param_name}}}"
        # Include any extra runtime values (e.g. force injected by --force flag)
        for extra_key in extra_values or {}:
            if extra_key not in auto_with:
                auto_with[extra_key] = f"${{values.{extra_key}}}"
        steps = [
            StepSpec(
                id="run",
                invoke=InvokeSpec(kind=StepKind.callable, target=target),
                wait_for=[],
                **{"with": auto_with},
                cache=wf.resolved_cache(),
                execution=ExecutionSpec(),
                artifacts=ArtifactSpec(),
            )
        ]
    else:
        steps = _expand_pipeline(wf.pipeline, all_workflows, reg, _ancestors=frozenset({wf.name}))

    return WorkflowSpec(
        name=wf.name,
        description=wf.description,
        defaults=wf.defaults,
        steps=steps,
    )


def _expand_pipeline(
    pipeline: list[PipelineStep],
    all_workflows: dict[str, WorkflowDefV2],
    reg: WorkflowRegistry,
    *,
    _ancestors: frozenset[str],
) -> list[StepSpec]:
    """Recursively expand a list of pipeline steps into a flat :class:`StepSpec` list.

    Steps whose ``run:`` references another workflow are inlined (sub-workflow
    expansion).  A ``terminal_map`` tracks the leaf steps of each expanded
    sub-workflow so that ``after:`` / ``wait_for:`` dependencies on the parent
    step ID resolve correctly to the sub-workflow's terminal steps.
    """
    result: list[StepSpec] = []
    terminal_map: dict[str, list[str]] = {}

    for ps in pipeline:
        is_sub_wf = ps.run in all_workflows and ps.run not in _ancestors

        if is_sub_wf:
            sub_name = ps.run
            if sub_name in _ancestors:
                raise WorkflowResolutionError(
                    f"Workflow composition cycle: {' -> '.join(sorted(_ancestors))} -> {sub_name}"
                )

            sub_wf = all_workflows[sub_name]
            if sub_wf.run:
                # Single-step workflow used as a sub-workflow
                sub_pipeline = [PipelineStep(id="run", run=sub_wf.run, **{"with": {}})]
                sub_pipeline[0].cache = sub_wf.resolved_cache()
            else:
                sub_pipeline = sub_wf.pipeline

            sub_steps = _expand_pipeline(
                sub_pipeline,
                all_workflows,
                reg,
                _ancestors=_ancestors | {sub_name},
            )

            parent_deps = _resolve_deps(ps.dependencies, terminal_map)
            step_id = ps.id
            sub_ids = {s.id for s in sub_steps}
            depended_on = {dep for s in sub_steps for dep in s.wait_for}
            leaf_ids = [s.id for s in sub_steps if s.id not in depended_on]
            terminal_map[step_id] = (
                [f"{step_id}.{sid}" for sid in leaf_ids]
                if leaf_ids
                else ([f"{step_id}.{sub_steps[-1].id}"] if sub_steps else [step_id])
            )

            for sub_step in sub_steps:
                is_root = not (set(sub_step.wait_for) & sub_ids)
                new_wait_for = [f"{step_id}.{n}" for n in sub_step.wait_for]
                if is_root and parent_deps:
                    new_wait_for = parent_deps + new_wait_for

                # Merge parent step's with_ on top of sub-step's with_
                merged_with = {**sub_step.with_, **ps.with_} if is_root else dict(sub_step.with_)

                result.append(
                    StepSpec(
                        id=f"{step_id}.{sub_step.id}",
                        invoke=sub_step.invoke,
                        wait_for=new_wait_for,
                        **{"with": merged_with},
                        cache=sub_step.cache,
                        execution=sub_step.execution,
                        artifacts=sub_step.artifacts,
                        foreach=sub_step.foreach,
                    )
                )
        else:
            # Leaf step — resolve run: to a dotted path
            target = _resolve_run_to_path(ps.run, all_workflows, reg, step_id=ps.id)
            deps = _resolve_deps(ps.dependencies, terminal_map)
            result.append(
                StepSpec(
                    id=ps.id,
                    invoke=InvokeSpec(kind=StepKind.callable, target=target),
                    wait_for=deps,
                    **{"with": dict(ps.with_)},
                    cache=ps.cache,
                    execution=ps.execution,
                    artifacts=ArtifactSpec(),
                    foreach=ps.foreach,
                )
            )

    return result


def _resolve_run_to_path(
    run_target: str,
    all_workflows: dict[str, WorkflowDefV2],
    reg: WorkflowRegistry,
    *,
    step_id: str = "?",
) -> str:
    """Resolve a ``run:`` value to a dotted Python path.

    Priority:
    1. Python registry (registered via ``@workflow`` decorator)
    2. Use as dotted path directly (compiler will auto-detect kind)

    Note: sub-workflow expansion (when ``run:`` is another workflow name) is
    handled *before* this function is called.
    """
    reg_entry = reg.get(run_target)
    if reg_entry is not None:
        return reg_entry.dotted_path
    return run_target


def _resolve_deps(
    deps: list[str],
    terminal_map: dict[str, list[str]],
) -> list[str]:
    """Replace sub-workflow parent IDs with their leaf step IDs."""
    result: list[str] = []
    for dep in deps:
        if dep in terminal_map:
            result.extend(terminal_map[dep])
        else:
            result.append(dep)
    return result


# ---------------------------------------------------------------------------
# Internal: value merging
# ---------------------------------------------------------------------------


def _merge_dicts(*parts: dict[str, Any]) -> dict[str, Any]:
    """Merge value dicts and resolve OmegaConf interpolations against global config.

    Using the global config root as the parent context allows preset values that
    contain ``${paths.*}`` or other top-level interpolations to resolve correctly.
    """
    # Build the merged values dict first (unresolved)
    merged = OmegaConf.create({})
    for part in parts:
        merged = OmegaConf.merge(merged, OmegaConf.create(part))

    # Resolve within the global config context so ${paths.*} etc. resolve correctly.
    try:
        cfg = global_config()
        # Overlay merged values onto the full config, then extract just the values
        combined = OmegaConf.merge(cfg.root, OmegaConf.create({"_merged_values_": merged}))
        resolved_section = OmegaConf.select(combined, "_merged_values_")
        resolved = OmegaConf.to_container(resolved_section, resolve=True)
    except InterpolationKeyError as exc:
        missing_key = re.search(r"Interpolation key '([^']+)' not found", str(exc))
        hint = missing_key.group(1) if missing_key else str(exc)
        raise WorkflowResolutionError(
            f"Configuration interpolation error while merging values: key '{hint}' not found."
        ) from exc
    except Exception:
        # Fall back to resolving in isolation if global config is unavailable
        try:
            resolved = OmegaConf.to_container(merged, resolve=True)
        except InterpolationKeyError as exc2:
            missing_key = re.search(r"Interpolation key '([^']+)' not found", str(exc2))
            hint = missing_key.group(1) if missing_key else str(exc2)
            raise WorkflowResolutionError(f"Configuration interpolation error: key '{hint}' not found.") from exc2
    return resolved if isinstance(resolved, dict) else {}
