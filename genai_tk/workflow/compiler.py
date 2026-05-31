"""Workflow compiler: transforms authoring-level WorkflowSpec into CompiledWorkflow.

The compiler is the bridge between the YAML authoring layer and the Prefect
execution layer.  It:

- validates all step IDs are unique;
- validates all ``wait_for`` references resolve to real steps;
- validates the step graph is acyclic;
- auto-detects ``StepKind`` for steps where ``invoke.kind`` is ``callable``;
- resolves ``${values.*}`` placeholders in each step's ``with`` dict;
- resolves ``${paths.*}`` and other OmegaConf-style strings against global config;
- produces a :class:`~genai_tk.workflow.compiled_models.CompiledWorkflow` ready
  for :class:`~genai_tk.workflow.prefect.flow_factory.PrefectFlowFactory`.
"""

from __future__ import annotations

import importlib
import re
from typing import Any

from loguru import logger

from genai_tk.workflow.compiled_models import (
    ArtifactSpec,
    CacheSpec,
    CompiledStep,
    CompiledWorkflow,
    ExecutionSpec,
    InvokeSpec,
    StepKind,
)
from genai_tk.workflow.models import StepSpec, WorkflowSpec


class WorkflowCompilationError(ValueError):
    """Raised when a workflow cannot be compiled."""


class WorkflowCompiler:
    """Compile a :class:`WorkflowSpec` + resolved values into a :class:`CompiledWorkflow`.

    Example:
        ```python
        from genai_tk.workflow.compiler import WorkflowCompiler
        from genai_tk.workflow.resolver import resolve_workflow_invocation

        invocation = resolve_workflow_invocation("my_profile")
        compiled = WorkflowCompiler().compile(invocation.workflow, invocation.values)
        ```
    """

    def compile(self, spec: WorkflowSpec, values: dict[str, Any]) -> CompiledWorkflow:
        """Compile a workflow spec and its resolved values into a CompiledWorkflow.

        Args:
            spec: Authoring-level workflow definition (produced by the resolver).
            values: Fully merged runtime values (workflow defaults + profile + CLI).

        Returns:
            A :class:`CompiledWorkflow` ready for :class:`PrefectFlowFactory`.
        """
        self._validate_step_ids(spec.steps)
        self._validate_wait_for_refs(spec.steps)
        self._validate_dag(spec.steps)

        compiled_steps = [self._compile_step(step, values) for step in spec.steps]
        logger.debug("Compiled workflow '{}': {} steps", spec.name, len(compiled_steps))

        return CompiledWorkflow(
            name=spec.name,
            description=spec.description,
            steps=compiled_steps,
            values=values,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compile_step(self, step: StepSpec, values: dict[str, Any]) -> CompiledStep:
        invoke = self._resolve_invoke(step)
        resolved_with = _resolve_values_dict(step.with_, values)

        return CompiledStep(
            id=step.id,
            invoke=invoke,
            wait_for=list(step.wait_for),
            **{"with": resolved_with},
            execution=ExecutionSpec(**step.execution.model_dump()),
            cache=CacheSpec(**step.cache.model_dump()),
            artifacts=ArtifactSpec(**step.artifacts.model_dump()),
            foreach=step.foreach,
            inline=step.inline,
        )

    def _resolve_invoke(self, step: StepSpec) -> InvokeSpec:
        """Determine the final InvokeSpec, auto-detecting kind from the target object."""
        if not step.invoke.target:
            raise WorkflowCompilationError(
                f"Step '{step.id}' has no invoke.target.  Set 'invoke: {{{{target: my.module.callable}}}}' in the YAML."
            )

        if step.invoke.kind != StepKind.callable:
            # Kind explicitly provided — trust it.
            return step.invoke

        # Auto-detect: try to import the target and inspect its type.
        try:
            obj = _import_target(step.invoke.target)
        except Exception:
            # Can't import at compile time; leave as callable for runtime.
            return InvokeSpec(kind=StepKind.callable, target=step.invoke.target)

        try:
            from prefect.flows import Flow as PrefectFlow
            from prefect.tasks import Task as PrefectTask

            if isinstance(obj, PrefectFlow):
                return InvokeSpec(kind=StepKind.flow, target=step.invoke.target)
            if isinstance(obj, PrefectTask):
                return InvokeSpec(kind=StepKind.task, target=step.invoke.target)
        except ImportError:
            pass

        return InvokeSpec(kind=StepKind.callable, target=step.invoke.target)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_step_ids(self, steps: list[StepSpec]) -> None:
        seen: set[str] = set()
        for step in steps:
            if step.id in seen:
                raise WorkflowCompilationError(f"Duplicate step ID: '{step.id}'")
            seen.add(step.id)

    def _validate_wait_for_refs(self, steps: list[StepSpec]) -> None:
        step_ids = {s.id for s in steps}
        for step in steps:
            for dep in step.wait_for:
                if dep not in step_ids:
                    raise WorkflowCompilationError(f"Step '{step.id}' has wait_for dependency on unknown step '{dep}'.")

    def _validate_dag(self, steps: list[StepSpec]) -> None:
        """Kahn's algorithm to detect cycles in the step dependency graph."""
        in_degree = {s.id: len(s.wait_for) for s in steps}
        queue = [s.id for s, deg in zip(steps, in_degree.values(), strict=False) if deg == 0]
        visited = 0

        while queue:
            current = queue.pop(0)
            visited += 1
            for s in steps:
                if current in s.wait_for:
                    in_degree[s.id] -= 1
                    if in_degree[s.id] == 0:
                        queue.append(s.id)

        if visited != len(steps):
            raise WorkflowCompilationError("Cycle detected in workflow step dependencies.")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def topological_sort(steps: list[CompiledStep]) -> list[CompiledStep]:
    """Return steps in dependency order (Kahn's algorithm)."""
    step_map = {s.id: s for s in steps}
    in_degree = {s.id: len(s.wait_for) for s in steps}
    queue = [step_map[sid] for sid, deg in in_degree.items() if deg == 0]
    ordered: list[CompiledStep] = []

    while queue:
        current = queue.pop(0)
        ordered.append(current)
        for s in steps:
            if current.id in s.wait_for:
                in_degree[s.id] -= 1
                if in_degree[s.id] == 0:
                    queue.append(step_map[s.id])

    return ordered


def _import_target(dotted_path: str) -> Any:
    """Import a Python object from a dotted path."""
    module_path, _, attr_name = dotted_path.rpartition(".")
    if not module_path:
        raise WorkflowCompilationError(f"Invalid dotted path '{dotted_path}': must be module.attr")
    module = importlib.import_module(module_path)
    if not hasattr(module, attr_name):
        raise WorkflowCompilationError(f"Module '{module_path}' has no attribute '{attr_name}'")
    return getattr(module, attr_name)


def _resolve_values_dict(d: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``${values.*}`` placeholders in a dict of step inputs."""
    return {k: _resolve_value(v, values) for k, v in d.items()}


def _resolve_value(val: Any, values: dict[str, Any]) -> Any:
    """Recursively resolve ``${values.*}`` and OmegaConf ``${paths.*}`` in a value."""
    if isinstance(val, str):
        # Full replacement: "${values.key}" → the value itself (preserves type)
        m = re.fullmatch(r"\$\{values\.([^}]+)\}", val)
        if m:
            return values.get(m.group(1))

        # Inline substitution: "prefix/${values.key}/suffix" → string result
        if "${values." in val:

            def _subst(match: re.Match) -> str:
                v = values.get(match.group(1), "")
                return str(v) if v is not None else ""

            return re.sub(r"\$\{values\.([^}]+)\}", _subst, val)

        # Step output references — kept as-is; resolved at flow runtime.
        if val.startswith("${steps."):
            return val

        # OmegaConf-style interpolations (e.g. ``${paths.data_root}``) —
        # resolve against the global config.
        if "${" in val:
            return _resolve_omegaconf_string(val)

    elif isinstance(val, dict):
        return {k: _resolve_value(v, values) for k, v in val.items()}
    elif isinstance(val, list):
        return [_resolve_value(item, values) for item in val]

    return val


def _resolve_omegaconf_string(val: str) -> Any:
    """Resolve a string that may contain OmegaConf interpolations against global config."""
    from omegaconf import OmegaConf

    from genai_tk.utils.config_mngr import get_raw_config

    cfg = get_raw_config()
    try:
        tmp = OmegaConf.create({"_tmp": val})
        merged = OmegaConf.merge(cfg, tmp)
        resolved = OmegaConf.select(merged, "_tmp")
        return resolved if resolved is not None else val
    except Exception:
        return val
