"""Pydantic models for the Workflow DSL.

Authoring models (what you write in YAML):  :class:`WorkflowDefV2` → :class:`PipelineStep`
Intermediate representation (used by compiler/executor):  :class:`WorkflowSpec` → :class:`StepSpec`

The resolver converts authoring models to the intermediate representation before
handing off to the compiler and executor, which produce
:class:`~genai_tk.workflow.compiled_models.CompiledWorkflow`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

# Re-export shared spec types from compiled_models so callers can import from one place.
from genai_tk.workflow.compiled_models import (
    ArtifactSpec,
    CacheSpec,
    ExecutionSpec,
    ForeachSpec,
    InvokeSpec,
    StepKind,
)

__all__ = [
    "ArtifactSpec",
    "CacheSpec",
    "ExecutionSpec",
    "ForeachSpec",
    "InvokeSpec",
    "ParamSpec",
    "PipelineStep",
    "ResolvedWorkflowInvocation",
    "StepKind",
    "StepSpec",
    "WorkflowDefV2",
    "WorkflowSpec",
]


# ---------------------------------------------------------------------------
# Authoring models — what a user writes in YAML
# ---------------------------------------------------------------------------


class ParamSpec(BaseModel):
    """Declarative parameter schema entry (optional, used for docs + validation)."""

    required: bool = False
    default: Any = None
    description: str = ""

    model_config = {"extra": "allow"}


class PipelineStep(BaseModel):
    """One step inside a ``pipeline:`` list.

    Set ``run:`` to a dotted Python path, a registry name, or another workflow
    name.  ``after:`` is a readable alias for ``wait_for:``.
    """

    id: str
    run: str
    after: list[str] = Field(default_factory=list)
    wait_for: list[str] = Field(default_factory=list)
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")
    cache: CacheSpec = Field(default_factory=CacheSpec)
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    foreach: ForeachSpec | None = None

    model_config = {"populate_by_name": True}

    @property
    def dependencies(self) -> list[str]:
        """Combined ``after`` + ``wait_for`` — all predecessor step IDs."""
        return list(dict.fromkeys(self.after + self.wait_for))


class WorkflowDefV2(BaseModel):
    """A workflow definition.

    A workflow is one of:

    - A **single-step shorthand** using ``run:`` (points to a dotted path or a
      registry name).
    - A **multi-step pipeline** using ``pipeline:`` (list of
      :class:`PipelineStep`).

    **Presets** are named parameter sets nested inside the workflow.  Reference
    them with ``workflow_name/preset_name`` on the CLI::

        workflows:
          markdownize:
            run: genai_tk.workflow.prefect.flows...markdownize_flow
            defaults:
              batch_size: 5
            presets:
              rainbow:
                base_dir: "${paths.rainbow_pdf}"

    """

    name: str
    description: str = ""

    # Single-step shorthand
    run: str | None = None
    # Multi-step DAG
    pipeline: list[PipelineStep] = Field(default_factory=list)

    # Cache policy for single-step workflows (ignored when pipeline is used)
    cache: CacheSpec | str = Field(default_factory=CacheSpec)

    # Workflow-level default values for ``${values.*}`` placeholders
    defaults: dict[str, Any] = Field(default_factory=dict)

    # Optional declarative param schema (for documentation + validation)
    params: dict[str, Any] = Field(default_factory=dict)

    # Named presets (concrete value sets, referenced as ``name/preset``)
    presets: dict[str, dict[str, Any]] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _require_run_or_pipeline(self) -> WorkflowDefV2:
        if self.run is None and not self.pipeline:
            raise ValueError(
                f"Workflow '{self.name}' must specify either 'run:' (single-step) or 'pipeline:' (multi-step)."
            )
        if self.run is not None and self.pipeline:
            raise ValueError(f"Workflow '{self.name}' cannot specify both 'run:' and 'pipeline:'.")
        return self

    def resolved_cache(self) -> CacheSpec:
        """Return the CacheSpec, accepting a shorthand string like ``'manifest'``."""
        if isinstance(self.cache, str):
            return CacheSpec(backend=self.cache)
        return self.cache  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Intermediate representation — produced by resolver, consumed by compiler
# ---------------------------------------------------------------------------


class StepSpec(BaseModel):
    """One normalized step in a workflow, ready for compilation."""

    id: str
    invoke: InvokeSpec = Field(default_factory=InvokeSpec)
    wait_for: list[str] = Field(default_factory=list)
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")
    cache: CacheSpec = Field(default_factory=CacheSpec)
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    artifacts: ArtifactSpec = Field(default_factory=ArtifactSpec)
    foreach: ForeachSpec | None = None

    model_config = {"populate_by_name": True}


class WorkflowSpec(BaseModel):
    """Workflow definition in intermediate representation."""

    name: str
    description: str = ""
    defaults: dict[str, Any] = Field(default_factory=dict)
    steps: list[StepSpec] = Field(default_factory=list)


class ResolvedWorkflowInvocation(BaseModel):
    """Resolved workflow plus effective runtime values."""

    requested_name: str
    workflow_name: str
    workflow: WorkflowSpec
    profile_name: str | None = None
    values: dict[str, Any] = Field(default_factory=dict)
    step_overrides: dict[str, Any] = Field(default_factory=dict)
    cli_overrides: dict[str, Any] = Field(default_factory=dict)
    force: bool = False
