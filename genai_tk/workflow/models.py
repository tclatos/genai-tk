"""Pydantic authoring models for workflow YAML configuration.

These are the *parse-level* models that represent what a user writes in YAML.
They are validated by the resolver and then compiled into runtime models by
:class:`~genai_tk.workflow.compiler.WorkflowCompiler`.

New DSL summary
---------------
- ``invoke: {kind: flow, target: dotted.path}`` replaces ``uses: dotted.path``
- ``wait_for: [step_id]`` replaces ``needs: [step_id]``
- ``with: {key: value}`` replaces separate ``inputs:`` and ``params:`` blocks
- ``execution: {on_failure: abort, retries: 2, tags: [...]}`` replaces inline fields
- ``${values.key}`` replaces ``${profile.key}`` in step ``with:`` values
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

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
    "ResolvedWorkflowInvocation",
    "StepKind",
    "StepSpec",
    "StepTemplateSpec",
    "WorkflowProfileSpec",
    "WorkflowSpec",
]


class StepTemplateSpec(BaseModel):
    """Reusable step definition referenced by ``ref: <name>`` in workflow steps.

    Any field set in the referencing step overrides the template value.  For
    the ``with`` dict the merge happens at key level — the step adds or
    replaces individual keys while the rest come from the template.
    """

    invoke: InvokeSpec = Field(default_factory=InvokeSpec)
    wait_for: list[str] = Field(default_factory=list)
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")
    cache: CacheSpec = Field(default_factory=CacheSpec)
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    artifacts: ArtifactSpec = Field(default_factory=ArtifactSpec)

    model_config = {"populate_by_name": True}


class StepSpec(BaseModel):
    """One declarative step in a workflow.

    Set ``ref`` to inherit from a named step template defined in ``step_templates:``.
    Any field set directly on the step overrides the template value.
    """

    id: str
    ref: str | None = None
    invoke: InvokeSpec = Field(default_factory=InvokeSpec)
    wait_for: list[str] = Field(default_factory=list)
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")
    cache: CacheSpec = Field(default_factory=CacheSpec)
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    artifacts: ArtifactSpec = Field(default_factory=ArtifactSpec)
    foreach: ForeachSpec | None = None

    model_config = {"populate_by_name": True}


class WorkflowSpec(BaseModel):
    """Workflow definition loaded from YAML."""

    name: str
    description: str = ""
    defaults: dict[str, Any] = Field(default_factory=dict)
    steps: list[StepSpec] = Field(default_factory=list)


class WorkflowProfileSpec(BaseModel):
    """Reusable workflow invocation profile for CLI use."""

    workflow: str
    values: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)


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
