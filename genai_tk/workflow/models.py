"""Pydantic models for workflow YAML configuration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CacheSpec(BaseModel):
    """Execution reuse policy for a workflow step."""

    mode: Literal["off", "inputs", "inputs_and_params", "inputs_params_and_code"] = "off"
    allow_prefect_result_reuse: bool = False


class MaterializationSpec(BaseModel):
    """Persistence policy for a workflow step's outputs."""

    mode: Literal["none", "metadata_only", "payload_only", "full"] = "metadata_only"
    format: Literal["auto", "json", "jsonl", "parquet", "directory"] = "auto"
    target: str | None = None


class StepTemplateSpec(BaseModel):
    """Reusable step definition that can be referenced by workflow steps via `ref:`.

    A step template captures the `uses`, `inputs`, `params`, and other execution
    fields shared across multiple steps. Individual steps can override any field.
    """

    uses: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    cache: CacheSpec = Field(default_factory=CacheSpec)
    materialization: MaterializationSpec = Field(default_factory=MaterializationSpec)
    concurrency: Literal["auto", "serial", "parallel"] = "auto"
    on_failure: Literal["abort", "skip", "continue"] = "abort"


class StepSpec(BaseModel):
    """One declarative step in a workflow.

    Set `ref` to inherit from a named step template defined in `step_templates:`.
    Set `uses_workflow` to inline another workflow as a sub-workflow (mutually
    exclusive with `uses` and `ref`).
    Any field set directly on the step overrides the template value.
    """

    id: str
    ref: str | None = None
    uses: str = ""
    uses_workflow: str | None = None
    needs: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    cache: CacheSpec = Field(default_factory=CacheSpec)
    materialization: MaterializationSpec = Field(default_factory=MaterializationSpec)
    concurrency: Literal["auto", "serial", "parallel"] = "auto"
    on_failure: Literal["abort", "skip", "continue"] = "abort"


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
