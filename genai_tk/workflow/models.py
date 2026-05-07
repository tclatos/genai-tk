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


class StepSpec(BaseModel):
    """One declarative step in a workflow."""

    id: str
    uses: str
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
