"""Workflow models re-exported from standalone prefect_yaml package."""

from __future__ import annotations

from prefect_yaml.models import (
    ArtifactSpec,
    CacheSpec,
    ExecutionSpec,
    ForeachSpec,
    InvokeSpec,
    ParamSpec,
    PipelineStep,
    ResolvedWorkflowInvocation,
    StepKind,
    StepSpec,
    WorkflowDef,
    WorkflowSpec,
)

WorkflowDefV2 = WorkflowDef

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
    "WorkflowDef",
    "WorkflowDefV2",
    "WorkflowSpec",
]
