"""Workflow registry re-exported from standalone prefect_yaml package."""

from __future__ import annotations

from prefect_yaml.registry import (
    RegisteredWorkflow,
    WorkflowRegistry,
    registry,
    workflow,
)

__all__ = [
    "RegisteredWorkflow",
    "WorkflowRegistry",
    "registry",
    "workflow",
]
