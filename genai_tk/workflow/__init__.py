"""Workflow configuration and resolution primitives.

This package provides the first building blocks for a YAML-driven workflow
layer in genai-tk. The initial implementation focuses on configuration
validation, workflow/profile resolution, and CLI dry-run support.
"""

from genai_tk.workflow.executor import WorkflowExecutionError, execute_workflow
from genai_tk.workflow.models import (
    CacheSpec,
    MaterializationSpec,
    ResolvedWorkflowInvocation,
    StepSpec,
    StepTemplateSpec,
    WorkflowProfileSpec,
    WorkflowSpec,
)
from genai_tk.workflow.resolver import (
    WorkflowResolutionError,
    list_step_template_names,
    list_workflow_names,
    list_workflow_profile_names,
    load_workflow_profile,
    load_workflow_spec,
    parse_cli_overrides,
    resolve_workflow_invocation,
)

__all__ = [
    "CacheSpec",
    "MaterializationSpec",
    "ResolvedWorkflowInvocation",
    "StepSpec",
    "StepTemplateSpec",
    "WorkflowExecutionError",
    "WorkflowProfileSpec",
    "WorkflowResolutionError",
    "WorkflowSpec",
    "execute_workflow",
    "list_step_template_names",
    "list_workflow_names",
    "list_workflow_profile_names",
    "load_workflow_profile",
    "load_workflow_spec",
    "parse_cli_overrides",
    "resolve_workflow_invocation",
]
