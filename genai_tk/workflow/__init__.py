"""Workflow configuration and resolution primitives.

This package provides YAML-driven workflow orchestration for genai-tk,
backed by the Prefect engine for parallel, dependency-aware execution.
"""

from genai_tk.workflow.cache.manifest import ManifestCache
from genai_tk.workflow.compiled_models import (
    ArtifactSpec,
    CacheSpec,
    CompiledStep,
    CompiledWorkflow,
    ExecutionSpec,
    ForeachSpec,
    InvokeSpec,
    StepKind,
)
from genai_tk.workflow.compiler import WorkflowCompilationError, WorkflowCompiler
from genai_tk.workflow.executor import WorkflowExecutionError, execute_workflow
from genai_tk.workflow.models import (
    ResolvedWorkflowInvocation,
    StepSpec,
    StepTemplateSpec,
    WorkflowProfileSpec,
    WorkflowSpec,
)
from genai_tk.workflow.prefect.flow_factory import PrefectFlowFactory
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
    "ArtifactSpec",
    "CacheSpec",
    "CompiledStep",
    "CompiledWorkflow",
    "ExecutionSpec",
    "ForeachSpec",
    "InvokeSpec",
    "ManifestCache",
    "PrefectFlowFactory",
    "ResolvedWorkflowInvocation",
    "StepKind",
    "StepSpec",
    "StepTemplateSpec",
    "WorkflowCompilationError",
    "WorkflowCompiler",
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
