"""Workflow configuration and resolution primitives.

This package provides YAML-driven workflow orchestration for genai-tk,
backed by the Prefect engine for parallel, dependency-aware execution.
"""

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
from genai_tk.workflow.flow_cache.manifest import ManifestCache
from genai_tk.workflow.models import (
    ParamSpec,
    PipelineStep,
    ResolvedWorkflowInvocation,
    StepSpec,
    WorkflowDefV2,
    WorkflowSpec,
)
from genai_tk.workflow.prefect.flow_factory import PrefectFlowFactory
from genai_tk.workflow.registry import RegisteredWorkflow, WorkflowRegistry, registry, workflow
from genai_tk.workflow.resolver import (
    WorkflowResolutionError,
    list_preset_names,
    list_workflow_names,
    load_workflows,
    parse_cli_overrides,
    resolve_workflow_invocation,
)

__all__ = [
    # compiled models
    "ArtifactSpec",
    "CacheSpec",
    "CompiledStep",
    "CompiledWorkflow",
    "ExecutionSpec",
    "ForeachSpec",
    "InvokeSpec",
    "ManifestCache",
    "PrefectFlowFactory",
    "StepKind",
    # models
    "ParamSpec",
    "PipelineStep",
    "ResolvedWorkflowInvocation",
    "StepSpec",
    "WorkflowDefV2",
    "WorkflowSpec",
    # compiled models
    "ArtifactSpec",
    "CacheSpec",
    "CompiledStep",
    "CompiledWorkflow",
    "ExecutionSpec",
    "ForeachSpec",
    "InvokeSpec",
    "ManifestCache",
    "PrefectFlowFactory",
    "StepKind",
    # registry
    "RegisteredWorkflow",
    "WorkflowRegistry",
    "registry",
    "workflow",
    # compiler / executor
    "WorkflowCompilationError",
    "WorkflowCompiler",
    "WorkflowExecutionError",
    "execute_workflow",
    # resolver
    "WorkflowResolutionError",
    "list_preset_names",
    "list_workflow_names",
    "load_workflows",
    "parse_cli_overrides",
    "resolve_workflow_invocation",
]
