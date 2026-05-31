"""Compiled runtime models for the Prefect-native workflow engine.

These models are produced by :class:`~genai_tk.workflow.compiler.WorkflowCompiler`
from the authoring-level :class:`~genai_tk.workflow.models.WorkflowSpec`.  They are
normalized, fully-resolved, and ready for :class:`~genai_tk.workflow.prefect.flow_factory.PrefectFlowFactory`
to turn into a real Prefect flow object.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class StepKind(str, Enum):
    """Execution strategy for a compiled step."""

    task = "task"
    flow = "flow"
    workflow = "workflow"
    deployment = "deployment"
    factory = "factory"
    callable = "callable"


class InvokeSpec(BaseModel):
    """How to execute a step: the execution kind and the Python dotted-path target."""

    kind: StepKind = StepKind.callable
    target: str = ""


class ForeachSpec(BaseModel):
    """Fan-out / map spec: run a step once per item in a collection."""

    from_ref: str = Field(alias="from")
    as_var: str = Field(default="item", alias="as")
    concurrency_limit: int | None = None

    model_config = {"populate_by_name": True}


class ExecutionSpec(BaseModel):
    """Runtime execution policy for a step (retries, tags, work placement)."""

    retries: int = 0
    retry_delay_seconds: float = 0.0
    tags: list[str] = Field(default_factory=list)
    work_pool: str | None = None
    work_queue: str | None = None
    concurrency_limit: int | None = None
    on_failure: Literal["abort", "skip", "continue"] = "abort"


class CacheSpec(BaseModel):
    """Cache reuse policy for a step."""

    backend: Literal["none", "prefect_result", "manifest", "hybrid"] = "none"
    key_include: list[str] = Field(default_factory=list)


class ArtifactSpec(BaseModel):
    """Artifact and result publication policy for a step."""

    publish_result: bool = False
    publish_metadata: bool = False
    output_dir: str | None = None


class CompiledStep(BaseModel):
    """A normalized, fully-resolved step ready for Prefect execution."""

    id: str
    invoke: InvokeSpec
    wait_for: list[str] = Field(default_factory=list)
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    cache: CacheSpec = Field(default_factory=CacheSpec)
    artifacts: ArtifactSpec = Field(default_factory=ArtifactSpec)
    foreach: ForeachSpec | None = None
    inline: bool = False

    model_config = {"populate_by_name": True}


class CompiledWorkflow(BaseModel):
    """A fully normalized workflow graph ready for :class:`PrefectFlowFactory`."""

    name: str
    description: str = ""
    steps: list[CompiledStep] = Field(default_factory=list)
    values: dict[str, Any] = Field(default_factory=dict)
