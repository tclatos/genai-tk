"""Unit tests for workflow executor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from genai_tk.workflow.compiled_models import CompiledStep, CompiledWorkflow, InvokeSpec
from genai_tk.workflow.executor import WorkflowExecutionError, execute_workflow
from genai_tk.workflow.models import ResolvedWorkflowInvocation, StepSpec, WorkflowSpec

# ---------------------------------------------------------------------------
# execute_workflow tests
# ---------------------------------------------------------------------------


def _make_invocation(steps: list[StepSpec], values: dict | None = None, force: bool = False) -> ResolvedWorkflowInvocation:
    workflow = WorkflowSpec(name="test_wf", steps=steps)
    return ResolvedWorkflowInvocation(
        requested_name="test_profile",
        workflow_name="test_wf",
        workflow=workflow,
        values=values or {},
        force=force,
    )


@patch("genai_tk.workflow.compiler.WorkflowCompiler.compile")
@patch("genai_tk.workflow.prefect.flow_factory.PrefectFlowFactory.run")
def test_execute_workflow_delegates_to_factory(mock_run, mock_compile):
    """execute_workflow compiles and runs via PrefectFlowFactory."""
    mock_compiled = MagicMock(spec=CompiledWorkflow)
    mock_compile.return_value = mock_compiled
    mock_run.return_value = {"convert": {"files_processed": 5}}

    invocation = _make_invocation(
        [StepSpec(id="convert", invoke=InvokeSpec(target="json.dumps"))],
        values={"root_dir": "/tmp"},
    )
    results = execute_workflow(invocation)

    mock_compile.assert_called_once()
    mock_run.assert_called_once()
    assert results == {"convert": {"files_processed": 5}}


@patch("genai_tk.workflow.compiler.WorkflowCompiler.compile")
@patch("genai_tk.workflow.prefect.flow_factory.PrefectFlowFactory.run")
def test_execute_workflow_force_sets_values(mock_run, mock_compile):
    """When force=True, values dict gets force/force_rebuild keys."""
    mock_compile.return_value = MagicMock(spec=CompiledWorkflow)
    mock_run.return_value = {}

    invocation = _make_invocation(
        [StepSpec(id="s1", invoke=InvokeSpec(target="json.dumps"))],
        values={},
        force=True,
    )
    execute_workflow(invocation)

    compile_call = mock_compile.call_args
    values_passed = compile_call[0][1]  # positional: (spec, values)
    assert values_passed.get("force") is True
    assert values_passed.get("force_rebuild") is True


@patch("genai_tk.workflow.prefect.flow_factory.PrefectFlowFactory.run")
@patch("genai_tk.workflow.compiler.WorkflowCompiler.compile")
def test_execute_workflow_wraps_flow_error(mock_compile, mock_run):
    """PrefectFlowFactory errors are re-raised as WorkflowExecutionError."""
    from genai_tk.workflow.prefect.flow_factory import WorkflowExecutionError as FlowError

    mock_compile.return_value = MagicMock(spec=CompiledWorkflow)
    mock_run.side_effect = FlowError("step failed")

    invocation = _make_invocation([StepSpec(id="s1", invoke=InvokeSpec(target="json.dumps"))])

    with pytest.raises(WorkflowExecutionError, match="step failed"):
        execute_workflow(invocation)

