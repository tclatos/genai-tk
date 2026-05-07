"""Unit tests for workflow executor."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from genai_tk.workflow.executor import (
    WorkflowExecutionError,
    _import_callable,
    _resolve_step_inputs,
    _topological_sort,
    execute_workflow,
)
from genai_tk.workflow.models import ResolvedWorkflowInvocation, StepSpec, WorkflowSpec

# ---------------------------------------------------------------------------
# _import_callable tests
# ---------------------------------------------------------------------------


def test_import_callable_builtin():
    result = _import_callable("json.dumps")
    import json

    assert result is json.dumps


def test_import_callable_invalid_path():
    with pytest.raises(WorkflowExecutionError, match="Invalid step path"):
        _import_callable("no_dots_here")


def test_import_callable_missing_module():
    with pytest.raises(WorkflowExecutionError, match="Cannot import module"):
        _import_callable("nonexistent_module_xyz.func")


def test_import_callable_missing_attribute():
    with pytest.raises(WorkflowExecutionError, match="has no attribute"):
        _import_callable("json.nonexistent_function_xyz")


# ---------------------------------------------------------------------------
# _resolve_step_inputs tests
# ---------------------------------------------------------------------------


def test_resolve_step_inputs_substitution():
    step = StepSpec(
        id="convert",
        uses="some.module.func",
        inputs={"root_dir": "${profile.root_dir}", "output_dir": "${profile.output_dir}"},
        params={"batch_size": "${profile.batch_size}", "recursive": True},
    )
    values = {"root_dir": "/data/docs", "output_dir": "/data/md", "batch_size": 10}
    result = _resolve_step_inputs(step, values)
    assert result == {
        "root_dir": "/data/docs",
        "output_dir": "/data/md",
        "batch_size": 10,
        "recursive": True,
    }


def test_resolve_step_inputs_missing_value():
    step = StepSpec(
        id="convert",
        uses="some.module.func",
        inputs={"root_dir": "${profile.missing_key}"},
    )
    result = _resolve_step_inputs(step, {})
    assert result == {"root_dir": None}


# ---------------------------------------------------------------------------
# _topological_sort tests
# ---------------------------------------------------------------------------


def test_topological_sort_linear():
    steps = [
        StepSpec(id="a", uses="m.a"),
        StepSpec(id="b", uses="m.b", needs=["a"]),
        StepSpec(id="c", uses="m.c", needs=["b"]),
    ]
    ordered = _topological_sort(steps)
    ids = [s.id for s in ordered]
    assert ids == ["a", "b", "c"]


def test_topological_sort_parallel():
    steps = [
        StepSpec(id="a", uses="m.a"),
        StepSpec(id="b", uses="m.b"),
        StepSpec(id="c", uses="m.c", needs=["a", "b"]),
    ]
    ordered = _topological_sort(steps)
    ids = [s.id for s in ordered]
    assert ids.index("a") < ids.index("c")
    assert ids.index("b") < ids.index("c")


def test_topological_sort_cycle():
    steps = [
        StepSpec(id="a", uses="m.a", needs=["b"]),
        StepSpec(id="b", uses="m.b", needs=["a"]),
    ]
    with pytest.raises(WorkflowExecutionError, match="Cycle detected"):
        _topological_sort(steps)


def test_topological_sort_unknown_dep():
    steps = [
        StepSpec(id="a", uses="m.a", needs=["unknown"]),
    ]
    with pytest.raises(WorkflowExecutionError, match="unknown step 'unknown'"):
        _topological_sort(steps)


# ---------------------------------------------------------------------------
# execute_workflow tests
# ---------------------------------------------------------------------------


@patch("genai_tk.extra.prefect.runtime.run_flow_ephemeral")
def test_execute_workflow_single_step(mock_run):
    mock_run.return_value = {"files_processed": 5}

    workflow = WorkflowSpec(
        name="test_wf",
        steps=[
            StepSpec(
                id="convert",
                uses="json.dumps",
                inputs={"root_dir": "${profile.root_dir}"},
                params={"batch_size": "${profile.batch_size}"},
            )
        ],
    )
    invocation = ResolvedWorkflowInvocation(
        requested_name="test_profile",
        workflow_name="test_wf",
        workflow=workflow,
        values={"root_dir": "/tmp/docs", "batch_size": 5},
    )

    results = execute_workflow(invocation)
    assert "convert" in results
    assert results["convert"] == {"files_processed": 5}
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args
    assert call_kwargs[1]["root_dir"] == "/tmp/docs"
    assert call_kwargs[1]["batch_size"] == 5


@patch("genai_tk.extra.prefect.runtime.run_flow_ephemeral")
def test_execute_workflow_force_flag(mock_run):
    mock_run.return_value = None

    workflow = WorkflowSpec(
        name="test_wf",
        steps=[StepSpec(id="step1", uses="json.dumps", inputs={"x": "val"})],
    )
    invocation = ResolvedWorkflowInvocation(
        requested_name="p",
        workflow_name="test_wf",
        workflow=workflow,
        values={},
        force=True,
    )

    execute_workflow(invocation)
    call_kwargs = mock_run.call_args[1]
    assert call_kwargs["force"] is True


@patch("genai_tk.extra.prefect.runtime.run_flow_ephemeral")
def test_execute_workflow_step_failure_abort(mock_run):
    mock_run.side_effect = RuntimeError("disk full")

    workflow = WorkflowSpec(
        name="test_wf",
        steps=[StepSpec(id="step1", uses="json.dumps", on_failure="abort")],
    )
    invocation = ResolvedWorkflowInvocation(
        requested_name="p",
        workflow_name="test_wf",
        workflow=workflow,
        values={},
    )

    with pytest.raises(WorkflowExecutionError, match="disk full"):
        execute_workflow(invocation)


@patch("genai_tk.extra.prefect.runtime.run_flow_ephemeral")
def test_execute_workflow_step_failure_skip(mock_run):
    mock_run.side_effect = [RuntimeError("oops"), "ok"]

    workflow = WorkflowSpec(
        name="test_wf",
        steps=[
            StepSpec(id="step1", uses="json.dumps", on_failure="skip"),
            StepSpec(id="step2", uses="json.loads"),
        ],
    )
    invocation = ResolvedWorkflowInvocation(
        requested_name="p",
        workflow_name="test_wf",
        workflow=workflow,
        values={},
    )

    results = execute_workflow(invocation)
    assert results["step1"] is None
    assert results["step2"] == "ok"
