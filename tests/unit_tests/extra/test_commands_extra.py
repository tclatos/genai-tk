"""Unit tests for commands_extra workflow profile support."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import typer

from genai_tk.extra.commands_extra import _resolve_document_flow_params
from genai_tk.workflow.models import ResolvedWorkflowInvocation, StepSpec, WorkflowSpec


def _make_invocation(values: dict) -> ResolvedWorkflowInvocation:
    return ResolvedWorkflowInvocation(
        requested_name="test_profile",
        workflow_name="markdownize_documents",
        workflow=WorkflowSpec(name="markdownize_documents", steps=[StepSpec(id="convert", uses="m.f")]),
        values=values,
    )


def test_resolve_direct_cli_params():
    params = _resolve_document_flow_params(
        root_dir="/data/docs",
        output_dir="/data/md",
        include_patterns=["*.pdf"],
        exclude_patterns=None,
        recursive=True,
        batch_size=10,
        force=False,
        workflow_config=None,
    )
    assert params["root_dir"] == "/data/docs"
    assert params["output_dir"] == "/data/md"
    assert params["include_patterns"] == ["*.pdf"]
    assert params["recursive"] is True
    assert params["batch_size"] == 10


def test_resolve_direct_missing_root_dir():
    with pytest.raises(typer.BadParameter, match="root_dir and output_dir are required"):
        _resolve_document_flow_params(
            root_dir=None,
            output_dir=None,
            include_patterns=None,
            exclude_patterns=None,
            recursive=False,
            batch_size=5,
            force=False,
            workflow_config=None,
        )


@patch("genai_tk.workflow.resolver.resolve_workflow_invocation")
def test_resolve_from_workflow_config(mock_resolve):
    mock_resolve.return_value = _make_invocation(
        {"root_dir": "/resolved/docs", "output_dir": "/resolved/md", "recursive": True, "batch_size": 8}
    )
    params = _resolve_document_flow_params(
        root_dir=None,
        output_dir=None,
        include_patterns=None,
        exclude_patterns=None,
        recursive=False,
        batch_size=5,
        force=True,
        workflow_config="markdownize_docs",
    )
    assert params["root_dir"] == "/resolved/docs"
    assert params["output_dir"] == "/resolved/md"
    assert params["recursive"] is True
    assert params["batch_size"] == 8
    assert params["force"] is True


@patch("genai_tk.workflow.resolver.resolve_workflow_invocation")
def test_resolve_config_cli_overrides_merged(mock_resolve):
    mock_resolve.return_value = _make_invocation(
        {"root_dir": "/override/docs", "output_dir": "/override/md", "recursive": False, "batch_size": 5}
    )
    _resolve_document_flow_params(
        root_dir="/override/docs",
        output_dir=None,
        include_patterns=None,
        exclude_patterns=None,
        recursive=False,
        batch_size=5,
        force=False,
        workflow_config="markdownize_docs",
    )
    # The override should have been passed to resolve_workflow_invocation
    call_kwargs = mock_resolve.call_args[1]
    assert call_kwargs["cli_overrides"]["root_dir"] == "/override/docs"


@patch("genai_tk.workflow.resolver.resolve_workflow_invocation")
def test_resolve_config_missing_output_dir(mock_resolve):
    mock_resolve.return_value = _make_invocation({"root_dir": "/docs"})
    with pytest.raises(typer.BadParameter, match="missing required"):
        _resolve_document_flow_params(
            root_dir=None,
            output_dir=None,
            include_patterns=None,
            exclude_patterns=None,
            recursive=False,
            batch_size=5,
            force=False,
            workflow_config="markdownize_docs",
        )
