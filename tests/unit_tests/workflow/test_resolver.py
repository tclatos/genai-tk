"""Unit tests for workflow resolution."""

from __future__ import annotations

import pytest

from genai_tk.workflow.resolver import (
    WorkflowResolutionError,
    parse_cli_overrides,
    resolve_workflow_invocation,
)


# ---------------------------------------------------------------------------
# parse_cli_overrides
# ---------------------------------------------------------------------------


def test_parse_cli_overrides_supports_nested_values() -> None:
    result = parse_cli_overrides(["batch_size=10", "converter=mistral"])
    assert result == {"batch_size": 10, "converter": "mistral"}


def test_parse_cli_overrides_empty_returns_empty_dict() -> None:
    assert parse_cli_overrides(None) == {}
    assert parse_cli_overrides([]) == {}


def test_parse_cli_overrides_rejects_invalid_item() -> None:
    with pytest.raises(WorkflowResolutionError, match="KEY=VALUE"):
        parse_cli_overrides(["no_equals_sign"])


def test_parse_cli_overrides_rejects_missing_key() -> None:
    with pytest.raises(WorkflowResolutionError):
        parse_cli_overrides(["=value"])


# ---------------------------------------------------------------------------
# resolve_workflow_invocation — requires a real config fixture; these smoke
# tests use the built-in genai-tk workflows via global config.
# ---------------------------------------------------------------------------


def test_resolve_unknown_workflow_raises() -> None:
    with pytest.raises(WorkflowResolutionError, match="not found"):
        resolve_workflow_invocation("__nonexistent_workflow__")


def test_resolve_unknown_preset_raises() -> None:
    with pytest.raises(WorkflowResolutionError):
        resolve_workflow_invocation("markdownize/__nonexistent_preset__")
