"""Unit tests for canonical harness trace metadata (see docs/design/harness_interoperability_proposal.md §4)."""

import os

import pytest
from pydantic import ValidationError

from genai_tk.utils.tracing import (
    HarnessTraceMetadata,
    apply_harness_trace_metadata,
    trace_project_name,
)


@pytest.fixture
def _clear_trace_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("LANGSMITH_PROJECT", "LANGSMITH_SESSION_ID"):
        monkeypatch.delenv(var, raising=False)


def test_trace_project_name_format() -> None:
    assert trace_project_name("deerflow", "Research Assistant") == "GenAITk-deerflow-Research Assistant"
    assert trace_project_name("langchain", "deep") == "GenAITk-langchain-deep"


def test_apply_metadata_sets_langsmith_project(_clear_trace_env: None) -> None:
    meta = apply_harness_trace_metadata(
        HarnessTraceMetadata(harness="langchain", profile_name="Research")
    )
    assert os.environ["LANGSMITH_PROJECT"] == "GenAITk-langchain-Research"
    assert meta.harness == "langchain"


def test_apply_metadata_sets_session_id_when_present(_clear_trace_env: None) -> None:
    apply_harness_trace_metadata(
        HarnessTraceMetadata(
            harness="deerflow",
            profile_name="Coder",
            session_id="sess-123",
            thread_id="t-1",
            model_name="gpt_41mini@openai",
            environment="dev",
        )
    )
    assert os.environ["LANGSMITH_SESSION_ID"] == "sess-123"
    assert os.environ["LANGSMITH_PROJECT"] == "GenAITk-deerflow-Coder"


def test_apply_metadata_clears_session_id_when_absent(_clear_trace_env: None) -> None:
    os.environ["LANGSMITH_SESSION_ID"] = "stale"
    apply_harness_trace_metadata(HarnessTraceMetadata(harness="langchain", profile_name="x"))
    assert "LANGSMITH_SESSION_ID" not in os.environ


def test_metadata_model_is_frozen() -> None:
    meta = HarnessTraceMetadata(harness="langchain", profile_name="x")
    with pytest.raises(ValidationError):
        meta.harness = "deerflow"  # type: ignore[misc]
