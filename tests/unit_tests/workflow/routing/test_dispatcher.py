"""Unit tests for the ingest dispatcher flow.

Follows the test_flow_factory.py pattern: the flow body is invoked via ``.fn()``
with ``_run_workflow_task`` mocked at the task level, so no live Prefect server
is needed. This also exercises the ``PrefectFlowFactory`` subflow resolution
path (registry names and dotted paths) and manifest-cache code-versioning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from genai_tk.workflow.registry import workflow
from genai_tk.workflow.routing.dispatcher import _run_workflow_task, ingest_dispatch_flow
from genai_tk.workflow.routing.models import IngestRouteTable, IngestRule

# ---------------------------------------------------------------------------
# Test doubles — routed workflows (ingestion contract: sources + md_output_dir)
# ---------------------------------------------------------------------------

_registered_calls: list[dict[str, Any]] = []


@workflow(name="routing_test_files_flow", description="test double for file routing")
def _files_test_flow(*, sources: list[str], md_output_dir: str, **kwargs: Any) -> dict[str, Any]:
    _registered_calls.append({"sources": sources, "md_output_dir": md_output_dir, **kwargs})
    for src in sources:
        Path(md_output_dir, Path(src).name + ".md").write_text("md", encoding="utf-8")
    return {"workflow": "files", "processed": len(sources)}


@workflow(name="routing_test_url_flow", description="test double for URL routing")
def _url_test_flow(*, sources: list[str], md_output_dir: str, **kwargs: Any) -> dict[str, Any]:
    _registered_calls.append({"sources": sources, "md_output_dir": md_output_dir, **kwargs})
    return {"workflow": "urls", "processed": len(sources)}


class _MockFuture:
    def __init__(self, value: Any = None, exception: BaseException | None = None) -> None:
        self._value = value
        self._exc = exception

    def result(self) -> Any:
        if self._exc is not None:
            raise self._exc
        return self._value


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dispatch_routes_files_and_urls_to_parallel_buckets(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "report.pdf").write_text("pdf", encoding="utf-8")
    (docs / "readme.txt").write_text("txt", encoding="utf-8")

    table = IngestRouteTable(
        routes=[
            IngestRule(pathspec="https://**", workflow="routing_test_url_flow"),
            IngestRule(pathspec="**/*.pdf", workflow="routing_test_files_flow", params={"marker": "pdf-route"}),
        ],
        default="routing_test_files_flow",
    )

    captured: list[dict[str, Any]] = []

    def _fake_submit(bucket_workflow: str, bucket_values: dict[str, Any]) -> _MockFuture:
        captured.append({"workflow": bucket_workflow, **bucket_values})
        # Execute the real task logic synchronously — resolves via the registry
        # (PrefectFlowFactory path) exactly like production.
        return _MockFuture(value=_run_workflow_task.fn(bucket_workflow, bucket_values))

    with patch("genai_tk.workflow.routing.dispatcher._run_workflow_task") as mock_task:
        mock_task.submit.side_effect = _fake_submit
        result = ingest_dispatch_flow.fn(
            sources=[str(docs), "https://example.com/article"],
            md_output_dir=str(tmp_path / "md"),
            routes=table,
        )

    assert result["failures"] == []
    assert result["items"] == 3  # 2 files + 1 URL

    # URL bucket → URL flow; file buckets → files flow (pdf rule before default)
    url_calls = [c for c in captured if c["workflow"] == "routing_test_url_flow"]
    assert len(url_calls) == 1
    assert url_calls[0]["sources"] == ["https://example.com/article"]

    file_calls = [c for c in captured if c["workflow"] == "routing_test_files_flow"]
    assert len(file_calls) == 2  # pdf bucket + default bucket (distinct params → distinct buckets)
    pdf_bucket = next(c for c in file_calls if c.get("marker") == "pdf-route")
    default_bucket = next(c for c in file_calls if "marker" not in c)
    assert pdf_bucket["sources"] == [str(docs / "report.pdf")]
    assert default_bucket["sources"] == [str(docs / "readme.txt")]

    # Merged results
    assert result["buckets"]["routing_test_url_flow"] == {"workflow": "urls", "processed": 1}


def test_dispatch_failure_isolation(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("x", encoding="utf-8")
    table = IngestRouteTable(default="routing_test_files_flow")

    def _fake_submit(bucket_workflow: str, bucket_values: dict[str, Any]) -> _MockFuture:
        return _MockFuture(exception=RuntimeError("boom"))

    with patch("genai_tk.workflow.routing.dispatcher._run_workflow_task") as mock_task:
        mock_task.submit.side_effect = _fake_submit
        result = ingest_dispatch_flow.fn(
            sources=[str(tmp_path)],
            md_output_dir=str(tmp_path / "md"),
            routes=table,
        )

    assert len(result["failures"]) == 1
    assert "boom" in result["failures"][0]


def test_dispatch_dotted_path_target(tmp_path: Path) -> None:
    """A dotted-path workflow reference resolves without the DSL resolver."""
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    table = IngestRouteTable(default="tests.unit_tests.workflow.routing.test_dispatcher._plain_flow")

    def _fake_submit(bucket_workflow: str, bucket_values: dict[str, Any]) -> _MockFuture:
        return _MockFuture(value=_run_workflow_task.fn(bucket_workflow, bucket_values))

    with patch("genai_tk.workflow.routing.dispatcher._run_workflow_task") as mock_task:
        mock_task.submit.side_effect = _fake_submit
        result = ingest_dispatch_flow.fn(sources=[str(tmp_path)], md_output_dir=str(tmp_path / "md"), routes=table)

    assert result["failures"] == []
    assert result["buckets"]["tests.unit_tests.workflow.routing.test_dispatcher._plain_flow"]["ok"] is True


def _plain_flow(*, sources: list[str], md_output_dir: str, **kwargs: Any) -> dict[str, Any]:
    return {"ok": True, "sources": sources}
