"""Integration-style tests for markdownize flow with deterministic stubs."""

from pathlib import Path

import pytest

import genai_tk.workflow.markdownize.flow as mod
from genai_tk.workflow.markdownize.flow import (
    MarkdownizeManifest,
    markdownize_flow,
)
from genai_tk.workflow.sources import ResolvedSourceFile

# These tests drive ``markdownize_flow.fn(...)`` directly (bypassing the @flow
# decorator) for deterministic stubbed execution, so Prefect artifact creation
# happens outside a flow/task run context. The flow already treats artifacts as
# best-effort (try/except); suppress the resulting deprecation warning rather
# than adding a real flow-run context that would defeat the fast stub path.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Artifact creation outside of a flow or task run is deprecated:FutureWarning",
)


class _FakeFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


@pytest.mark.integration
@pytest.mark.fake_models
def test_markdownize_flow_creates_manifest(tmp_path, monkeypatch) -> None:
    """Ensure markdownize flow writes a manifest with entries."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    pdf_file = input_dir / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\ncontent")

    def fake_submit(original_src, convert_src, route, out_abs, rel_out, pdf_converter):
        return _FakeFuture((original_src, rel_out))

    monkeypatch.setattr(mod._convert_file_task, "submit", fake_submit)
    monkeypatch.setattr(
        mod,
        "resolve_sources",
        lambda *args, **kwargs: [ResolvedSourceFile(path=p, root=input_dir) for p in input_dir.iterdir()],
    )

    manifest = markdownize_flow.fn(
        sources=str(input_dir),
        md_output_dir=str(output_dir),
        pathspecs=["**/*.pdf"],
        batch_size=1,
        profile="fast",
    )

    assert isinstance(manifest, MarkdownizeManifest)
    assert len(manifest.entries) == 1

    manifest_path = Path(output_dir) / ".cache" / "manifest.json"
    assert manifest_path.exists()


@pytest.mark.integration
@pytest.mark.fake_models
def test_markdownize_flow_skips_unchanged(tmp_path, monkeypatch) -> None:
    """Ensure unchanged files are skipped on rerun."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    pdf_file = input_dir / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\ncontent")

    call_count = 0

    def fake_submit(original_src, convert_src, route, out_abs, rel_out, pdf_converter):
        nonlocal call_count
        call_count += 1
        return _FakeFuture((original_src, rel_out))

    monkeypatch.setattr(mod._convert_file_task, "submit", fake_submit)
    monkeypatch.setattr(
        mod,
        "resolve_sources",
        lambda *args, **kwargs: [ResolvedSourceFile(path=p, root=input_dir) for p in input_dir.iterdir()],
    )

    # First run: file is processed and manifest is written
    markdownize_flow.fn(
        sources=str(input_dir),
        md_output_dir=str(output_dir),
        pathspecs=["**/*.pdf"],
        batch_size=1,
        profile="fast",
    )
    assert call_count == 1

    # Second run on same unchanged file: should be skipped via manifest
    markdownize_flow.fn(
        sources=str(input_dir),
        md_output_dir=str(output_dir),
        pathspecs=["**/*.pdf"],
        batch_size=1,
        profile="fast",
    )
    assert call_count == 1, "Unchanged file should be skipped on rerun"
