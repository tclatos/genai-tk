"""Integration-style tests for markdownize flow with deterministic stubs."""

import pytest
from upath import UPath

import genai_tk.workflow.prefect.flows.markdownize_flow as mod
from genai_tk.workflow.prefect.flows.markdownize_flow import (
    MarkdownizeManifest,
    markdownize_flow,
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

    def fake_submit(file_info, output_dir, root_dir, pdf_converter):
        output_path = f"{file_info.path.stem}.md"
        return _FakeFuture((str(file_info.path), output_path))

    monkeypatch.setattr(mod._process_single_file_task, "submit", fake_submit)
    monkeypatch.setattr(mod, "resolve_files", lambda *args, **kwargs: [str(p) for p in input_dir.iterdir()])

    manifest = markdownize_flow.fn(
        base_dir=str(input_dir),
        output_dir=str(output_dir),
        pathspecs=["**/*.pdf"],
        batch_size=1,
        force=False,
        pdf_converter="markitdown",
    )

    assert isinstance(manifest, MarkdownizeManifest)
    assert len(manifest.entries) == 1

    manifest_path = UPath(output_dir) / "manifest.json"
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

    def fake_submit(file_info, output_dir, root_dir, pdf_converter):
        nonlocal call_count
        call_count += 1
        output_path = f"{file_info.path.stem}.md"
        return _FakeFuture((str(file_info.path), output_path))

    monkeypatch.setattr(mod._process_single_file_task, "submit", fake_submit)
    monkeypatch.setattr(mod, "resolve_files", lambda *args, **kwargs: [str(p) for p in input_dir.iterdir()])

    # First run: file is processed and manifest is written
    markdownize_flow.fn(
        base_dir=str(input_dir),
        output_dir=str(output_dir),
        pathspecs=["**/*.pdf"],
        batch_size=1,
        force=False,
        pdf_converter="markitdown",
    )
    assert call_count == 1

    # Second run on same unchanged file: should be skipped via manifest
    markdownize_flow.fn(
        base_dir=str(input_dir),
        output_dir=str(output_dir),
        pathspecs=["**/*.pdf"],
        batch_size=1,
        force=False,
        pdf_converter="markitdown",
    )
    assert call_count == 1, "Unchanged file should be skipped on rerun"
