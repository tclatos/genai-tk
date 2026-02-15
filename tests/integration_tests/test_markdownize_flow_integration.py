"""Integration-style tests for markdownize flow with deterministic stubs."""

from datetime import datetime, timezone

import pytest
from upath import UPath

import genai_tk.extra.markdownize_prefect_flow as mod
from genai_tk.extra.markdownize_prefect_flow import MarkdownizeManifest, MarkdownizeManifestEntry, markdownize_flow


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

    def fake_submit(file_info, output_dir, root_dir, use_mistral_ocr):
        entry = MarkdownizeManifestEntry(
            source_hash=file_info.content_hash,
            output_path=f"{file_info.path.stem}.md",
            processed_at=datetime.now(timezone.utc),
        )
        return _FakeFuture((str(file_info.path), entry))

    monkeypatch.setattr(mod._process_single_file_task, "submit", fake_submit)
    monkeypatch.setattr(mod, "resolve_files", lambda *args, **kwargs: [str(p) for p in input_dir.iterdir()])

    manifest = markdownize_flow.fn(
        root_dir=str(input_dir),
        output_dir=str(output_dir),
        include_patterns=["*.pdf"],
        recursive=False,
        batch_size=1,
        force=False,
        use_mistral_ocr=False,
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

    def fake_submit(file_info, output_dir, root_dir, use_mistral_ocr):
        entry = MarkdownizeManifestEntry(
            source_hash=file_info.content_hash,
            output_path=f"{file_info.path.stem}.md",
            processed_at=datetime.now(timezone.utc),
        )
        return _FakeFuture((str(file_info.path), entry))

    monkeypatch.setattr(mod._process_single_file_task, "submit", fake_submit)
    monkeypatch.setattr(mod, "resolve_files", lambda *args, **kwargs: [str(p) for p in input_dir.iterdir()])

    manifest1 = markdownize_flow.fn(
        root_dir=str(input_dir),
        output_dir=str(output_dir),
        include_patterns=["*.pdf"],
        recursive=False,
        batch_size=1,
        force=False,
        use_mistral_ocr=False,
    )

    manifest2 = markdownize_flow.fn(
        root_dir=str(input_dir),
        output_dir=str(output_dir),
        include_patterns=["*.pdf"],
        recursive=False,
        batch_size=1,
        force=False,
        use_mistral_ocr=False,
    )

    assert len(manifest1.entries) == 1
    assert len(manifest2.entries) == 1
