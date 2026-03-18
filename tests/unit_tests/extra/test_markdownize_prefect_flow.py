"""Tests for markdownize Prefect flow."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from upath import UPath

import genai_tk.extra.markdownize_prefect_flow as mod
from genai_tk.extra.markdownize_prefect_flow import (
    MarkdownizeManifest,
    MarkdownizeManifestEntry,
    _compute_hash,
    _load_manifest,
    _prepare_files,
    _save_manifest,
    markdownize_flow,
)


class _FakeFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_files(temp_dir: Path) -> list[Path]:
    """Create sample files for testing."""
    files = []

    # Create a sample PDF-like file
    pdf_file = temp_dir / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\ntest content")
    files.append(pdf_file)

    # Create a sample DOCX file (really a zip)
    docx_file = temp_dir / "sample.docx"
    docx_file.write_bytes(b"PK\x03\x04test docx")
    files.append(docx_file)

    return files


def test_compute_hash() -> None:
    """Test hash computation."""
    content1 = b"test content"
    content2 = b"test content"
    content3 = b"different content"

    hash1 = _compute_hash(content1)
    hash2 = _compute_hash(content2)
    hash3 = _compute_hash(content3)

    assert hash1 == hash2
    assert hash1 != hash3
    assert isinstance(hash1, str)
    assert len(hash1) > 0


def test_manifest_save_load(temp_dir: Path) -> None:
    """Test manifest save and load."""
    from datetime import datetime, timezone

    manifest_path = UPath(temp_dir / "manifest.json")

    # Create and save manifest
    now = datetime.now(timezone.utc)
    manifest = MarkdownizeManifest(
        entries={
            "/test/file1.pdf": MarkdownizeManifestEntry(
                source_hash="hash1",
                output_path="file1.md",
                processed_at=now,
            ),
            "/test/file2.docx": MarkdownizeManifestEntry(
                source_hash="hash2",
                output_path="file2.md",
                processed_at=now,
            ),
        }
    )

    _save_manifest(manifest, manifest_path)

    # Load and verify
    loaded = _load_manifest(manifest_path)
    assert loaded is not None
    assert len(loaded.entries) == 2
    assert loaded.entries["/test/file1.pdf"].output_path == "file1.md"
    assert loaded.entries["/test/file2.docx"].output_path == "file2.md"


def test_manifest_load_nonexistent(temp_dir: Path) -> None:
    """Test loading nonexistent manifest returns None."""
    manifest_path = UPath(temp_dir / "nonexistent.json")
    result = _load_manifest(manifest_path)
    assert result is None


def test_prepare_files_new_files(temp_dir: Path, sample_files: list[Path]) -> None:
    """Test prepare_files with new files."""
    manifest = MarkdownizeManifest()
    files_to_process = [UPath(f) for f in sample_files]

    result, skipped = _prepare_files(files_to_process, manifest, force=False)

    assert len(result) == 2
    assert skipped == 0
    assert result[0].path == UPath(sample_files[0])


def test_prepare_files_unchanged_files(temp_dir: Path, sample_files: list[Path]) -> None:
    """Test prepare_files with unchanged files (manifest contains same hashes)."""
    from datetime import datetime, timezone

    files = [UPath(f) for f in sample_files]
    hashes = {str(f): _compute_hash(f.read_bytes()) for f in files}

    # Create manifest with existing entries
    now = datetime.now(timezone.utc)
    manifest = MarkdownizeManifest(
        entries={
            str(files[0]): MarkdownizeManifestEntry(
                source_hash=hashes[str(files[0])],
                output_path="file1.md",
                processed_at=now,
            ),
            str(files[1]): MarkdownizeManifestEntry(
                source_hash=hashes[str(files[1])],
                output_path="file2.md",
                processed_at=now,
            ),
        }
    )

    result, skipped = _prepare_files(files, manifest, force=False)

    assert len(result) == 0
    assert skipped == 2


def test_prepare_files_force_reprocess(temp_dir: Path, sample_files: list[Path]) -> None:
    """Test prepare_files with force flag reprocesses all files."""
    from datetime import datetime, timezone

    files = [UPath(f) for f in sample_files]
    hashes = {str(f): _compute_hash(f.read_bytes()) for f in files}

    # Create manifest with existing entries
    now = datetime.now(timezone.utc)
    manifest = MarkdownizeManifest(
        entries={
            str(files[0]): MarkdownizeManifestEntry(
                source_hash=hashes[str(files[0])],
                output_path="file1.md",
                processed_at=now,
            ),
        }
    )

    result, skipped = _prepare_files(files, manifest, force=True)

    # With force=True, should reprocess all files
    assert len(result) == 2
    assert skipped == 0


def test_markdownize_flow_basic(temp_dir: Path, sample_files: list[Path], monkeypatch) -> None:
    """Test basic markdownize flow creates a manifest entry per processed file."""
    from datetime import datetime, timezone

    output_dir = temp_dir / "output"
    output_dir.mkdir()

    def fake_submit(file_info, output_dir, root_dir, use_mistral_ocr):
        entry = MarkdownizeManifestEntry(
            source_hash=file_info.content_hash,
            output_path=f"{file_info.path.stem}.md",
            processed_at=datetime.now(timezone.utc),
        )
        return _FakeFuture((str(file_info.path), entry))

    monkeypatch.setattr(mod._process_single_file_task, "submit", fake_submit)
    monkeypatch.setattr(
        mod,
        "resolve_files",
        lambda *args, **kwargs: [str(p) for p in sample_files],
    )

    manifest = markdownize_flow.fn(
        root_dir=str(temp_dir),
        output_dir=str(output_dir),
        include_patterns=["*.pdf", "*.docx"],
        recursive=False,
        batch_size=2,
        force=False,
        use_mistral_ocr=False,
    )

    assert isinstance(manifest, MarkdownizeManifest)
    assert len(manifest.entries) == 2
    manifest_path = UPath(output_dir) / "manifest.json"
    assert manifest_path.exists()


def test_manifest_persistence(temp_dir: Path, sample_files: list[Path], monkeypatch) -> None:
    """Test manifest is persisted and reloaded correctly across two flow runs."""
    from datetime import datetime, timezone

    output_dir = temp_dir / "output"
    output_dir.mkdir()

    call_count = {"n": 0}

    def fake_submit(file_info, output_dir, root_dir, use_mistral_ocr):
        call_count["n"] += 1
        entry = MarkdownizeManifestEntry(
            source_hash=file_info.content_hash,
            output_path=f"{file_info.path.stem}.md",
            processed_at=datetime.now(timezone.utc),
        )
        return _FakeFuture((str(file_info.path), entry))

    monkeypatch.setattr(mod._process_single_file_task, "submit", fake_submit)
    pdf_only = [str(sample_files[0])]
    monkeypatch.setattr(mod, "resolve_files", lambda *args, **kwargs: pdf_only)

    run_kwargs = {
        "root_dir": str(temp_dir),
        "output_dir": str(output_dir),
        "include_patterns": ["*.pdf"],
        "recursive": False,
        "batch_size": 1,
        "force": False,
        "use_mistral_ocr": False,
    }
    manifest1 = markdownize_flow.fn(**run_kwargs)
    assert len(manifest1.entries) == 1
    assert call_count["n"] == 1

    # Second run on unchanged files — nothing should be re-processed.
    manifest2 = markdownize_flow.fn(**run_kwargs)
    assert len(manifest2.entries) == 1
    assert call_count["n"] == 1  # still 1, no new submissions
