"""Tests for markdownize Prefect flow."""

import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from upath import UPath

from genai_tk.extra.markdownize_prefect_flow import (
    MarkdownizeManifest,
    MarkdownizeManifestEntry,
    _compute_hash,
    _load_manifest,
    _prepare_files,
    _save_manifest,
    markdownize_flow,
)


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
    manifest_path = UPath(temp_dir / "manifest.json")

    # Create and save manifest
    manifest = MarkdownizeManifest(
        entries={
            "/test/file1.pdf": MarkdownizeManifestEntry(
                source_hash="hash1",
                output_path="file1.md",
            ),
            "/test/file2.docx": MarkdownizeManifestEntry(
                source_hash="hash2",
                output_path="file2.md",
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
    files = [UPath(f) for f in sample_files]
    hashes = {str(f): _compute_hash(f.read_bytes()) for f in files}

    # Create manifest with existing entries
    manifest = MarkdownizeManifest(
        entries={
            str(files[0]): MarkdownizeManifestEntry(
                source_hash=hashes[str(files[0])],
                output_path="file1.md",
            ),
            str(files[1]): MarkdownizeManifestEntry(
                source_hash=hashes[str(files[1])],
                output_path="file2.md",
            ),
        }
    )

    result, skipped = _prepare_files(files, manifest, force=False)

    assert len(result) == 0
    assert skipped == 2


def test_prepare_files_force_reprocess(temp_dir: Path, sample_files: list[Path]) -> None:
    """Test prepare_files with force flag reprocesses all files."""
    files = [UPath(f) for f in sample_files]
    hashes = {str(f): _compute_hash(f.read_bytes()) for f in files}

    # Create manifest with existing entries
    manifest = MarkdownizeManifest(
        entries={
            str(files[0]): MarkdownizeManifestEntry(
                source_hash=hashes[str(files[0])],
                output_path="file1.md",
            ),
        }
    )

    result, skipped = _prepare_files(files, manifest, force=True)

    # With force=True, should reprocess all files
    assert len(result) == 2
    assert skipped == 0


def test_markdownize_flow_basic(temp_dir: Path, sample_files: list[Path]) -> None:
    """Test basic markdownize flow execution."""
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Copy sample files to input directory
    for sample_file in sample_files:
        (input_dir / sample_file.name).write_bytes(sample_file.read_bytes())

    # Run flow with basic patterns (should not process as we lack dependencies)
    # This is a smoke test to ensure flow structure is correct
    try:
        manifest = markdownize_flow(
            root_dir=str(input_dir),
            output_dir=str(output_dir),
            include_patterns=["*.pdf"],
            recursive=False,
            batch_size=2,
            force=False,
            use_mistral_ocr=False,
        )

        assert isinstance(manifest, MarkdownizeManifest)
        # Manifest may be empty if markitdown is not installed
    except ImportError:
        # markitdown may not be installed in test environment
        pytest.skip("markitdown not installed")


def test_manifest_persistence(temp_dir: Path, sample_files: list[Path]) -> None:
    """Test manifest persists across runs."""
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Copy sample files
    for sample_file in sample_files:
        (input_dir / sample_file.name).write_bytes(sample_file.read_bytes())

    # First run
    manifest_path = UPath(output_dir) / "manifest.json"

    try:
        # First run should process files
        manifest1 = markdownize_flow(
            root_dir=str(input_dir),
            output_dir=str(output_dir),
            include_patterns=["*.pdf"],
            recursive=False,
            batch_size=2,
            force=False,
            use_mistral_ocr=False,
        )

        if manifest_path.exists():
            # Second run should skip unchanged files
            manifest2 = markdownize_flow(
                root_dir=str(input_dir),
                output_dir=str(output_dir),
                include_patterns=["*.pdf"],
                recursive=False,
                batch_size=2,
                force=False,
                use_mistral_ocr=False,
            )

            # Manifest file should exist
            assert manifest_path.exists()
            manifest_data = json.loads(manifest_path.read_text())
            assert "entries" in manifest_data

    except ImportError:
        pytest.skip("markitdown not installed")
