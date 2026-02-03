"""Prefect-powered PowerPoint to PDF conversion using LibreOffice.

This module provides a Prefect flow that converts PowerPoint files (PPT, PPTX, ODP)
to PDF using LibreOffice in headless mode. Supports parallel batch processing with
manifest-based tracking for incremental processing.

Typical usage from the CLI:
```bash
uv run cli tools ppt2pdf ./presentations ./output --recursive --force
```
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

from loguru import logger
from prefect import flow, task
from prefect.task_runners import ThreadPoolTaskRunner
from pydantic import BaseModel, Field
from upath import UPath

from genai_tk.utils.file_patterns import resolve_config_path, resolve_files
from genai_tk.utils.hashing import buffer_digest

# Supported PowerPoint-like file extensions
SUPPORTED_EXTENSIONS = {".ppt", ".pptx", ".odp"}


class Ppt2PdfManifestEntry(BaseModel):
    """Single processed file entry in the ppt2pdf manifest."""

    source_hash: str
    output_path: str
    processed_at: datetime


class Ppt2PdfManifest(BaseModel):
    """Manifest of processed files for ppt2pdf operations."""

    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    entries: dict[str, Ppt2PdfManifestEntry] = Field(default_factory=dict)


@dataclass(slots=True)
class _FileToProcess:
    path: UPath
    content_hash: str


@dataclass(slots=True)
class _TaskResult:
    """Result of processing a single file."""

    success: bool
    source_path: str
    entry: Ppt2PdfManifestEntry | None = None
    error: str | None = None


def _compute_hash(content: bytes) -> str:
    return buffer_digest(content)


def _load_manifest(manifest_path: UPath) -> Ppt2PdfManifest | None:
    if not manifest_path.exists():
        return None

    try:
        text = manifest_path.read_text(encoding="utf-8")
        data = json.loads(text)
        return Ppt2PdfManifest.model_validate(data)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to load manifest from {manifest_path}: {exc}. Ignoring it.")
        return None


def _save_manifest(manifest: Ppt2PdfManifest, manifest_path: UPath) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def _prepare_files(
    files: Iterable[UPath],
    manifest: Ppt2PdfManifest,
    force: bool,
) -> tuple[list[_FileToProcess], int]:
    """Prepare files for processing based on manifest state."""
    to_process: list[_FileToProcess] = []
    skipped = 0

    for path in files:
        try:
            content_bytes = path.read_bytes()
            content_hash = _compute_hash(content_bytes)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Error reading {path}: {exc}")
            continue

        key = str(path)
        existing = manifest.entries.get(key)

        if existing and not force and existing.source_hash == content_hash:
            skipped += 1
            logger.info(f"Skipping unchanged file: {path}")
            continue

        to_process.append(_FileToProcess(path=path, content_hash=content_hash))

    return to_process, skipped


def _is_ppt_compatible(file_path: UPath) -> bool:
    """Check if file is a PowerPoint-compatible format."""
    suffix = file_path.suffix.lower()
    return suffix in SUPPORTED_EXTENSIONS


def _convert_with_libreoffice(input_path: UPath, output_dir: UPath) -> UPath:
    """Convert a file to PDF using LibreOffice in headless mode.

    Args:
        input_path: Path to the input PowerPoint file
        output_dir: Directory to write the PDF output

    Returns:
        Path to the generated PDF file

    Raises:
        RuntimeError: If LibreOffice conversion fails
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build LibreOffice command
    cmd = [
        "libreoffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(input_path),
    ]

    logger.debug(f"Running LibreOffice command: {' '.join(cmd)}")

    # LibreOffice creates PDF with same stem name
    expected_output = output_dir / f"{input_path.stem}.pdf"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per file
        )

        # Log stdout/stderr for debugging
        if result.stdout:
            logger.debug(f"LibreOffice stdout: {result.stdout}")
        if result.stderr:
            logger.debug(f"LibreOffice stderr: {result.stderr}")

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or f"Return code {result.returncode}"
            raise RuntimeError(f"LibreOffice conversion failed: {error_msg}")

        # Check output file exists (LibreOffice may return 0 even on failure)
        if not expected_output.exists():
            error_info = result.stderr or result.stdout or "No output produced"
            raise RuntimeError(f"Expected output file not found: {expected_output}. LibreOffice output: {error_info}")

        return expected_output

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"LibreOffice conversion timed out for {input_path}") from exc


@task(log_prints=False)
def _process_single_file_task(
    file_info: _FileToProcess,
    output_dir: str,
    root_dir: str,
) -> _TaskResult:
    """Process a single PowerPoint file and convert to PDF.

    Returns a TaskResult indicating success or failure.
    """
    upath = file_info.path

    output_upath = UPath(output_dir)

    # Preserve directory structure: compute relative path from root_dir to source file
    root_dir_path = UPath(root_dir)
    try:
        relative_source_path = upath.relative_to(root_dir_path)
    except ValueError:
        # If file is not under root_dir, use just the filename
        relative_source_path = UPath(upath.name)

    # Compute output directory preserving structure
    relative_output_dir = relative_source_path.parent
    full_output_dir = output_upath / relative_output_dir

    try:
        # Convert using LibreOffice
        output_pdf = _convert_with_libreoffice(upath, full_output_dir)
        logger.success(f"✓ {upath.name}")

        # Compute relative output path for manifest
        relative_output_path = output_pdf.relative_to(output_upath)

        entry = Ppt2PdfManifestEntry(
            source_hash=file_info.content_hash,
            output_path=str(relative_output_path),
            processed_at=datetime.now(timezone.utc),
        )

        return _TaskResult(success=True, source_path=str(upath), entry=entry)

    except Exception as e:
        # Return failure result instead of raising
        error_msg = str(e)
        logger.warning(f"✗ {upath.name}: {error_msg}")
        return _TaskResult(success=False, source_path=str(upath), error=error_msg)


def _chunked[T](items: list[T], size: int) -> Iterable[list[T]]:
    if size <= 0:
        yield items
        return
    for i in range(0, len(items), size):
        yield items[i : i + size]


@flow(name="ppt2pdf", task_runner=ThreadPoolTaskRunner())  # type: ignore[call-arg]
def ppt2pdf_flow(
    root_dir: str,
    output_dir: str,
    *,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    recursive: bool = False,
    batch_size: int = 5,
    force: bool = False,
) -> Ppt2PdfManifest:
    """Run PowerPoint to PDF conversion as a Prefect flow.

    Uses LibreOffice in headless mode for conversion. Supports parallel
    batch processing with manifest-based incremental processing.

    Args:
        root_dir: Root directory to search for PowerPoint files
        output_dir: Directory to write PDF files and manifest
        include_patterns: List of glob patterns to include (default: *.ppt, *.pptx, *.odp)
        exclude_patterns: List of glob patterns to exclude
        recursive: Search recursively in subdirectories
        batch_size: Number of files to process concurrently per batch
        force: Reprocess files even if unchanged in manifest

    Returns:
        Updated manifest with processing results
    """

    # Default include patterns if not specified
    if include_patterns is None:
        include_patterns = ["*.ppt", "*.pptx", "*.odp"]

    # Resolve file list using utility function
    file_paths = resolve_files(
        root_dir,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        recursive=recursive,
    )

    if not file_paths:
        logger.warning("No PowerPoint files found to process")
        return Ppt2PdfManifest()

    logger.info(f"Discovered {len(file_paths)} files to process")

    # Resolve output directory
    resolved_output = resolve_config_path(output_dir)
    output_upath = UPath(resolved_output)
    output_upath.mkdir(parents=True, exist_ok=True)

    # Load or create manifest
    manifest_path = output_upath / "manifest.json"
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        manifest = Ppt2PdfManifest()

    # Filter to compatible files
    files = [UPath(p) for p in file_paths if _is_ppt_compatible(UPath(p))]

    to_process, skipped = _prepare_files(files, manifest, force=force)

    if skipped:
        logger.info(f"Skipped {skipped} unchanged files based on manifest")

    if not to_process:
        logger.info("No files left to process after manifest filtering")
        return manifest

    logger.info(f"Processing {len(to_process)} files with LibreOffice")

    all_entries: dict[str, Ppt2PdfManifestEntry] = dict(manifest.entries)
    failed_files: list[str] = []
    processed_count = 0

    # Process in batches
    for batch in _chunked(to_process, batch_size):
        futures = [
            _process_single_file_task.submit(
                file_info,
                output_dir=str(output_upath),
                root_dir=root_dir,
            )
            for file_info in batch
        ]

        for future in futures:
            result: _TaskResult = future.result()  # type: ignore[misc]
            if result.success and result.entry:
                all_entries[result.source_path] = result.entry
                processed_count += 1
            else:
                failed_files.append(result.source_path)

    # Update and save manifest
    updated_manifest = Ppt2PdfManifest(entries=all_entries)
    _save_manifest(updated_manifest, manifest_path)

    # Report results
    total = len(to_process)
    if failed_files:
        logger.warning(f"{len(failed_files)}/{total} files failed conversion")
    logger.success(f"Converted {processed_count}/{total} files (skipped {skipped} unchanged)")

    return updated_manifest
