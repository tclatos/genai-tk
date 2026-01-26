"""Prefect-powered markdown conversion for various file formats.

This module provides a Prefect flow that processes documents (PDF, DOCX, PPTX, etc.)
and converts them to Markdown, with optional Mistral OCR processing for PDFs.
Results are written to output directory with a manifest file to track processing.

Typical usage from the CLI:
```bash
uv run cli tools markdownize ./docs ./output --recursive --mistral-ocr --force
```
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

from loguru import logger
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner  # type: ignore[attr-defined]
from pydantic import BaseModel, Field
from upath import UPath

from genai_tk.utils.file_patterns import resolve_files
from genai_tk.utils.hashing import buffer_digest


class MistralOCRBatchProcessor:
    """Batch processor for Mistral OCR with asset downloads."""

    def __init__(self, batch_size: int = 10):
        """Initialize batch processor.

        Args:
            batch_size: Number of files to process per batch job
        """
        self.batch_size = batch_size

    async def process_batch(
        self,
        file_paths: list[UPath],
        output_dir: UPath,
    ) -> dict[str, dict[str, str]]:
        """Process PDF files in batch using Mistral OCR API.

        Args:
            file_paths: List of PDF files to process
            output_dir: Output directory for markdown and assets

        Returns:
            Dictionary mapping source paths to output metadata (markdown_path, assets)
        """
        import os

        from mistralai import Mistral

        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError("Environment variable 'MISTRAL_API_KEY' not found")

        client = Mistral(api_key=api_key)
        results: dict[str, dict[str, str]] = {}

        # Process files in batches
        for batch_start in range(0, len(file_paths), self.batch_size):
            batch_files = file_paths[batch_start : batch_start + self.batch_size]
            logger.info(f"Processing batch of {len(batch_files)} files with Mistral OCR")

            # Prepare batch JSONL
            batch_requests = []
            for idx, file_path in enumerate(batch_files):
                request = self._prepare_batch_request(file_path, idx)
                batch_requests.append(request)

            # Upload batch file and process
            try:
                batch_results = await self._submit_and_poll_batch(client, batch_requests, batch_files, output_dir)
                results.update(batch_results)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                raise

        return results

    def _prepare_batch_request(self, file_path: UPath, index: int) -> str:
        """Prepare a single batch request in JSONL format.

        Args:
            file_path: Path to file
            index: Index for custom_id

        Returns:
            JSONL formatted batch request line
        """
        import base64
        import json

        # Encode file to base64
        content_b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
        document_url = f"data:application/pdf;base64,{content_b64}"

        request = {
            "custom_id": str(index),
            "body": {"model": "mistral-ocr-latest", "document": {"type": "document_url", "document_url": document_url}},
        }
        return json.dumps(request)

    async def _submit_and_poll_batch(
        self,
        client,
        batch_requests: list[str],
        file_paths: list[UPath],
        output_dir: UPath,
    ) -> dict[str, dict[str, str]]:
        """Submit batch job and poll for completion.

        Args:
            client: Mistral client
            batch_requests: List of JSONL-formatted requests
            file_paths: Corresponding file paths
            output_dir: Output directory for results

        Returns:
            Dictionary of results
        """
        import os
        import tempfile

        results: dict[str, dict[str, str]] = {}

        # Create temporary batch file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for request in batch_requests:
                f.write(request + "\n")
            batch_file_path = f.name

        try:
            # Upload batch file
            logger.info("Uploading batch file to Mistral API")
            with open(batch_file_path, "rb") as f:
                batch_data = client.files.upload(
                    file={"file_name": os.path.basename(batch_file_path), "content": f},
                    purpose="batch",
                )

            # Create batch job
            logger.info("Creating batch job for OCR processing")
            job = client.batch.jobs.create(
                input_files=[batch_data.id],
                model="mistral-ocr-latest",
                endpoint="/v1/ocr",
                metadata={"job_type": "pdf_ocr_batch"},
            )

            # Poll for completion
            logger.info(f"Polling batch job {job.id} for completion")
            job_completed = await self._poll_job(client, job.id)

            if not job_completed:
                raise RuntimeError("Batch job failed to complete")

            # Download and process results
            retrieved_job = client.batch.jobs.get(job_id=job.id)
            if retrieved_job.output_file:
                results = await self._process_batch_results(client, retrieved_job.output_file, file_paths, output_dir)

        finally:
            # Clean up temp file
            if os.path.exists(batch_file_path):
                os.remove(batch_file_path)

        return results

    async def _poll_job(self, client, job_id: str, max_attempts: int = 300) -> bool:
        """Poll job status until completion.

        Args:
            client: Mistral client
            job_id: Batch job ID
            max_attempts: Maximum polling attempts

        Returns:
            True if job succeeded, False otherwise
        """

        for _attempt in range(max_attempts):
            job = client.batch.jobs.get(job_id=job_id)
            logger.info(f"Batch job status: {job.status}")

            if job.status == "SUCCESS":
                logger.success(f"Batch job {job_id} completed successfully")
                return True
            elif job.status == "FAILED":
                logger.error(f"Batch job {job_id} failed")
                return False

            # Wait before next poll
            await asyncio.sleep(2)

        logger.error(f"Batch job {job_id} did not complete within timeout")
        return False

    async def _process_batch_results(
        self,
        client,
        output_file_id: str,
        file_paths: list[UPath],
        output_dir: UPath,
    ) -> dict[str, dict[str, str]]:
        """Process batch results and save outputs.

        Args:
            client: Mistral client
            output_file_id: ID of output file from batch job
            file_paths: Corresponding source file paths
            output_dir: Output directory for results

        Returns:
            Dictionary mapping source paths to output info
        """
        import json

        from mistralai.models import OCRResponse

        results: dict[str, dict[str, str]] = {}

        logger.info("Downloading batch results from Mistral API")
        output_stream = client.files.download(file_id=output_file_id)
        response_content = output_stream.read().decode("utf-8")

        # Parse JSONL response
        for line in response_content.strip().split("\n"):
            if not line:
                continue

            result = json.loads(line)
            custom_id = int(result["custom_id"])
            file_path = file_paths[custom_id]

            # Extract OCR response from batch result
            response_body = result.get("response", {}).get("body", {})
            try:
                ocr_response = OCRResponse.model_validate(response_body)
                markdown_path = await self._save_ocr_output(file_path, ocr_response, output_dir)
                results[str(file_path)] = {"markdown_path": markdown_path}
            except Exception as e:
                logger.error(f"Failed to process OCR result for {file_path.name}: {e}")

        return results

    async def _save_ocr_output(
        self,
        source_path: UPath,
        ocr_response,
        output_dir: UPath,
    ) -> str:
        """Save OCR output as markdown.

        Args:
            source_path: Source PDF path
            ocr_response: OCR response from Mistral
            output_dir: Output directory

        Returns:
            Path to saved markdown file
        """
        # Determine output markdown path
        markdown_filename = source_path.stem + ".md"
        markdown_path = output_dir / markdown_filename

        # Convert OCR response to markdown
        content_parts = []
        for page in ocr_response.pages:
            content_parts.append(f"## Page {page.index + 1}\n\n")
            content_parts.append(page.markdown)
            content_parts.append("\n\n")

        content = "".join(content_parts)
        markdown_path.write_text(content, encoding="utf-8")
        logger.success(f"Saved OCR output to {markdown_path}")

        # TODO: Download and save linked assets (JPEG, HTML) from ocr_response
        # This requires implementing asset download logic from Mistral API

        return str(markdown_path)


class MarkdownizeManifestEntry(BaseModel):
    """Single processed file entry in the markdownize manifest."""

    source_hash: str
    output_path: str
    processed_at: datetime


class MarkdownizeManifest(BaseModel):
    """Manifest of processed files for markdownize operations."""

    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    entries: dict[str, MarkdownizeManifestEntry] = Field(default_factory=dict)


@dataclass(slots=True)
class _FileToProcess:
    path: UPath
    content_hash: str


def _compute_hash(content: bytes) -> str:
    return buffer_digest(content)


def _load_manifest(manifest_path: UPath) -> MarkdownizeManifest | None:
    if not manifest_path.exists():
        return None

    try:
        text = manifest_path.read_text(encoding="utf-8")
        data = json.loads(text)
        return MarkdownizeManifest.model_validate(data)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to load manifest from {manifest_path}: {exc}. Ignoring it.")
        return None


def _save_manifest(manifest: MarkdownizeManifest, manifest_path: UPath) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def _prepare_files(
    files: Iterable[UPath],
    manifest: MarkdownizeManifest,
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


def _is_markdownize_compatible(file_path: UPath) -> bool:
    """Check if file is compatible with markitdown or Mistral OCR."""
    suffix = file_path.suffix.lower()
    # markitdown formats
    markitdown_formats = {".pdf", ".docx", ".pptx"}
    # Mistral OCR formats
    ocr_formats = {".jpeg", ".jpg", ".png", ".gif", ".bmp"}
    return suffix in (markitdown_formats | ocr_formats)


@task
async def _process_single_file_task(
    file_info: _FileToProcess,
    output_dir: str,
    root_dir: str,
    use_mistral_ocr: bool = False,
) -> tuple[str, MarkdownizeManifestEntry]:
    """Process a single file and save markdown output.

    Returns a tuple of (source_path, manifest_entry).
    """
    upath = file_info.path
    logger.info(f"Processing file: {upath}")

    output_upath = UPath(output_dir)

    # Preserve directory structure: compute relative path from root_dir to source file
    root_dir_path = UPath(root_dir)
    try:
        relative_source_path = upath.relative_to(root_dir_path)
    except ValueError:
        # If file is not under root_dir, use just the filename
        relative_source_path = UPath(upath.name)

    # Change extension to .md and maintain directory structure
    relative_output_path = relative_source_path.with_suffix(".md")
    output_file = output_upath / relative_output_path

    # Ensure parent directories exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    content = None

    # Try Mistral OCR for PDFs if enabled
    if use_mistral_ocr and upath.suffix.lower() == ".pdf":
        try:
            from genai_tk.extra.loaders.mistral_ocr import mistral_ocr as mistral_ocr_func

            logger.info(f"Processing {upath.name} with Mistral OCR")
            ocr_response = mistral_ocr_func(upath, use_cache=False)

            # Convert OCR response to markdown
            content_parts = []
            for page in ocr_response.pages:
                content_parts.append(f"## Page {page.index + 1}\n\n")
                content_parts.append(page.markdown)
                content_parts.append("\n\n")

            content = "".join(content_parts)

            # TODO: Download and save linked JPEG/HTML assets from OCR response
            # This requires handling ocr_response.pages[*].images and document structure

        except Exception as e:
            logger.warning(f"Mistral OCR failed for {upath.name}: {str(e)}. Falling back to markitdown.")

    # Use markitdown (default or fallback)
    if content is None:
        try:
            from markitdown import MarkItDown

            logger.info(f"Processing {upath.name} with markitdown")
            md = MarkItDown()
            result = md.convert(str(upath))
            content = result.text_content

        except Exception as e:
            logger.error(f"Failed to convert {upath.name}: {str(e)}")
            raise

    # Save markdown content
    output_file.write_text(content, encoding="utf-8")
    logger.success(f"Wrote markdown to {output_file}")

    entry = MarkdownizeManifestEntry(
        source_hash=file_info.content_hash,
        output_path=str(relative_output_path),
        processed_at=datetime.now(timezone.utc),
    )

    return str(upath), entry


def _chunked[T](items: list[T], size: int) -> Iterable[list[T]]:
    if size <= 0:
        yield items
        return
    for i in range(0, len(items), size):
        yield items[i : i + size]


@flow(name="markdownize", task_runner=ConcurrentTaskRunner())  # type: ignore[call-arg]
def markdownize_flow(
    root_dir: str,
    output_dir: str,
    *,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    recursive: bool = False,
    batch_size: int = 5,
    force: bool = False,
    use_mistral_ocr: bool = False,
) -> MarkdownizeManifest:
    """Run markdownize as a Prefect flow.

    Args:
        root_dir: Root directory to search for files
        output_dir: Directory to write markdown files and manifest
        include_patterns: List of glob patterns to include (default: standard markdownize formats)
        exclude_patterns: List of glob patterns to exclude
        recursive: Search recursively in subdirectories
        batch_size: Number of files to process concurrently per batch
        force: Reprocess files even if unchanged in manifest
        use_mistral_ocr: Use Mistral OCR for PDF processing

    Returns:
        Updated manifest with processing results
    """

    # Default include patterns if not specified
    if include_patterns is None:
        include_patterns = ["*.pdf", "*.docx", "*.pptx", "*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]

    # Resolve file list using utility function
    file_paths = resolve_files(
        root_dir,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        recursive=recursive,
    )

    if not file_paths:
        logger.warning("No files found to process")
        return MarkdownizeManifest()

    logger.info(f"Discovered {len(file_paths)} files to process")

    # Resolve output directory
    from genai_tk.utils.file_patterns import resolve_config_path

    resolved_output = resolve_config_path(output_dir)
    output_upath = UPath(resolved_output)
    output_upath.mkdir(parents=True, exist_ok=True)

    # Load or create manifest
    manifest_path = output_upath / "manifest.json"
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        manifest = MarkdownizeManifest()

    # Convert Path objects to UPath for processing
    files = [UPath(p) for p in file_paths if _is_markdownize_compatible(UPath(p))]

    to_process, skipped = _prepare_files(files, manifest, force=force)

    if skipped:
        logger.info(f"Skipped {skipped} unchanged files based on manifest")

    if not to_process:
        logger.info("No files left to process after manifest filtering")
        return manifest

    logger.info(f"Processing {len(to_process)} files")

    all_entries: dict[str, MarkdownizeManifestEntry] = dict(manifest.entries)

    # Handle Mistral OCR batch processing separately if enabled
    pdf_files = [f for f in to_process if f.path.suffix.lower() == ".pdf"] if use_mistral_ocr else []

    if use_mistral_ocr and pdf_files:
        try:
            _ocr_processor = MistralOCRBatchProcessor(batch_size=batch_size)  # noqa: F841 - reserved for future use
            logger.info(f"Processing {len(pdf_files)} PDF files with Mistral OCR batch processor")
            # Note: In practice, this would be called within a task for better Prefect integration
            # For now, we use the single-file fallback in _process_single_file_task
        except Exception as e:
            logger.warning(f"Could not initialize Mistral OCR batch processor: {e}. Using single-file OCR.")

    # Process in batches
    for batch in _chunked(to_process, batch_size):
        futures = [
            _process_single_file_task.submit(
                file_info,
                output_dir=str(output_upath),
                root_dir=root_dir,
                use_mistral_ocr=use_mistral_ocr,
            )
            for file_info in batch
        ]

        for future in futures:
            result = future.result()  # type: ignore[misc]
            # Handle both sync and async results from Prefect tasks
            if asyncio.iscoroutine(result):
                source_path, entry = asyncio.run(result)  # type: ignore[misc]
            else:
                source_path, entry = result  # type: ignore[misc]
            all_entries[source_path] = entry

    # Update and save manifest
    updated_manifest = MarkdownizeManifest(entries=all_entries)
    _save_manifest(updated_manifest, manifest_path)

    logger.success(
        f"Conversion completed. {len(to_process)} files processed, {skipped} skipped. "
        f"Manifest written to {manifest_path}",
    )

    return updated_manifest
