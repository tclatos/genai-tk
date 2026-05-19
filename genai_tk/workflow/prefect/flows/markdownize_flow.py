"""Prefect-powered markdown conversion for various file formats.

Converts documents (PDF, DOCX, PPTX, etc.) to Markdown.  Three backends:
- ``markitdown`` (default): PDF, DOCX, PPTX, images
- ``mistral``: Mistral OCR API, PDFs only (falls back to markitdown)
- ``edgeparse``: fast Rust engine, PDFs only (falls back to markitdown)

Typical usage::

    uv run cli workflow run markdownize \\
        --pathspec '**/*.pdf' --pathspec '!**/*_draft*' \\
        --to ./output
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner  # type: ignore[attr-defined]
from pydantic import BaseModel, Field
from upath import UPath

from genai_tk.utils.file_patterns import resolve_config_path, resolve_files
from genai_tk.workflow.flow_cache.manifest import ManifestCache


class MarkdownizeManifestEntry(BaseModel):
    """A single markdown conversion entry in the manifest."""

    source_hash: str
    output_path: str
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MarkdownizeManifest(BaseModel):
    """Manifest for markdown conversion results to avoid reprocessing."""

    entries: dict[str, MarkdownizeManifestEntry] = Field(default_factory=dict)

    def model_dump_json(self, **kwargs: Any) -> str:
        """Serialize manifest to JSON string."""
        return super().model_dump_json(**kwargs)


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


@dataclass(slots=True)
class _FileToProcess:
    path: UPath
    content_hash: str


def _prepare_files(
    files: Iterable[UPath],
    cache: ManifestCache,
    force: bool,
) -> tuple[list[_FileToProcess], int]:
    """Prepare files for processing, skipping unchanged entries in the cache."""
    from genai_tk.utils.hashing import buffer_digest

    to_process: list[_FileToProcess] = []
    skipped = 0

    for path in files:
        try:
            content_bytes = path.read_bytes()
            content_hash = buffer_digest(content_bytes)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Error reading {path}: {exc}")
            continue

        if cache.is_fresh(str(path), fingerprint=content_hash, force=force):
            skipped += 1
            logger.info(f"Skipping unchanged file: {path}")
            continue

        to_process.append(_FileToProcess(path=path, content_hash=content_hash))

    return to_process, skipped


def _is_markdownize_compatible(file_path: UPath) -> bool:
    """Check if file is compatible with any supported converter."""
    suffix = file_path.suffix.lower()
    # markitdown formats
    markitdown_formats = {".pdf", ".docx", ".pptx"}
    # Image formats (markitdown only)
    ocr_formats = {".jpeg", ".jpg", ".png", ".gif", ".bmp"}
    return suffix in (markitdown_formats | ocr_formats)


@task
async def _process_single_file_task(
    file_info: _FileToProcess,
    output_dir: str,
    root_dir: str,
    converter: str = "markitdown",
) -> tuple[str, str]:  # (source_path, relative_output_path)
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

    # Try edgeparse for PDFs if selected
    if converter == "edgeparse" and upath.suffix.lower() == ".pdf":
        try:
            import edgeparse

            logger.info(f"Processing {upath.name} with edgeparse")
            content = edgeparse.convert(str(upath), format="markdown")
        except Exception as e:
            logger.warning(f"edgeparse failed for {upath.name}: {str(e)}. Falling back to markitdown.")

    # Try Mistral OCR for PDFs if selected
    elif converter == "mistral" and upath.suffix.lower() == ".pdf":
        try:
            from genai_tk.workflow.loaders.mistral_ocr import mistral_ocr as mistral_ocr_func

            logger.info(f"Processing {upath.name} with Mistral OCR")
            ocr_response = mistral_ocr_func(upath, use_cache=False)

            # Convert OCR response to markdown
            content_parts = []
            for page in ocr_response.pages:
                content_parts.append(f"## Page {page.index + 1}\n\n")
                content_parts.append(page.markdown)
                content_parts.append("\n\n")

            content = "".join(content_parts)

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

    return str(upath), str(relative_output_path)


def _chunked[T](items: list[T], size: int) -> Iterable[list[T]]:
    if size <= 0:
        yield items
        return
    for i in range(0, len(items), size):
        yield items[i : i + size]


@flow(name="markdownize", task_runner=ConcurrentTaskRunner())  # type: ignore[call-arg]
def markdownize_flow(
    base_dir: str,
    output_dir: str,
    *,
    pathspecs: list[str] | None = None,
    batch_size: int = 5,
    force: bool = False,
    converter: str = "markitdown",
) -> MarkdownizeManifest:
    """Run markdownize as a Prefect flow.

    Args:
        base_dir: Root directory to walk.  Supports ``${paths.*}`` config vars.
        output_dir: Directory to write markdown files and manifest.
        pathspecs: Gitwildmatch patterns (``!`` prefix = exclude).  Defaults to
            common document extensions.
        batch_size: Number of files to process concurrently per batch.
        force: Reprocess files even if unchanged in manifest.
        converter: Backend -- ``"markitdown"`` (default), ``"mistral"``, or
            ``"edgeparse"``.

    Returns:
        Updated manifest with processing results.
    """
    if pathspecs is None:
        pathspecs = ["**/*.pdf", "**/*.docx", "**/*.pptx", "**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.gif", "**/*.bmp"]

    file_paths = resolve_files(base_dir, pathspecs=pathspecs)

    if not file_paths:
        logger.warning("No files found to process")
        return MarkdownizeManifest()

    logger.info(f"Discovered {len(file_paths)} files to process")

    resolved_output = resolve_config_path(output_dir)
    output_upath = UPath(resolved_output)
    output_upath.mkdir(parents=True, exist_ok=True)

    manifest_path = output_upath / "manifest.json"
    cache = ManifestCache.load(manifest_path)

    files = [UPath(p) for p in file_paths if _is_markdownize_compatible(UPath(p))]
    to_process, skipped = _prepare_files(files, cache, force=force)

    if skipped:
        logger.info(f"Skipped {skipped} unchanged files based on manifest")

    if not to_process:
        logger.info("No files left to process after manifest filtering")
        manifest = MarkdownizeManifest(
            entries={
                k: MarkdownizeManifestEntry(
                    source_hash=rec.fingerprint,
                    output_path=rec.outputs.get("output_path", ""),
                    processed_at=rec.processed_at,
                )
                for k, rec in cache.records.items()
            }
        )
        return manifest

    logger.info(f"Processing {len(to_process)} files")

    for batch in _chunked(to_process, batch_size):
        futures = [
            _process_single_file_task.submit(
                file_info,
                output_dir=str(output_upath),
                root_dir=resolved_output,
                converter=converter,
            )
            for file_info in batch
        ]

        for future in futures:
            result = future.result()  # type: ignore[misc]
            if asyncio.iscoroutine(result):
                source_path, relative_output_path = asyncio.run(result)  # type: ignore[misc]
            else:
                source_path, relative_output_path = result  # type: ignore[misc]

            file_hash = next((f.content_hash for f in to_process if str(f.path) == source_path), "")
            cache.record_success(
                key=source_path,
                fingerprint=file_hash,
                outputs={"output_path": relative_output_path},
            )

    cache.save(manifest_path)
    logger.success(f"Conversion completed. {len(to_process)} files processed, {skipped} skipped.")

    manifest = MarkdownizeManifest(
        entries={
            k: MarkdownizeManifestEntry(
                source_hash=rec.fingerprint,
                output_path=rec.outputs.get("output_path", ""),
                processed_at=rec.processed_at,
            )
            for k, rec in cache.records.items()
        }
    )
    return manifest
