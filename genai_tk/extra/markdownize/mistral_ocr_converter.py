"""Mistral OCR document to Markdown converter supporting single-file and batch modes."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import Field

from genai_tk.extra.markdownize.base import DocumentConverter
from genai_tk.extra.markdownize.image_describer import describe_image_with_vlm
from genai_tk.extra.markdownize.table_processor import process_markdown_tables
from genai_tk.utils.hashing import buffer_digest

_MISTRAL_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".odt",
    ".odp",
    ".png",
    ".jpg",
    ".jpeg",
}


_ocr_request_lock = threading.Lock()
_ocr_next_start = 0.0


def _pace_request_start(min_interval: float) -> None:
    """Space OCR API request starts at least min_interval seconds apart across threads."""
    global _ocr_next_start
    with _ocr_request_lock:
        delay = _ocr_next_start - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        _ocr_next_start = time.monotonic() + min_interval


def _chunk_batch_files(paths: list[Path], max_count: int, max_bytes: int) -> list[list[Path]]:
    """Partition paths into batches bounded by both count and total raw byte size."""
    batches: list[list[Path]] = []
    current_batch: list[Path] = []
    current_bytes = 0

    for p in paths:
        try:
            size = p.stat().st_size
        except OSError:
            size = 0

        # If adding this file would exceed max_count or max_bytes (and current_batch is not empty)
        if current_batch and (len(current_batch) >= max_count or (current_bytes + size > max_bytes)):
            batches.append(current_batch)
            current_batch = []
            current_bytes = 0

        current_batch.append(p)
        current_bytes += size

    if current_batch:
        batches.append(current_batch)

    return batches


class MistralOCRConverter(DocumentConverter):
    """Document converter using Mistral's OCR and Batch APIs."""

    api_key: str | None = Field(default=None, description="Mistral API key (defaults to MISTRAL_API_KEY env var)")
    min_request_interval_seconds: float = Field(
        default=1.0,
        description="Minimum spacing between OCR API request starts across threads to avoid rate limits",
    )
    model: str = Field(default="mistral-ocr-latest", description="Mistral OCR model name")
    batch_size: int = Field(default=25, description="Maximum files per batch API request")
    max_batch_bytes: int = Field(
        default=100 * 1024 * 1024,
        description="Maximum total raw file size in bytes per batch request (default 100MB, safely under Mistral's 512MB limit)",
    )
    use_batch_api: bool = Field(default=True, description="Whether to use the Mistral Batch API for batch conversions")
    poll_interval_seconds: float = Field(default=2.0, description="Polling interval in seconds for batch jobs")
    max_poll_attempts: int = Field(default=300, description="Maximum polling attempts for batch jobs")
    include_image_base64: bool = Field(
        default=False, description="Whether to extract images as base64 from Mistral OCR"
    )
    image_min_size: int | None = Field(
        default=100,
        description="Minimum height and width of image in pixels to extract (filters out small logos/icons)",
    )
    images_dir: Path | str | None = Field(
        default=None, description="Directory to store extracted images named by xxhash32"
    )
    table_format: Literal["markdown", "html"] | None = Field(
        default=None, description="Table format for Mistral OCR ('markdown' or 'html')"
    )
    table_expanded: bool = Field(
        default=True,
        description="Whether to expand rowspan/colspan in HTML tables into rectangular Markdown tables",
    )
    describe_uncaptioned_images: bool = Field(
        default=False,
        description="Whether to call VLM to describe uncaptioned images larger than min_image_desc_size_bytes",
    )
    min_image_desc_size_bytes: int = Field(
        default=10 * 1024,
        description="Minimum byte size of uncaptioned image to describe with VLM (default 10KB)",
    )
    vlm_model: str = Field(
        default="gemini-2.5-flash@openrouter",
        description="VLM model identifier for uncaptioned image descriptions",
    )

    def supported_extensions(self) -> set[str]:
        """Return file extensions supported by Mistral OCR."""
        return _MISTRAL_SUPPORTED_EXTENSIONS

    def _resolve_api_key(self) -> str:
        """Resolve the Mistral API key from explicit setting or environment."""
        key = self.api_key or os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise EnvironmentError("Mistral API key not found. Set MISTRAL_API_KEY environment variable.")
        return key

    def _get_client(self):
        """Lazy import and instantiate the Mistral client."""
        try:
            from mistralai.client import Mistral
        except ImportError as e:
            raise ImportError("mistralai is required for MistralOCRConverter. Install with 'uv add mistralai'.") from e
        return Mistral(api_key=self._resolve_api_key())

    @staticmethod
    def _document_data_url(path: Path) -> str:
        """Encode file to a data URL with guessed or resolved MIME type."""
        content_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            mime_type = "application/pdf"
        elif suffix in (".jpg", ".jpeg"):
            mime_type = "image/jpeg"
        elif suffix == ".png":
            mime_type = "image/png"
        else:
            mime_type, _ = mimetypes.guess_type(str(path))
            mime_type = mime_type or "application/octet-stream"
        return f"data:{mime_type};base64,{content_b64}"

    async def convert(self, path: Path) -> str:
        """Convert a single document file to Markdown text using Mistral OCR."""
        return await asyncio.to_thread(self._sync_convert_single, path)

    def _sync_convert_single(self, path: Path) -> str:
        """Execute single-file Mistral OCR synchronously."""
        _pace_request_start(self.min_request_interval_seconds)
        client = self._get_client()
        document_url = self._document_data_url(path)

        ocr_kwargs: dict[str, Any] = {}
        if self.include_image_base64:
            ocr_kwargs["include_image_base64"] = True
        if self.image_min_size is not None:
            ocr_kwargs["image_min_size"] = self.image_min_size
        if self.table_format is not None:
            ocr_kwargs["table_format"] = self.table_format

        ocr_response = client.ocr.process(
            model=self.model,
            document={"type": "document_url", "document_url": document_url},
            **ocr_kwargs,
        )
        return self._format_ocr_pages(ocr_response.pages)

    def _format_ocr_pages(self, pages: list) -> str:
        """Format Mistral OCR pages into a unified Markdown string, optionally extracting images and inlining tables."""
        parts: list[str] = []
        for page in pages:
            page_index = getattr(page, "index", 0) if not isinstance(page, dict) else page.get("index", 0)
            page_markdown = getattr(page, "markdown", "") if not isinstance(page, dict) else page.get("markdown", "")
            page_images = getattr(page, "images", None) if not isinstance(page, dict) else page.get("images", None)
            page_tables = getattr(page, "tables", None) if not isinstance(page, dict) else page.get("tables", None)

            if self.include_image_base64 and page_images:
                page_markdown = self._process_page_images(page_markdown, page_images)

            if page_tables:
                page_markdown = self._process_page_tables(page_markdown, page_tables)

            page_markdown = process_markdown_tables(page_markdown, table_expanded=self.table_expanded)

            parts.append(f"## Page {page_index + 1}\n\n{page_markdown}\n\n")
        return "".join(parts)

    def _process_page_tables(self, markdown: str, tables: list) -> str:
        """Inline table content from OCR page.tables into markdown placeholders."""
        for tbl in tables:
            tbl_id = getattr(tbl, "id", "") if not isinstance(tbl, dict) else tbl.get("id", "")
            tbl_content = getattr(tbl, "content", "") if not isinstance(tbl, dict) else tbl.get("content", "")

            if not tbl_content:
                continue

            tbl_content = tbl_content.strip()

            replaced = False
            if tbl_id:
                escaped_id = re.escape(tbl_id)
                id_stem = re.escape(Path(tbl_id).stem)
                pattern = re.compile(
                    rf"!*\[(?P<alt>[^\]]*)\]\((?P<url>{escaped_id}|{id_stem}(?:\.html?)?)(?:\s+[\"'][^\"']*[\"'])?\)",
                    re.IGNORECASE,
                )
                if pattern.search(markdown):
                    markdown = pattern.sub(lambda _, c=tbl_content: f"\n\n{c}\n\n", markdown)
                    replaced = True

            if not replaced:
                markdown = f"{markdown}\n\n{tbl_content}\n\n"

        return markdown

    def _process_page_images(self, markdown: str, images: list) -> str:
        """Extract base64 images, compute xxhash32, save to disk, and annotate markdown."""
        target_dir = Path(self.images_dir) if self.images_dir else Path("images")

        for img in images:
            img_id = getattr(img, "id", "") if not isinstance(img, dict) else img.get("id", "")
            img_b64 = getattr(img, "image_base64", None) if not isinstance(img, dict) else img.get("image_base64", None)

            if not img_b64:
                continue

            b64_str = str(img_b64)
            inferred_ext = ""
            if b64_str.startswith("data:"):
                header, sep, rest = b64_str.partition(",")
                if sep:
                    b64_str = rest
                    if "image/jpeg" in header or "image/jpg" in header:
                        inferred_ext = ".jpg"
                    elif "image/png" in header:
                        inferred_ext = ".png"
                    elif "image/webp" in header:
                        inferred_ext = ".webp"
                    elif "image/gif" in header:
                        inferred_ext = ".gif"

            try:
                raw_bytes = base64.b64decode(b64_str)
            except Exception as exc:
                logger.warning(f"Failed to decode base64 for image '{img_id}': {exc}")
                continue

            # Determine extension
            ext = ""
            if img_id:
                suffix = Path(img_id).suffix.lower()
                if suffix in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".bmp"):
                    ext = suffix
            if not ext:
                if inferred_ext:
                    ext = inferred_ext
                elif raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                    ext = ".png"
                elif raw_bytes.startswith(b"\xff\xd8\xff"):
                    ext = ".jpg"
                elif raw_bytes.startswith((b"GIF87a", b"GIF89a")):
                    ext = ".gif"
                elif raw_bytes.startswith(b"RIFF") and len(raw_bytes) > 12 and raw_bytes[8:12] == b"WEBP":
                    ext = ".webp"
                else:
                    ext = ".jpg"

            img_hash = buffer_digest(raw_bytes, algorithm="xxh32")
            filename = f"{img_hash}{ext}"

            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                out_path = target_dir / filename
                out_path.write_bytes(raw_bytes)
                logger.debug(f"Saved extracted image '{img_id}' to {out_path} (xxhash32: {img_hash})")
            except Exception as exc:
                logger.error(f"Failed to save image {filename} to {target_dir}: {exc}")
                continue

            saved_target = str(target_dir / filename)
            comment_parts = [f"<!-- Image: {filename} (hash: {img_hash}) -->"]

            if self.describe_uncaptioned_images:
                vlm_desc = describe_image_with_vlm(
                    image_path=out_path,
                    vlm_model=self.vlm_model,
                    min_size_bytes=self.min_image_desc_size_bytes,
                    images_dir=target_dir,
                )
                if vlm_desc and vlm_desc.description:
                    comment_parts.append(f"<!-- Image Description: {vlm_desc.description} -->")
                    if vlm_desc.keywords:
                        kw_str = ", ".join(vlm_desc.keywords)
                        comment_parts.append(f"<!-- Image Keywords: {kw_str} -->")

            commentary = "\n".join(comment_parts)

            escaped_id = re.escape(img_id) if img_id else ""
            id_stem = re.escape(Path(img_id).stem) if img_id else ""

            replaced = False
            if escaped_id:
                pattern = re.compile(
                    rf"!\[(?P<alt>.*?)\]\((?P<url>{escaped_id}|{id_stem})(?P<title>\s+[\"'].*?[\"'])?\)"
                )

                def _replace_match(match: re.Match, _comm: str = commentary, _target: str = saved_target) -> str:
                    nonlocal replaced
                    replaced = True
                    alt = match.group("alt")
                    title = match.group("title") or ""
                    return f"{_comm}\n![{alt}]({_target}{title})"

                markdown = pattern.sub(_replace_match, markdown)

            if not replaced:
                markdown = f"{markdown}\n\n{commentary}\n![{img_id or filename}]({saved_target})\n"

        return markdown

    async def batch_convert(self, paths: list[Path]) -> dict[str, str]:
        """Convert a batch of files using Mistral Batch API when enabled."""
        if not paths:
            return {}

        if not self.use_batch_api or len(paths) == 1:
            return await super().batch_convert(paths)

        client = self._get_client()
        chunks = _chunk_batch_files(paths, max_count=self.batch_size, max_bytes=self.max_batch_bytes)
        logger.info(f"Submitting {len(chunks)} Mistral OCR batch job(s) for {len(paths)} file(s)")

        async def _run_chunk(chunk_files: list[Path], chunk_idx: int) -> dict[str, str]:
            logger.info(f"Submitting Mistral OCR batch #{chunk_idx + 1} of {len(chunk_files)} file(s)")
            requests = [self._prepare_batch_request(p, i) for i, p in enumerate(chunk_files)]
            try:
                return await self._submit_and_poll_batch(client, requests, chunk_files)
            except Exception as e:
                logger.warning(
                    f"Mistral OCR batch #{chunk_idx + 1} failed ({e}); falling back to single conversions for these files"
                )
                return await super(MistralOCRConverter, self).batch_convert(chunk_files)

        chunk_results = await asyncio.gather(*[_run_chunk(chunk, idx) for idx, chunk in enumerate(chunks)])
        results: dict[str, str] = {}
        for res in chunk_results:
            results.update(res)

        return results

    def _prepare_batch_request(self, file_path: Path, index: int) -> str:
        """Prepare a single JSONL batch request line."""
        document_url = self._document_data_url(file_path)
        suffix = file_path.suffix.lower()
        if suffix in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            doc_field: dict[str, Any] = {"type": "image_url", "image_url": document_url}
        else:
            doc_field = {"type": "document_url", "document_url": document_url}

        body: dict[str, Any] = {
            "model": self.model,
            "document": doc_field,
        }
        if self.include_image_base64:
            body["include_image_base64"] = True
        if self.image_min_size is not None:
            body["image_min_size"] = self.image_min_size
        if self.table_format is not None:
            body["table_format"] = self.table_format
        request = {
            "custom_id": str(index),
            "body": body,
        }
        return json.dumps(request)

    async def _submit_and_poll_batch(
        self,
        client,
        batch_requests: list[str],
        file_paths: list[Path],
    ) -> dict[str, str]:
        """Submit one batch job, poll until done, and return per-PDF Markdown text."""
        results: dict[str, str] = {}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for request in batch_requests:
                f.write(request + "\n")
            batch_file_path = f.name

        batch_data_id = None
        try:
            with open(batch_file_path, "rb") as f:
                batch_data = client.files.upload(
                    file={"file_name": os.path.basename(batch_file_path), "content": f},
                    purpose="batch",
                )
            batch_data_id = batch_data.id

            job = client.batch.jobs.create(
                input_files=[batch_data_id],
                model=self.model,
                endpoint="/v1/ocr",
                metadata={"job_type": "pdf_ocr_batch"},
            )

            logger.info(f"Polling Mistral batch job {job.id} for completion ({len(file_paths)} files)")
            if not await self._poll_job(client, job.id):
                raise RuntimeError(f"Mistral OCR batch job {job.id} failed to complete")

            retrieved_job = client.batch.jobs.get(job_id=job.id)
            if retrieved_job.output_file:
                results = self._parse_batch_results(client, retrieved_job.output_file, file_paths)

        finally:
            if os.path.exists(batch_file_path):
                os.remove(batch_file_path)
            if batch_data_id:
                try:
                    client.files.delete(file_id=batch_data_id)
                except Exception as exc:
                    logger.debug(f"Could not delete temporary batch input file {batch_data_id}: {exc}")

        return results

    async def _poll_job(self, client, job_id: str) -> bool:
        """Poll job status until completion."""
        for attempt in range(self.max_poll_attempts):
            job = client.batch.jobs.get(job_id=job_id)
            status = getattr(job, "status", None) or str(job.status)
            total = getattr(job, "total_requests", 0) or 0
            succeeded = getattr(job, "succeeded_requests", 0) or 0
            failed = getattr(job, "failed_requests", 0) or 0

            if status == "SUCCESS":
                logger.success(f"Mistral batch job {job_id} completed successfully ({succeeded}/{total} succeeded)")
                return True
            if status in ("FAILED", "TIMEOUT_EXCEEDED", "CANCELLED", "CANCELLATION_REQUESTED"):
                logger.error(f"Mistral batch job {job_id} failed with status {status} (failed: {failed}/{total})")
                return False

            if attempt == 0 or (attempt + 1) % 5 == 0:
                logger.info(f"Mistral batch job {job_id}: status={status}, progress={succeeded + failed}/{total}")
            await asyncio.sleep(self.poll_interval_seconds)

        logger.error(
            f"Mistral batch job {job_id} timed out after {self.max_poll_attempts * self.poll_interval_seconds}s"
        )
        return False

    def _parse_batch_results(self, client, output_file_id: str, file_paths: list[Path]) -> dict[str, str]:
        """Download batch output and map to str(file_path) -> Markdown text."""
        from mistralai.client.models import OCRResponse

        results: dict[str, str] = {}
        output_stream = client.files.download(file_id=output_file_id)
        if hasattr(output_stream, "read"):
            content_bytes = output_stream.read()
            response_content = content_bytes.decode("utf-8") if isinstance(content_bytes, bytes) else str(content_bytes)
        elif hasattr(output_stream, "text"):
            response_content = output_stream.text
        elif hasattr(output_stream, "content"):
            raw_c = output_stream.content
            response_content = raw_c.decode("utf-8") if isinstance(raw_c, bytes) else str(raw_c)
        else:
            response_content = str(output_stream)

        for line in response_content.strip().split("\n"):
            if not line:
                continue
            try:
                result = json.loads(line)
            except Exception as e:
                logger.warning(f"Failed to parse batch response line as JSON: {e}")
                continue

            custom_id_str = result.get("custom_id")
            if custom_id_str is None:
                continue
            try:
                custom_id = int(custom_id_str)
            except (ValueError, TypeError):
                continue

            if custom_id < 0 or custom_id >= len(file_paths):
                continue
            file_path = file_paths[custom_id]
            response_info = result.get("response", {})
            status_code = response_info.get("status_code", 200)
            if status_code != 200:
                logger.warning(f"Batch item for {file_path.name} failed with status {status_code}: {response_info}")
                continue

            response_body = response_info.get("body", {})
            try:
                ocr_response = OCRResponse.model_validate(response_body)
                results[str(file_path)] = self._format_ocr_pages(ocr_response.pages)
            except Exception as e:
                logger.error(f"Failed to parse OCR result for {file_path.name}: {e}")

        return results
