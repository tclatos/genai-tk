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


class MistralOCRConverter(DocumentConverter):
    """Document converter using Mistral's OCR and Batch APIs."""

    api_key: str | None = Field(default=None, description="Mistral API key (defaults to MISTRAL_API_KEY env var)")
    min_request_interval_seconds: float = Field(
        default=1.0,
        description="Minimum spacing between OCR API request starts across threads to avoid rate limits",
    )
    model: str = Field(default="mistral-ocr-latest", description="Mistral OCR model name")
    batch_size: int = Field(default=100, description="Maximum files per batch API request")
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
        """Format Mistral OCR pages into a unified Markdown string, optionally extracting images."""
        parts: list[str] = []
        for page in pages:
            page_index = getattr(page, "index", 0) if not isinstance(page, dict) else page.get("index", 0)
            page_markdown = getattr(page, "markdown", "") if not isinstance(page, dict) else page.get("markdown", "")
            page_images = getattr(page, "images", None) if not isinstance(page, dict) else page.get("images", None)

            if self.include_image_base64 and page_images:
                page_markdown = self._process_page_images(page_markdown, page_images)

            page_markdown = process_markdown_tables(page_markdown)

            parts.append(f"## Page {page_index + 1}\n\n{page_markdown}\n\n")
        return "".join(parts)

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
        results: dict[str, str] = {}

        for start in range(0, len(paths), self.batch_size):
            batch_files = paths[start : start + self.batch_size]
            logger.info(f"Submitting Mistral OCR batch of {len(batch_files)} file(s)")
            requests = [self._prepare_batch_request(p, i) for i, p in enumerate(batch_files)]
            batch_results = await self._submit_and_poll_batch(client, requests, batch_files)
            results.update(batch_results)

        return results

    def _prepare_batch_request(self, file_path: Path, index: int) -> str:
        """Prepare a single JSONL batch request line."""
        document_url = self._document_data_url(file_path)
        body: dict[str, Any] = {
            "model": self.model,
            "document": {"type": "document_url", "document_url": document_url},
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

        try:
            with open(batch_file_path, "rb") as f:
                batch_data = client.files.upload(
                    file={"file_name": os.path.basename(batch_file_path), "content": f},
                    purpose="batch",
                )

            job = client.batch.jobs.create(
                input_files=[batch_data.id],
                model=self.model,
                endpoint="/v1/ocr",
                metadata={"job_type": "pdf_ocr_batch"},
            )

            logger.info(f"Polling Mistral batch job {job.id} for completion")
            if not await self._poll_job(client, job.id):
                raise RuntimeError(f"Mistral OCR batch job {job.id} failed to complete")

            retrieved_job = client.batch.jobs.get(job_id=job.id)
            if retrieved_job.output_file:
                results = self._parse_batch_results(client, retrieved_job.output_file, file_paths)

        finally:
            if os.path.exists(batch_file_path):
                os.remove(batch_file_path)

        return results

    async def _poll_job(self, client, job_id: str) -> bool:
        """Poll job status until completion."""
        for _attempt in range(self.max_poll_attempts):
            job = client.batch.jobs.get(job_id=job_id)
            if job.status == "SUCCESS":
                logger.success(f"Mistral batch job {job_id} completed successfully")
                return True
            if job.status == "FAILED":
                logger.error(f"Mistral batch job {job_id} failed")
                return False
            await asyncio.sleep(self.poll_interval_seconds)

        logger.error(f"Mistral batch job {job_id} timed out")
        return False

    def _parse_batch_results(self, client, output_file_id: str, file_paths: list[Path]) -> dict[str, str]:
        """Download batch output and map to str(file_path) -> Markdown text."""
        from mistralai.client.models import OCRResponse

        results: dict[str, str] = {}
        output_stream = client.files.download(file_id=output_file_id)
        response_content = output_stream.read().decode("utf-8")

        for line in response_content.strip().split("\n"):
            if not line:
                continue
            result = json.loads(line)
            file_path = file_paths[int(result["custom_id"])]
            response_body = result.get("response", {}).get("body", {})
            try:
                ocr_response = OCRResponse.model_validate(response_body)
                results[str(file_path)] = self._format_ocr_pages(ocr_response.pages)
            except Exception as e:
                logger.error(f"Failed to parse OCR result for {file_path.name}: {e}")

        return results
