"""Mistral OCR document to Markdown converter supporting single-file and batch modes."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import tempfile
from pathlib import Path

from loguru import logger
from pydantic import Field

from genai_tk.extra.markdownize.base import DocumentConverter

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


class MistralOCRConverter(DocumentConverter):
    """Document converter using Mistral's OCR and Batch APIs."""

    api_key: str | None = Field(default=None, description="Mistral API key (defaults to MISTRAL_API_KEY env var)")
    model: str = Field(default="mistral-ocr-latest", description="Mistral OCR model name")
    batch_size: int = Field(default=100, description="Maximum files per batch API request")
    use_batch_api: bool = Field(default=True, description="Whether to use the Mistral Batch API for batch conversions")
    poll_interval_seconds: float = Field(default=2.0, description="Polling interval in seconds for batch jobs")
    max_poll_attempts: int = Field(default=300, description="Maximum polling attempts for batch jobs")

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
        client = self._get_client()
        document_url = self._document_data_url(path)

        ocr_response = client.ocr.process(
            model=self.model,
            document={"type": "document_url", "document_url": document_url},
        )
        return self._format_ocr_pages(ocr_response.pages)

    @staticmethod
    def _format_ocr_pages(pages: list) -> str:
        """Format Mistral OCR pages into a unified Markdown string."""
        parts: list[str] = []
        for page in pages:
            parts.append(f"## Page {page.index + 1}\n\n{page.markdown}\n\n")
        return "".join(parts)

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
        request = {
            "custom_id": str(index),
            "body": {"model": self.model, "document": {"type": "document_url", "document_url": document_url}},
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
