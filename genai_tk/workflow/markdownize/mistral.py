"""Mistral OCR batch processing: convert PDFs to Markdown via the Mistral OCR batch API."""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from genai_tk.workflow.markdownize.converters import _markitdown_text
from genai_tk.workflow.markdownize.routing import _FileToProcess, _output_paths, _write_markdown


class MistralOCRBatchProcessor:
    """Submit PDFs to Mistral's OCR *batch* API and return Markdown text per file.

    Batching every PDF into a single job is markedly cheaper than per-file OCR
    calls, at the cost of polling until the job finishes.
    """

    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size

    async def process_batch(self, file_paths: list[Path]) -> dict[str, str]:
        """Return a mapping of ``str(pdf_path)`` to its extracted Markdown text."""
        import os

        from mistralai.client import Mistral

        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError("Environment variable 'MISTRAL_API_KEY' not found")

        client = Mistral(api_key=api_key)
        results: dict[str, str] = {}
        for start in range(0, len(file_paths), self.batch_size):
            batch_files = file_paths[start : start + self.batch_size]
            logger.info(f"Submitting Mistral OCR batch of {len(batch_files)} PDF(s)")
            requests = [self._prepare_batch_request(p, i) for i, p in enumerate(batch_files)]
            results.update(await self._submit_and_poll_batch(client, requests, batch_files))
        return results

    def _prepare_batch_request(self, file_path: Path, index: int) -> str:
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
        file_paths: list[Path],
    ) -> dict[str, str]:
        """Submit one batch job, poll until done, and return per-PDF Markdown text."""
        import os
        import tempfile

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
                model="mistral-ocr-latest",
                endpoint="/v1/ocr",
                metadata={"job_type": "pdf_ocr_batch"},
            )

            logger.info(f"Polling Mistral batch job {job.id} for completion")
            if not await self._poll_job(client, job.id):
                raise RuntimeError("Mistral OCR batch job failed to complete")

            retrieved_job = client.batch.jobs.get(job_id=job.id)
            if retrieved_job.output_file:
                results = self._parse_results(client, retrieved_job.output_file, file_paths)

        finally:
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

    def _parse_results(self, client, output_file_id: str, file_paths: list[Path]) -> dict[str, str]:
        """Download the batch output and return ``str(pdf_path) -> Markdown text``."""
        import json

        from mistralai.client.models import OCRResponse

        results: dict[str, str] = {}

        logger.info("Downloading Mistral batch results")
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
                results[str(file_path)] = _ocr_response_to_markdown(ocr_response)
            except Exception as e:
                logger.error(f"Failed to parse OCR result for {file_path.name}: {e}")

        return results


def _ocr_response_to_markdown(ocr_response) -> str:
    """Join a Mistral OCR response's pages into a single Markdown string."""
    parts: list[str] = []
    for page in ocr_response.pages:
        parts.append(f"## Page {page.index + 1}\n\n{page.markdown}\n\n")
    return "".join(parts)


def _ocr_pdfs_with_mistral(
    pdf_items: list[tuple[_FileToProcess, str]],
    output_dir: Path,
) -> list[tuple[str, str]]:
    """OCR every PDF in one Mistral batch job, writing Markdown at each original's path."""
    pdf_paths = [Path(pdf) for _, pdf in pdf_items]
    try:
        texts = asyncio.run(MistralOCRBatchProcessor().process_batch(pdf_paths))
    except Exception as e:
        logger.warning(f"Mistral batch OCR failed ({e}); falling back to markitdown for PDFs.")
        texts = {}

    results: list[tuple[str, str]] = []
    for file_info, pdf in pdf_items:
        rel_out, out_abs = _output_paths(file_info.path, file_info.root, output_dir)
        text = texts.get(str(pdf))
        if text is None:
            logger.info(f"markitdown fallback for {Path(pdf).name}")
            text = _markitdown_text(Path(pdf))
        _write_markdown(out_abs, file_info.path, text)
        results.append((str(file_info.path), str(rel_out)))
    return results
