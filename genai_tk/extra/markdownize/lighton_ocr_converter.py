"""LightOn OCR document to Markdown converter via REST API."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
from pathlib import Path

from loguru import logger
from pydantic import Field

from genai_tk.extra.markdownize.base import DocumentConverter

_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".pptx",
    ".ppt",
    ".odp",
    ".docx",
    ".odt",
    ".doc",
    ".html",
}


class LightOnOCRConverter(DocumentConverter):
    """Document converter using LightOn Parse API (sync and async polling modes)."""

    api_key: str | None = Field(default=None, description="LightOn API key (defaults to LIGHTON_API_KEY env var)")
    base_url: str = Field(default="https://api.lighton.ai", description="LightOn API base URL")
    async_mode: bool = Field(default=False, description="Whether to use async polling mode on LightOn API")
    timeout_seconds: float = Field(default=120.0, description="HTTP request timeout in seconds")
    poll_interval_seconds: float = Field(default=2.0, description="Polling interval in seconds for async jobs")
    max_poll_attempts: int = Field(default=300, description="Maximum polling attempts for async jobs")

    def supported_extensions(self) -> set[str]:
        """Return file extensions supported by LightOn OCR."""
        return _SUPPORTED_EXTENSIONS

    def _resolve_api_key(self) -> str:
        """Resolve the LightOn API key from explicit setting or environment."""
        key = (
            self.api_key
            or os.environ.get("LIGHTON_API_KEY")
            or os.environ.get("LIGHTON_TOKEN")
            or os.environ.get("TOKEN")
        )
        if not key:
            raise EnvironmentError("LightOn API key not found. Set LIGHTON_API_KEY environment variable.")
        return key

    async def convert(self, path: Path) -> str:
        """Convert a document file to Markdown text using LightOn API."""
        try:
            import httpx
        except ImportError as e:
            raise ImportError("httpx is required for LightOnOCRConverter. Install with 'uv add httpx'.") from e

        api_key = self._resolve_api_key()
        headers = {"Authorization": f"Bearer {api_key}"}
        endpoint = f"{self.base_url.rstrip('/')}/api/v3/parse"

        mime_type, _ = mimetypes.guess_type(str(path))
        mime_type = mime_type or "application/octet-stream"

        file_bytes = path.read_bytes()
        files = {"file": (path.name, file_bytes, mime_type)}
        data = {"options": json.dumps({"async": True})} if self.async_mode else {}

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(endpoint, headers=headers, files=files, data=data)
            if response.status_code == 200:
                return self._parse_completed_payload(response.json())
            if response.status_code == 202:
                job_data = response.json()
                job_id = job_data.get("id")
                if not job_id:
                    raise RuntimeError(f"LightOn async job accepted without job ID: {job_data}")
                return await self._poll_async_job(client, headers, job_id)

            raise RuntimeError(f"LightOn API returned error status {response.status_code}: {response.text}")

    async def _poll_async_job(self, client, headers: dict[str, str], job_id: str) -> str:
        """Poll LightOn async job endpoint until completion."""
        status_url = f"{self.base_url.rstrip('/')}/api/v3/parse/{job_id}"

        for _attempt in range(self.max_poll_attempts):
            res = await client.get(status_url, headers=headers)
            if res.status_code == 200:
                payload = res.json()
                status = payload.get("status")
                if status == "completed":
                    logger.success(f"LightOn async job {job_id} completed successfully")
                    return self._parse_completed_payload(payload)
                if status == "failed":
                    error_detail = payload.get("error", "Unknown error")
                    raise RuntimeError(f"LightOn async job {job_id} failed: {error_detail}")
            elif res.status_code != 202:
                raise RuntimeError(f"LightOn poll returned error status {res.status_code}: {res.text}")

            await asyncio.sleep(self.poll_interval_seconds)

        raise TimeoutError(f"LightOn async job {job_id} timed out after {self.max_poll_attempts} attempts")

    @staticmethod
    def _parse_completed_payload(payload: dict) -> str:
        """Extract and format Markdown text from LightOn completed response."""
        result = payload.get("result", {})
        pages = result.get("pages", [])
        if not pages:
            # Fallback if markdown is directly in result or root
            return result.get("markdown") or payload.get("markdown") or ""

        parts: list[str] = []
        for p in pages:
            idx = p.get("index", 1)
            md = p.get("markdown", "")
            if len(pages) > 1:
                parts.append(f"## Page {idx}\n\n{md}\n\n")
            else:
                parts.append(md)

        return "".join(parts).strip()
