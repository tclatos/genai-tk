"""AnyDoc document to Markdown converter using Firecrawl's anydoc engine."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from pydantic import Field

from genai_tk.extra.markdownize.base import DocumentConverter

_ANYDOC_EXTENSIONS = {
    ".doc",
    ".docx",
    ".docm",
    ".ppt",
    ".pptx",
    ".pps",
    ".pot",
    ".pptm",
    ".ppsx",
    ".ppsm",
    ".xls",
    ".xlsx",
    ".xlsm",
    ".xlsb",
    ".odt",
    ".ods",
    ".odp",
    ".rtf",
    ".epub",
    ".csv",
    ".pdf",
}


class AnyDocConverter(DocumentConverter):
    """Document converter backed by Firecrawl anydoc Rust library."""

    ocr: str | None = Field(default=None, description="OCR mode (e.g. 'hosted' for Firecrawl Parse)")
    api_key: str | None = Field(default=None, description="Firecrawl API key (defaults to FIRECRAWL_API_KEY)")
    api_url: str | None = Field(default=None, description="Custom Firecrawl Parse API URL")

    def supported_extensions(self) -> set[str]:
        """Return file extensions supported by anydoc."""
        return _ANYDOC_EXTENSIONS

    async def convert(self, path: Path) -> str:
        """Convert a document file to Markdown text using anydoc."""
        return await asyncio.to_thread(self._sync_convert, path)

    def _sync_convert(self, path: Path) -> str:
        """Execute anydoc conversion synchronously."""
        try:
            import anydoc
        except ImportError as e:
            raise ImportError(
                "firecrawl-anydoc is required for AnyDocConverter. Install with 'uv add firecrawl-anydoc'."
            ) from e

        kwargs: dict = {}
        if self.ocr:
            kwargs["ocr"] = self.ocr
            key = self.api_key or os.environ.get("FIRECRAWL_API_KEY")
            if key:
                kwargs["api_key"] = key
            url = self.api_url or os.environ.get("FIRECRAWL_API_URL")
            if url:
                kwargs["api_url"] = url

        return anydoc.to_markdown(str(path), **kwargs)
