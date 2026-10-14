"""MarkItDown document to Markdown converter."""

from __future__ import annotations

import asyncio
from pathlib import Path

from genai_tk.extra.markdownize.base import DocumentConverter

_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".csv",
    ".json",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".xml",
    ".rss",
    ".txt",
    ".rtf",
    ".odt",
    ".ods",
    ".odp",
    ".epub",
}


class MarkItDownConverter(DocumentConverter):
    """Document converter backed by Microsoft MarkItDown."""

    def supported_extensions(self) -> set[str]:
        """Return file extensions supported by markitdown."""
        return _SUPPORTED_EXTENSIONS

    def check_dependencies(self) -> None:
        """Verify that markitdown is installed."""
        try:
            import markitdown  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "markitdown is required for MarkItDownConverter. Install with 'uv add markitdown'."
            ) from None

    async def convert(self, path: Path) -> str:
        """Convert a document file to Markdown text using markitdown."""
        return await asyncio.to_thread(self._sync_convert, path)

    def _sync_convert(self, path: Path) -> str:
        """Execute markitdown conversion synchronously."""
        try:
            from markitdown import MarkItDown, StreamInfo
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "markitdown is required for MarkItDownConverter. Install with 'uv add markitdown'."
            ) from None

        try:
            return MarkItDown().convert_local(str(path), stream_info=StreamInfo(charset="utf-8")).text_content
        except Exception:
            return MarkItDown().convert(str(path)).text_content
