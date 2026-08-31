"""EdgeParse PDF to Markdown converter."""

from __future__ import annotations

import asyncio
from pathlib import Path

from genai_tk.extra.markdownize.base import DocumentConverter

_PDF_EXTENSIONS = {".pdf"}


class EdgeParseConverter(DocumentConverter):
    """PDF converter backed by edgeparse."""

    def supported_extensions(self) -> set[str]:
        """Return file extensions supported by edgeparse."""
        return _PDF_EXTENSIONS

    async def convert(self, path: Path) -> str:
        """Convert a PDF file to Markdown text using edgeparse."""
        return await asyncio.to_thread(self._sync_convert, path)

    def _sync_convert(self, path: Path) -> str:
        """Execute edgeparse conversion synchronously."""
        try:
            import edgeparse
        except ImportError as e:
            raise ImportError("edgeparse is required for EdgeParseConverter. Install with 'uv add edgeparse'.") from e

        return edgeparse.convert(str(path), format="markdown")
