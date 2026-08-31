"""Spreadsheet to Markdown converter handling complex and messy tables."""

from __future__ import annotations

import asyncio
from pathlib import Path

from genai_tk.extra.markdownize.base import DocumentConverter
from genai_tk.workflow.markdownize.excel import excel_to_markdown_messy_xls

_EXCEL_EXTENSIONS = {".xlsx", ".xls", ".ods"}


class MessyExcelConverter(DocumentConverter):
    """Spreadsheet converter designed for merged headers and multi-table sheets."""

    def supported_extensions(self) -> set[str]:
        """Return supported spreadsheet file extensions."""
        return _EXCEL_EXTENSIONS

    async def convert(self, path: Path) -> str:
        """Convert a spreadsheet to structured Markdown text."""
        return await asyncio.to_thread(excel_to_markdown_messy_xls, path)
