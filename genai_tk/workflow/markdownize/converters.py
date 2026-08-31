"""File-to-Markdown text converters dispatch and fallbacks."""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from genai_tk.extra.markdownize.factory import ConverterFactory
from genai_tk.workflow.markdownize.excel import excel_to_markdown_messy_xls


def _markitdown_text(path: Path) -> str:
    """Convert any markitdown-supported file to Markdown text."""
    converter = ConverterFactory.create("markitdown")
    return asyncio.run(converter.convert(path))


def _edgeparse_text(path: Path) -> str | None:
    """Convert a PDF with edgeparse, returning None on failure."""
    try:
        converter = ConverterFactory.create("edgeparse")
        return asyncio.run(converter.convert(path))
    except Exception as e:
        logger.warning(f"edgeparse failed for {path.name}: {e}. Falling back to markitdown.")
        return None


def _convert_text(path: Path, route: str, pdf_converter: str = "markitdown") -> str:
    """Convert a single file to Markdown text using the specified route or converter.

    Args:
        path: File to convert.
        route: Converter name or route identifier.
        pdf_converter: Fallback converter name for 'pdf' route.

    Returns:
        Converted Markdown text.
    """
    if route == "messy_xls_parser" or route == "messy_xls":
        try:
            return excel_to_markdown_messy_xls(path)
        except Exception as e:
            logger.warning(f"messy_xls_parser failed for {path.name}: {e}. Falling back to markitdown.")
            return _markitdown_text(path)

    target = pdf_converter if route == "pdf" else route
    try:
        converter = ConverterFactory.create(target)
        return asyncio.run(converter.convert(path))
    except Exception as e:
        logger.warning(f"{target} failed for {path.name}: {e}. Falling back to markitdown.")
        return _markitdown_text(path)


__all__ = [
    "_convert_text",
    "_edgeparse_text",
    "_markitdown_text",
    "excel_to_markdown_messy_xls",
]
