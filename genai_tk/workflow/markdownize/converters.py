"""File-to-Markdown text converters (markitdown, edgeparse, spreadsheet parser dispatch)."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from genai_tk.workflow.markdownize.excel import excel_to_markdown_messy_xls


def _markitdown_text(path: Path) -> str:
    """Convert any markitdown-supported file to Markdown text."""
    from markitdown import MarkItDown

    return MarkItDown().convert(str(path)).text_content


def _edgeparse_text(path: Path) -> str | None:
    """Convert a PDF with edgeparse, returning None (to trigger fallback) on failure."""
    try:
        import edgeparse

        return edgeparse.convert(str(path), format="markdown")
    except Exception as e:
        logger.warning(f"edgeparse failed for {path.name}: {e}. Falling back to markitdown.")
        return None


def _convert_text(path: Path, route: str, pdf_converter: str) -> str:
    """Convert a single file to Markdown text for a non-Mistral route.

    ``route`` is one of ``messy_xls_parser`` / ``pdf`` / ``markitdown``. The
    ``pdf`` route honours ``pdf_converter`` (``edgeparse`` with markitdown
    fallback, or ``markitdown``); Mistral PDFs are handled in the flow via the
    batch API.
    """
    if route == "messy_xls_parser":
        try:
            return excel_to_markdown_messy_xls(path)
        except Exception as e:
            logger.warning(f"messy_xls_parser failed for {path.name}: {e}. Falling back to markitdown.")
            return _markitdown_text(path)
    if route == "pdf" and pdf_converter == "edgeparse":
        text = _edgeparse_text(path)
        if text is not None:
            return text
    return _markitdown_text(path)
