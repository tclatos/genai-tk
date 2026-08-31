"""Mistral OCR batch processing: convert PDFs to Markdown via the Mistral OCR batch API."""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from genai_tk.extra.markdownize.mistral_ocr_converter import MistralOCRConverter
from genai_tk.workflow.markdownize.converters import _markitdown_text
from genai_tk.workflow.markdownize.routing import _FileToProcess, _output_paths, _write_markdown


class MistralOCRBatchProcessor:
    """Submit PDFs to Mistral's OCR *batch* API and return Markdown text per file."""

    def __init__(self, batch_size: int = 100):
        self.converter = MistralOCRConverter(batch_size=batch_size, use_batch_api=True)

    async def process_batch(self, file_paths: list[Path]) -> dict[str, str]:
        """Return a mapping of ``str(pdf_path)`` to its extracted Markdown text."""
        return await self.converter.batch_convert(file_paths)


def _ocr_pdfs_with_mistral(
    pdf_items: list[tuple[_FileToProcess, str]],
    output_dir: Path,
) -> list[tuple[str, str]]:
    """OCR every PDF in one Mistral batch job, writing Markdown at each original's path."""
    pdf_paths = [Path(pdf) for _, pdf in pdf_items]
    try:
        converter = MistralOCRConverter(use_batch_api=True)
        texts = asyncio.run(converter.batch_convert(pdf_paths))
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
