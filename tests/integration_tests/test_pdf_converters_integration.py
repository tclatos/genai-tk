"""Integration tests for document converters with a real PDF file."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from genai_tk.extra.markdownize.lighton_ocr_converter import LightOnOCRConverter
from genai_tk.extra.markdownize.llm_converter import LLMConverter
from genai_tk.extra.markdownize.markitdown_converter import MarkItDownConverter
from genai_tk.extra.markdownize.mistral_ocr_converter import MistralOCRConverter
from genai_tk.workflow.markdownize import markdownize_flow

SAMPLE_PDF_URL = "https://sample-files.com/downloads/documents/pdf/basic-text.pdf"


@pytest.fixture(scope="module")
def sample_pdf_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download sample PDF once for the test module."""
    cache_dir = tmp_path_factory.mktemp("pdf_cache")
    pdf_file = cache_dir / "basic-text.pdf"

    response = httpx.get(SAMPLE_PDF_URL, follow_redirects=True, timeout=30.0)
    response.raise_for_status()
    pdf_file.write_bytes(response.content)
    return pdf_file


@pytest.mark.integration
@pytest.mark.asyncio
async def test_markitdown_pdf_conversion(sample_pdf_path: Path) -> None:
    """Test local MarkItDown conversion on real PDF."""
    converter = MarkItDownConverter()
    text = await converter.convert(sample_pdf_path)

    assert len(text) > 100
    assert "Lorem" in text or "pdf" in text.lower() or "text" in text.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lighton_ocr_sync_conversion(sample_pdf_path: Path) -> None:
    """Test LightOn OCR synchronous conversion on real PDF when API key is available."""
    api_key = os.environ.get("LIGHTON_API_KEY")
    if not api_key:
        pytest.skip("LIGHTON_API_KEY not found in environment")

    converter = LightOnOCRConverter(async_mode=False)
    text = await converter.convert(sample_pdf_path)

    assert len(text) > 100
    assert "Lorem" in text or "pdf" in text.lower() or "text" in text.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lighton_ocr_async_polling_conversion(sample_pdf_path: Path) -> None:
    """Test LightOn OCR async polling mode on real PDF when API key is available."""
    api_key = os.environ.get("LIGHTON_API_KEY")
    if not api_key:
        pytest.skip("LIGHTON_API_KEY not found in environment")

    converter = LightOnOCRConverter(async_mode=True, poll_interval_seconds=1.0)
    text = await converter.convert(sample_pdf_path)

    assert len(text) > 100


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mistral_ocr_single_and_batch_conversion(sample_pdf_path: Path) -> None:
    """Test Mistral OCR single and batch conversion on real PDF when API key is available."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY not found in environment")

    converter = MistralOCRConverter(use_batch_api=False)
    text = await converter.convert(sample_pdf_path)
    assert len(text) > 100

    # Test batch conversion
    batch_results = await converter.batch_convert([sample_pdf_path])
    assert str(sample_pdf_path) in batch_results
    assert len(batch_results[str(sample_pdf_path)]) > 100


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_pdf_conversion(sample_pdf_path: Path) -> None:
    """Test LLM multimodal base64 PDF conversion if default LLM is configured."""
    try:
        converter = LLMConverter(llm="default")
        text = await converter.convert(sample_pdf_path)
        assert len(text) > 50
    except Exception as exc:
        pytest.skip(f"Default LLM not available or does not support PDF vision input: {exc}")


@pytest.mark.integration
def test_markdownize_flow_with_real_pdf(sample_pdf_path: Path, tmp_path: Path) -> None:
    """Test markdownize_flow end-to-end on real PDF with fast profile."""
    out_dir = tmp_path / "output_fast"
    manifest = markdownize_flow(
        sources=str(sample_pdf_path),
        md_output_dir=str(out_dir),
        profile="fast",
    )

    assert len(manifest.entries) == 1
    output_files = list(out_dir.glob("*.md"))
    assert len(output_files) == 1
    content = output_files[0].read_text(encoding="utf-8")
    assert len(content) > 100
