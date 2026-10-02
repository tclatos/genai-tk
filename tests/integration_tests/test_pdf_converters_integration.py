"""Integration tests for document converters with a real PDF file."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
import pytest

from genai_tk.extra.markdownize.docling_converter import DoclingConverter
from genai_tk.extra.markdownize.factory import ConverterFactory
from genai_tk.extra.markdownize.lighton_ocr_converter import LightOnOCRConverter
from genai_tk.extra.markdownize.llm_converter import LLMConverter
from genai_tk.extra.markdownize.markitdown_converter import MarkItDownConverter
from genai_tk.extra.markdownize.mistral_ocr_converter import MistralOCRConverter
from genai_tk.workflow.markdownize import markdownize_flow

SAMPLE_PDF_URL = "https://sample-files.com/downloads/documents/pdf/basic-text.pdf"
LOCAL_SAMPLE_PDF = Path("/home/tcl/prj/genai-graph/tests/data/sample-pdf-a4-size.pdf")


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
async def test_mistral_ocr_with_image_extraction_real_pdf(tmp_path: Path) -> None:
    """Test Mistral OCR image extraction on a sample PDF from internet when API key is available."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY not found in environment")

    pdf_url = "https://raw.githubusercontent.com/mozilla/pdf.js/master/test/pdfs/tracemonkey.pdf"
    pdf_path = tmp_path / "tracemonkey.pdf"
    response = httpx.get(pdf_url, follow_redirects=True, timeout=30.0)
    response.raise_for_status()
    pdf_path.write_bytes(response.content)

    images_dir = tmp_path / "extracted_images"
    converter = MistralOCRConverter(
        use_batch_api=False,
        include_image_base64=True,
        images_dir=images_dir,
    )
    text = await converter.convert(pdf_path)
    assert len(text) > 100
    saved_images = list(images_dir.glob("*.*"))
    if saved_images:
        for img_file in saved_images:
            assert len(img_file.stem) == 8  # xxhash32 hex is 8 characters
            assert f"<!-- Image: {img_file.name}" in text


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


@pytest.fixture(scope="module")
def docling_sample_pdf() -> Path:
    """Return the local sample PDF used by Docling tests."""
    if not LOCAL_SAMPLE_PDF.exists():
        pytest.skip(f"Sample PDF not found: {LOCAL_SAMPLE_PDF}")
    return LOCAL_SAMPLE_PDF


@pytest.fixture(scope="module")
def docling_converted_text(docling_sample_pdf: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[str, Path]:
    """Convert the sample PDF once for the module with page markers and image extraction on."""
    pytest.importorskip("docling")
    images_dir = tmp_path_factory.mktemp("docling_images")
    converter = DoclingConverter(page_markers=True, images_dir=images_dir)
    text = asyncio.run(converter.convert(docling_sample_pdf))
    return text, images_dir


@pytest.mark.integration
def test_docling_pdf_conversion(docling_converted_text: tuple[str, Path]) -> None:
    """Test local Docling conversion on the sample PDF."""
    text, _ = docling_converted_text
    assert len(text) > 100
    assert "Sample PDF" in text
    assert "## Page " in text  # page_markers=True fixture


@pytest.mark.integration
def test_docling_image_extraction(docling_converted_text: tuple[str, Path]) -> None:
    """Test Docling picture extraction with xxhash32 naming and HTML comment markers."""
    text, images_dir = docling_converted_text
    saved_images = list(Path(images_dir).glob("*.png"))
    assert saved_images, "expected at least one extracted picture"
    for img_file in saved_images:
        assert len(img_file.stem) == 8  # xxhash32 hex is 8 characters
        assert f"<!-- Image: {img_file.name}" in text
    assert "![" in text


@pytest.mark.integration
def test_docling_table_markdown_default(docling_converted_text: tuple[str, Path]) -> None:
    """Test that Docling tables default to Markdown pipe tables."""
    text, _ = docling_converted_text
    assert "| Metric |" in text
    assert "<table" not in text


@pytest.mark.integration
def test_docling_table_format_html(docling_sample_pdf: Path) -> None:
    """Test that table_format='html' keeps tables as structured HTML."""
    pytest.importorskip("docling")
    converter = DoclingConverter(table_format="html")
    text = asyncio.run(converter.convert(docling_sample_pdf))
    assert "<table" in text.lower()
    assert "Metric" in text


@pytest.mark.integration
def test_docling_factory_and_profile_routing() -> None:
    """Test factory creation, supported extensions and the docling profile routing."""
    from genai_tk.workflow.markdownize import get_markdownize_profile

    converter = ConverterFactory.create("docling")
    assert isinstance(converter, DoclingConverter)
    assert ".pdf" in converter.supported_extensions()
    assert ".docx" in converter.supported_extensions()
    assert ".epub" in converter.supported_extensions()

    profile = get_markdownize_profile("docling")
    assert profile.select_route(Path("report.pdf")) == "docling"
    assert profile.select_route(Path("data.xlsx")) == "messy_xls"
    assert profile.select_route(Path("legacy.doc")) == "via_pdf"
    assert profile.select_route(Path("legacy.ppt")) == "via_pdf"
    assert profile.select_route(Path("page.html")) == "docling"
    assert profile.select_route(Path("photo.png")) == "docling"


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
