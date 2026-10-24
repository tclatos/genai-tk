"""Exercise image dependency upgrades without downloading model weights."""

import io
from pathlib import Path

import pytest
from PIL import Image

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("extension,image_format", [("png", "PNG"), ("jpg", "JPEG")])
def test_extracted_image_upscale_preserves_format(tmp_path: Path, extension: str, image_format: str) -> None:
    from genai_tk.extra.markdownize.image_describer import upscale_small_image

    source = tmp_path / f"chart.{extension}"
    with Image.new("RGB", (40, 20), color=(12, 64, 128)) as original:
        original.save(source, format=image_format)

    with Image.open(io.BytesIO(upscale_small_image(source))) as resized:
        assert (resized.size, resized.format) == ((120, 60), image_format)


@pytest.mark.requires_feature("streamlit")
def test_streamlit_resizes_png_bytes() -> None:
    # This dependency boundary uses Streamlit's image encoder directly so the
    # actual Pillow resize runs without starting a browser or Streamlit server.
    from streamlit.elements.lib.image_utils import _ensure_image_size_and_format
    from streamlit.elements.lib.layout_utils import LayoutConfig

    buffer = io.BytesIO()
    with Image.new("RGB", (80, 40), color=(12, 64, 128)) as original:
        original.save(buffer, format="PNG")

    encoded = _ensure_image_size_and_format(buffer.getvalue(), LayoutConfig(width=20), "PNG")

    with Image.open(io.BytesIO(encoded)) as resized:
        assert (resized.size, resized.format) == ((20, 10), "PNG")


@pytest.mark.requires_feature("docling")
async def test_docling_converts_local_html_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    from genai_tk.extra.markdownize.docling_converter import DoclingConverter

    source = tmp_path / "document.html"
    source.write_text(
        "<!doctype html><html><body><h1>Security compatibility</h1>"
        "<p>Offline image dependency check.</p></body></html>",
        encoding="utf-8",
    )
    # HTML uses Docling's simple pipeline, without PDF layout/OCR models.
    converter = DoclingConverter(ocr_engine="none", extract_images=False, recover_heading_levels=False)

    markdown = await converter.convert(source)

    assert "Security compatibility" in markdown
    assert "Offline image dependency check." in markdown


def test_fastembed_catalogue_matches_embedding_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    from fastembed import TextEmbedding

    # Both APIs inspect bundled model descriptions; constructing a model here
    # would download its weights and is intentionally outside this regression.
    model = TextEmbedding.list_supported_models()[0]

    assert TextEmbedding.get_embedding_size(model["model"]) == model["dim"]
