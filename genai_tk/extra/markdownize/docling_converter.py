"""Docling local document to Markdown converter.

Fully local, no-API-key alternative to API-based OCR converters such as Mistral OCR.
Backed by IBM's Docling toolkit: PDF layout analysis, TableFormer table structure
recognition, pluggable OCR engines and optional picture extraction with optional
VLM description.

Requires the optional `docling` dependency:

```bash
uv add "genai-tk[docling]"
# or
uv add docling
```

Docling model artifacts are downloaded on first use to the local Hugging Face cache
(`~/.cache/docling`), or can be pre-provisioned offline via `DOCLING_ARTIFACTS_PATH`.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger
from pydantic import Field

from genai_tk.extra.markdownize.base import DocumentConverter
from genai_tk.extra.markdownize.image_describer import describe_image_with_vlm
from genai_tk.extra.markdownize.table_processor import convert_html_table
from genai_tk.utils.hashing import buffer_digest
from genai_tk.utils.singleton import once

if TYPE_CHECKING:
    from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
    from docling_core.types.doc import DoclingDocument

_DOCLING_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".epub",
    ".odt",
    ".ods",
    ".odp",
    ".rtf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".csv",
    ".md",
    ".markdown",
    ".adoc",
    ".asciidoc",
}


class DoclingConverter(DocumentConverter):
    """Document converter backed by IBM Docling, running fully locally without an API key."""

    table_format: Literal["markdown", "html"] | None = Field(
        default=None,
        description="Table output format: 'markdown' converts tables through the lossless table processor, 'html' keeps Docling HTML tables, None behaves like 'markdown'",
    )
    table_expanded: bool = Field(
        default=True,
        description="Whether to expand rowspan/colspan in HTML tables into rectangular Markdown tables",
    )
    extract_images: bool = Field(default=True, description="Whether to extract embedded pictures to files")
    images_dir: Path | str | None = Field(
        default=None, description="Directory to store extracted images named by xxhash32"
    )
    image_min_size: int = Field(
        default=100,
        description="Minimum height and width of extracted pictures in pixels (filters out small logos/icons)",
    )
    images_scale: float = Field(
        default=2.0, description="Resolution scale for rendered pictures (1.0 ~ 72 DPI, 2.0 ~ 144 DPI)"
    )
    describe_uncaptioned_images: bool = Field(
        default=False,
        description="Whether to call VLM to describe uncaptioned images larger than min_image_desc_size_bytes",
    )
    min_image_desc_size_bytes: int = Field(
        default=10 * 1024,
        description="Minimum byte size of uncaptioned image to describe with VLM (default 10KB)",
    )
    vlm_model: str = Field(
        default="default_vlm",
        description="LLM tag or model id used to describe uncaptioned images (defaults to the 'default_vlm' config tag)",
    )
    page_markers: bool = Field(
        default=False, description="Whether to emit '## Page N' markers when the source page changes"
    )
    ocr_engine: Literal["easyocr", "tesseract", "none"] = Field(
        default="easyocr",
        description="OCR engine used for scanned content ('none' disables OCR for digital PDFs)",
    )
    tableformer_mode: Literal["fast", "accurate"] = Field(
        default="fast", description="TableFormer table structure recognition mode"
    )
    recover_heading_levels: bool = Field(
        default=True,
        description="Whether to recover PDF heading levels from bookmarks, outline numbering and font styling instead of flat level-1 headings",
    )
    max_concurrency: int = Field(
        default=1, description="Maximum number of documents converted concurrently in batch mode (CPU-bound)"
    )

    def supported_extensions(self) -> set[str]:
        """Return file extensions supported by Docling."""
        return _DOCLING_SUPPORTED_EXTENSIONS

    async def convert(self, path: Path) -> str:
        """Convert a single document file to Markdown text using Docling."""
        return await asyncio.to_thread(self._sync_convert, path)

    async def batch_convert(self, paths: list[Path]) -> dict[str, str]:
        """Convert a list of files to Markdown with bounded local concurrency."""
        if not paths:
            return {}
        if self.max_concurrency <= 1:
            return await super().batch_convert(paths)

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _convert_bounded(path: Path) -> str:
            async with semaphore:
                return await self.convert(path)

        outcomes = await asyncio.gather(*[_convert_bounded(p) for p in paths], return_exceptions=True)
        results: dict[str, str] = {}
        for path, outcome in zip(paths, outcomes, strict=False):
            if isinstance(outcome, Exception):
                logger.error(f"Failed to convert {path.name} with DoclingConverter: {outcome}")
                raise outcome
            results[str(path)] = outcome
        return results

    @once
    def _get_sdk_converter(
        ocr_engine: Literal["easyocr", "tesseract", "none"],
        tableformer_mode: Literal["fast", "accurate"],
        extract_images: bool,
        images_scale: float,
        recover_heading_levels: bool,
    ) -> Any:
        """Build and cache the Docling SDK converter for the given options (expensive model loading)."""
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                EasyOcrOptions,
                HeadingHierarchyOptions,
                PdfPipelineOptions,
                TableFormerMode,
                TesseractOcrOptions,
            )
            from docling.document_converter import DocumentConverter as SdkDocumentConverter
            from docling.document_converter import PdfFormatOption
        except ImportError as e:
            raise ImportError(
                "docling is required for DoclingConverter. Install with 'uv add docling' "
                "or 'uv add \"genai-tk[docling]\"'."
            ) from e

        options = PdfPipelineOptions()
        options.do_ocr = ocr_engine != "none"
        if ocr_engine == "easyocr":
            options.ocr_options = EasyOcrOptions()
        elif ocr_engine == "tesseract":
            options.ocr_options = TesseractOcrOptions()
        options.do_table_structure = True
        options.table_structure_options.mode = (
            TableFormerMode.ACCURATE if tableformer_mode == "accurate" else TableFormerMode.FAST
        )
        if extract_images:
            options.generate_picture_images = True
            options.images_scale = images_scale
        options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=recover_heading_levels)
        return SdkDocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})

    def _sync_convert(self, path: Path) -> str:
        """Execute Docling conversion synchronously."""
        from docling.datamodel.base_models import ConversionStatus

        sdk_converter = self._get_sdk_converter(
            ocr_engine=self.ocr_engine,
            tableformer_mode=self.tableformer_mode,
            extract_images=self.extract_images,
            images_scale=self.images_scale,
            recover_heading_levels=self.recover_heading_levels,
        )
        result = sdk_converter.convert(path, raises_on_error=True)
        if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            raise RuntimeError(
                f"Docling conversion failed for {path.name} (status: {result.status}, errors: {result.errors})"
            )
        return self._render_document(result.document)

    def _render_document(self, doc: DoclingDocument) -> str:
        """Render a DoclingDocument to Markdown in reading order."""
        from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

        serializer = MarkdownDocSerializer(doc=doc)
        chunks: list[str] = []
        current_page = 0
        for item, _level in doc.iterate_items():
            if self.page_markers:
                page_no = self._item_page_no(item)
                if page_no and page_no != current_page:
                    current_page = page_no
                    chunks.append(f"## Page {page_no}")
            rendered = self._render_item(item, doc, serializer)
            if rendered:
                chunks.append(rendered)
        if not chunks:
            return ""
        return "\n\n".join(chunks) + "\n"

    def _render_item(self, item: Any, doc: DoclingDocument, serializer: MarkdownDocSerializer) -> str:
        """Render a single document item to Markdown."""
        from docling_core.types.doc import ListItem, PictureItem, SectionHeaderItem, TableItem, TextItem

        if isinstance(item, TableItem):
            return self._render_table(item, doc)
        if isinstance(item, PictureItem):
            return self._render_picture(item, doc)
        if isinstance(item, SectionHeaderItem):
            level = min(max(int(item.level or 1), 1), 6)
            text = (item.text or "").strip()
            return f"{'#' * level} {text}" if text else ""
        if isinstance(item, (TextItem, ListItem)):
            try:
                result = serializer.serialize(item=item)
            except Exception as exc:
                logger.warning(f"Failed to serialize {type(item).__name__}: {exc}")
                return ""
            return (result.text or "").strip()
        return ""

    @staticmethod
    def _item_page_no(item: Any) -> int:
        """Return the source page number of an item, or 0 when unavailable."""
        prov = getattr(item, "prov", None)
        if prov:
            return int(getattr(prov[0], "page_no", 0) or 0)
        return 0

    def _render_table(self, item: Any, doc: DoclingDocument) -> str:
        """Render a table as Markdown or HTML depending on table_format."""
        try:
            html = item.export_to_html(doc=doc, add_caption=True).strip()
        except Exception as exc:
            logger.warning(f"Failed to export table to HTML: {exc}")
            return ""
        if not html:
            return ""
        if self.table_format == "html":
            return html
        return convert_html_table(html, table_expanded=self.table_expanded).strip()

    def _render_picture(self, item: Any, doc: DoclingDocument) -> str:
        """Save a picture to disk and render its Markdown reference with comments."""
        if not self.extract_images:
            return ""
        try:
            image = item.get_image(doc)
        except Exception as exc:
            logger.warning(f"Failed to extract picture: {exc}")
            return ""
        if image is None:
            return ""
        if self.image_min_size and (image.width < self.image_min_size or image.height < self.image_min_size):
            return ""

        target_dir = Path(self.images_dir) if self.images_dir else Path("images")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        raw_bytes = buffer.getvalue()
        img_hash = buffer_digest(raw_bytes, algorithm="xxh32")
        filename = f"{img_hash}.png"

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / filename
            out_path.write_bytes(raw_bytes)
            logger.debug(f"Saved extracted picture to {out_path} (xxhash32: {img_hash})")
        except Exception as exc:
            logger.error(f"Failed to save picture {filename} to {target_dir}: {exc}")
            return ""

        comment_parts = [f"<!-- Image: {filename} (hash: {img_hash}) -->"]
        if self.describe_uncaptioned_images:
            vlm_desc = describe_image_with_vlm(
                image_path=out_path,
                vlm_model=self.vlm_model,
                min_size_bytes=self.min_image_desc_size_bytes,
                images_dir=target_dir,
            )
            if vlm_desc and vlm_desc.description:
                comment_parts.append(f"<!-- Image Description: {vlm_desc.description} -->")
                if vlm_desc.keywords:
                    kw_str = ", ".join(vlm_desc.keywords)
                    comment_parts.append(f"<!-- Image Keywords: {kw_str} -->")

        commentary = "\n".join(comment_parts)
        return f"{commentary}\n![{filename}]({out_path})"
