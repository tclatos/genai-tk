"""Unit tests for ConverterFactory and ConverterSelector."""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_tk.extra.markdownize.base import DocumentConverter
from genai_tk.extra.markdownize.factory import ConverterFactory
from genai_tk.extra.markdownize.selector import ConverterRule, MarkdownizeProfile, expand_brace_pattern


def test_expand_pattern() -> None:
    assert expand_brace_pattern("*.pdf") == ["*.pdf"]
    assert expand_brace_pattern("**/*.{xlsx,xls}") == ["**/*.xlsx", "**/*.xls"]
    assert expand_brace_pattern("**/*.{docx,doc,odt}") == ["**/*.docx", "**/*.doc", "**/*.odt"]


def test_converter_rule_matches() -> None:
    rule_excel = ConverterRule(pathspec="**/*.{xlsx,xls,ods}", converter="messy_xls")
    assert rule_excel.matches(Path("report.xlsx"))
    assert rule_excel.matches(Path("data/sub/report.xls"))
    assert not rule_excel.matches(Path("report.pdf"))

    rule_pdf = ConverterRule(pathspec="**/*.pdf", converter="mistral_ocr")
    assert rule_pdf.matches(Path("doc.pdf"))
    assert not rule_pdf.matches(Path("doc.docx"))


def test_markdownize_profile_rule_order() -> None:
    profile = MarkdownizeProfile(
        name="custom_order",
        rules=[
            ConverterRule(pathspec="**/special.pdf", converter="lighton_ocr"),
            ConverterRule(pathspec="**/*.pdf", converter="mistral_ocr"),
            ConverterRule(pathspec="**/*.xlsx", converter="messy_xls"),
            ConverterRule(pathspec="**/*", converter="markitdown"),
        ],
    )

    assert profile.select_route(Path("special.pdf")) == "lighton_ocr"
    assert profile.select_route(Path("other.pdf")) == "mistral_ocr"
    assert profile.select_route(Path("sheet.xlsx")) == "messy_xls"
    assert profile.select_route(Path("file.txt")) == "markitdown"
    assert profile.select_route(Path("readme.md")) == "copy"


def test_converter_factory_builtin_names() -> None:
    names = [
        "markitdown",
        "messy_xls",
        "edgeparse",
        "mistral_ocr",
        "lighton_ocr",
        "anydoc",
        "llm",
        # aliases
        "messy_xls_parser",
        "mistral",
        "lighton",
    ]
    for name in names:
        conv = ConverterFactory.create(name)
        assert isinstance(conv, DocumentConverter)


def test_converter_factory_mistral_with_image_extraction() -> None:
    conv = ConverterFactory.create(
        "mistral_ocr",
        include_image_base64=True,
        image_min_size=150,
        images_dir="custom_images_dir",
    )
    assert getattr(conv, "include_image_base64", False) is True
    assert getattr(conv, "image_min_size", None) == 150
    assert getattr(conv, "images_dir", None) == "custom_images_dir"


def test_converter_factory_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown document converter"):
        ConverterFactory.create("non_existent_converter_xyz")


def test_converter_factory_unknown_error_lists_available() -> None:
    with pytest.raises(KeyError, match="mistral_ocr") as exc_info:
        ConverterFactory.create("non_existent_converter_xyz")
    assert "Available converters" in str(exc_info.value)
