"""Unit tests for ConverterFactory and ConverterSelector."""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_tk.extra.markdownize.base import DocumentConverter
from genai_tk.extra.markdownize.factory import ConverterFactory
from genai_tk.extra.markdownize.selector import ConverterRule, MarkdownizeProfile, _expand_pattern


def test_expand_pattern() -> None:
    assert _expand_pattern("*.pdf") == ["*.pdf"]
    assert _expand_pattern("**/*.{xlsx,xls}") == ["**/*.xlsx", "**/*.xls"]
    assert _expand_pattern("**/*.{docx,doc,odt}") == ["**/*.docx", "**/*.doc", "**/*.odt"]


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


def test_markdownize_profile_legacy_kwargs() -> None:
    profile = MarkdownizeProfile(
        ppt_converter="via_pdf",
        doc_converter="markitdown",
        excel_converter="messy_xls_parser",
        pdf_converter="mistral",
    )

    assert profile.select_route(Path("deck.pptx")) == "via_pdf"
    assert profile.select_route(Path("memo.docx")) == "markitdown"
    assert profile.select_route(Path("sheet.xlsx")) == "messy_xls_parser"
    assert profile.select_route(Path("scan.pdf")) == "mistral_ocr"
    assert profile.select_route(Path("notes.md")) == "copy"


def test_converter_factory_builtin_names() -> None:
    for name in ["markitdown", "messy_xls", "edgeparse", "mistral_ocr", "lighton_ocr", "anydoc", "llm"]:
        conv = ConverterFactory.create(name)
        assert isinstance(conv, DocumentConverter)


def test_converter_factory_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown document converter"):
        ConverterFactory.create("non_existent_converter_xyz")
