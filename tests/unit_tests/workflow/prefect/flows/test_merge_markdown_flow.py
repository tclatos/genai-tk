"""Unit tests for the ``merge_markdown`` Prefect flow.

Covers the pure helper functions (sorting, anchors, content building) plus
smoke runs of :func:`merge_markdown_flow` against an ephemeral Prefect server.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_tk.workflow.prefect.flows.merge_markdown_flow import (
    MergeResult,
    _build_merged_content,
    _collect_md_files,
    _extract_original_extension,
    _is_annex,
    _make_anchor,
    _original_display_name,
    _sort_key,
    merge_markdown_flow,
)

# ---------------------------------------------------------------------------
# _extract_original_extension / _original_display_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("review_xlsx.md", "xlsx"),
        ("report_pdf.md", "pdf"),
        ("notes_docx.md", "docx"),
        ("deck_pptx.md", "pptx"),
        ("plain.md", "md"),
        ("readme.md", "md"),
    ],
)
def test_extract_original_extension(filename: str, expected: str) -> None:
    assert _extract_original_extension(filename) == expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("review_xlsx.md", "review.xlsx"),
        ("report_pdf.md", "report.pdf"),
        ("plain.md", "plain.md"),
    ],
)
def test_original_display_name(filename: str, expected: str) -> None:
    assert _original_display_name(filename) == expected


# ---------------------------------------------------------------------------
# _is_annex
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("annex_A.md", True),
        ("Appendix_1.md", True),
        ("addendum_notes.md", True),
        ("supplement.md", True),
        ("attachment_x.md", True),
        ("main_report.md", False),
        ("intro.md", False),
    ],
)
def test_is_annex(filename: str, expected: bool) -> None:
    assert _is_annex(filename) is expected


# ---------------------------------------------------------------------------
# _make_anchor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("display_name", "expected"),
    [
        ("report.pdf", "reportpdf"),
        ("My Cool Doc.docx", "my-cool-docdocx"),
        ("review--1.xlsx", "review--1xlsx"),
    ],
)
def test_make_anchor(display_name: str, expected: str) -> None:
    assert _make_anchor(display_name) == expected


# ---------------------------------------------------------------------------
# _sort_key / _collect_md_files
# ---------------------------------------------------------------------------


def test_sort_key_orders_by_extension_then_annex_then_name() -> None:
    pdf = Path("a_report_pdf.md")
    docx = Path("b_notes_docx.md")
    annex_pdf = Path("z_annex_pdf.md")
    ordered = sorted([annex_pdf, docx, pdf], key=_sort_key)
    # pdf (priority 0) before docx (priority 1); annex pdf pushed after regular pdf
    assert ordered == [pdf, annex_pdf, docx]


def test_collect_md_files_excludes_merged_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "b_pdf.md").write_text("b")
    (tmp_path / "a_docx.md").write_text("a")
    (tmp_path / "MERGED.md").write_text("ignore")
    (tmp_path / "notes.txt").write_text("not md")

    collected = _collect_md_files(tmp_path)
    names = [p.name for p in collected]
    assert "MERGED.md" not in names
    assert "notes.txt" not in names
    # pdf (priority 0) sorts before docx (priority 1) regardless of leading letter
    assert names == ["b_pdf.md", "a_docx.md"]


def test_collect_md_files_custom_output_filename_excluded(tmp_path: Path) -> None:
    (tmp_path / "one.md").write_text("1")
    (tmp_path / "CUSTOM.md").write_text("custom")
    collected = _collect_md_files(tmp_path, output_filename="CUSTOM.md")
    assert [p.name for p in collected] == ["one.md"]


# ---------------------------------------------------------------------------
# _build_merged_content
# ---------------------------------------------------------------------------


def test_build_merged_content_includes_toc_and_sections(tmp_path: Path) -> None:
    f1 = tmp_path / "intro_pdf.md"
    f1.write_text("# Intro\n\nHello world.")
    f2 = tmp_path / "notes_docx.md"
    f2.write_text("Some notes.")

    content, sections = _build_merged_content([f1, f2])

    assert sections == ["intro.pdf", "notes.docx"]
    assert "## Table of Contents" in content
    assert "[intro.pdf](#intropdf)" in content
    assert "[notes.docx](#notesdocx)" in content
    assert "2 documents merged." in content
    assert "# Intro" in content
    assert "Some notes." in content


def test_build_merged_content_empty_files() -> None:
    content, sections = _build_merged_content([])
    assert sections == []
    assert "0 documents merged." in content


# ---------------------------------------------------------------------------
# merge_markdown_flow — smoke runs via the prefect test harness
# ---------------------------------------------------------------------------


@pytest.mark.fake_models
def test_merge_markdown_flow_missing_dir_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    result = merge_markdown_flow(base_dir=str(missing))
    assert isinstance(result, MergeResult)
    assert result.file_count == 0
    assert result.output_path == ""


@pytest.mark.fake_models
def test_merge_markdown_flow_no_markdown_returns_empty(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not markdown")
    result = merge_markdown_flow(base_dir=str(tmp_path))
    assert result.file_count == 0
    assert result.output_path == ""


@pytest.mark.fake_models
def test_merge_markdown_flow_writes_merged_document(tmp_path: Path) -> None:
    (tmp_path / "intro_pdf.md").write_text("# Intro\n\nHello.")
    (tmp_path / "annex_pdf.md").write_text("Annex body.")
    (tmp_path / "notes_docx.md").write_text("Notes here.")

    result = merge_markdown_flow(base_dir=str(tmp_path))

    assert result.file_count == 3
    assert result.sections == ["intro.pdf", "annex.pdf", "notes.docx"]
    merged_path = Path(result.output_path)
    assert merged_path.exists()
    text = merged_path.read_text(encoding="utf-8")
    assert "## Table of Contents" in text
    assert "intro.pdf" in text
    assert "# Intro" in text
    assert "Annex body." in text


@pytest.mark.fake_models
def test_merge_markdown_flow_custom_output_filename(tmp_path: Path) -> None:
    (tmp_path / "a_pdf.md").write_text("alpha")
    result = merge_markdown_flow(base_dir=str(tmp_path), output_filename="ALL.md")
    assert result.file_count == 1
    assert Path(tmp_path / "ALL.md").exists()
