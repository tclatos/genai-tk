"""Prefect flow to merge multiple Markdown files into a single MERGED.md.

Merges all `*.md` files in a given directory (typically output from the
markdownize flow) into one consolidated document.  Files are sorted by
original document type (PDF first, then Word, Excel, HTML, CSV, JSON,
images) and alphabetically within each group — with "annex/appendix"-style
documents pushed to the end.

The generated file starts with a table of contents listing every merged
source with GitHub-compatible anchor links.

Typical usage::

    uv run cli workflow run merge_markdown --set base_dir=./output
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from loguru import logger
from prefect import flow
from pydantic import BaseModel, Field

from genai_tk.utils.file_patterns import resolve_config_path

# ---------------------------------------------------------------------------
# Extension priority for sorting (lower = appears first)
# ---------------------------------------------------------------------------
_EXTENSION_PRIORITY: dict[str, int] = {
    "pdf": 0,
    "docx": 1,
    "doc": 1,
    "pptx": 2,
    "ppt": 2,
    "xlsx": 3,
    "xls": 3,
    "html": 4,
    "htm": 4,
    "csv": 5,
    "json": 6,
    "jpg": 7,
    "jpeg": 7,
    "png": 7,
    "gif": 7,
    "bmp": 7,
}

# Patterns that indicate appendix/annex-style documents (case-insensitive)
_ANNEX_PATTERNS = re.compile(r"(annex|appendix|addendum|supplement|attachment)", re.IGNORECASE)


class MergeResult(BaseModel):
    """Result of a merge operation."""

    output_path: str
    file_count: int
    sections: list[str] = Field(default_factory=list)


def _extract_original_extension(md_filename: str) -> str:
    """Extract the original file extension from a markdownize-generated filename.

    Examples:
        review_xlsx.md → xlsx
        report_pdf.md → pdf
        notes_docx.md → docx
        plain.md → md
    """
    stem = Path(md_filename).stem  # e.g. "review_xlsx"
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].lower() in _EXTENSION_PRIORITY:
        return parts[1].lower()
    return "md"


def _original_display_name(md_filename: str) -> str:
    """Reconstruct a human-friendly display name with original extension.

    Examples:
        review_xlsx.md → review.xlsx
        report_pdf.md → report.pdf
        plain.md → plain.md
    """
    stem = Path(md_filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].lower() in _EXTENSION_PRIORITY:
        return f"{parts[0]}.{parts[1].lower()}"
    return md_filename


def _is_annex(filename: str) -> bool:
    """Return True if filename looks like an annex/appendix document."""
    return bool(_ANNEX_PATTERNS.search(filename))


def _sort_key(md_path: Path) -> tuple[int, int, str]:
    """Sort key: (extension_priority, is_annex, lowercase_name)."""
    ext = _extract_original_extension(md_path.name)
    priority = _EXTENSION_PRIORITY.get(ext, 99)
    annex_flag = 1 if _is_annex(md_path.stem) else 0
    return (priority, annex_flag, md_path.name.lower())


def _make_anchor(display_name: str) -> str:
    """Generate a GitHub-compatible anchor from a heading text.

    GitHub rules: lowercase, replace spaces with hyphens, strip non-alphanum
    except hyphens.
    """
    anchor = display_name.lower()
    anchor = re.sub(r"[^\w\s-]", "", anchor)
    anchor = re.sub(r"[\s]+", "-", anchor)
    return anchor


def _collect_md_files(base_dir: Path) -> list[Path]:
    """Collect all .md files in base_dir (non-recursive), excluding MERGED.md."""
    return sorted(
        (p for p in base_dir.glob("*.md") if p.name.upper() != "MERGED.MD"),
        key=_sort_key,
    )


def _build_merged_content(files: Iterable[Path]) -> tuple[str, list[str]]:
    """Build the merged markdown content with TOC.

    Returns:
        Tuple of (full_content, list_of_section_names).
    """
    file_list = list(files)
    sections: list[str] = []
    toc_lines: list[str] = []
    body_parts: list[str] = []

    for md_file in file_list:
        display_name = _original_display_name(md_file.name)
        anchor = _make_anchor(display_name)
        sections.append(display_name)
        toc_lines.append(f"- [{display_name}](#{anchor})")

    # Build TOC section
    toc = "# Merged Documents\n\n"
    toc += f"*{len(file_list)} documents merged.*\n\n"
    toc += "## Table of Contents\n\n"
    toc += "\n".join(toc_lines)
    toc += "\n\n---\n\n"

    # Build body with each file as a section
    for md_file in file_list:
        display_name = _original_display_name(md_file.name)
        content = md_file.read_text(encoding="utf-8").strip()

        body_parts.append(f"## {display_name}\n\n")
        body_parts.append(content)
        body_parts.append("\n\n---\n\n")

    full_content = toc + "".join(body_parts)
    return full_content, sections


@flow(name="merge_markdown")
def merge_markdown_flow(
    base_dir: str,
    *,
    output_filename: str = "MERGED.md",
) -> MergeResult:
    """Merge all Markdown files in a directory into a single document.

    Args:
        base_dir: Directory containing .md files to merge.
        output_filename: Name of the output merged file.

    Returns:
        MergeResult with path and merged file count.
    """
    resolved = resolve_config_path(base_dir)
    base_upath = Path(resolved)

    if not base_upath.is_dir():
        logger.error(f"Directory does not exist: {base_upath}")
        return MergeResult(output_path="", file_count=0)

    md_files = _collect_md_files(base_upath)

    if not md_files:
        logger.warning(f"No markdown files found in {base_upath}")
        return MergeResult(output_path="", file_count=0)

    logger.info(f"Merging {len(md_files)} markdown files from {base_upath}")

    content, sections = _build_merged_content(md_files)

    output_path = base_upath / output_filename
    output_path.write_text(content, encoding="utf-8")
    logger.success(f"Wrote merged document to {output_path} ({len(md_files)} files)")

    return MergeResult(
        output_path=str(output_path),
        file_count=len(md_files),
        sections=sections,
    )
