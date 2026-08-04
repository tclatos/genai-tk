"""Markdown text normalization and source-file provenance comments."""

from __future__ import annotations

import re
from pathlib import Path


def _normalize_markdown(content: str) -> str:
    """Collapse excessive blank lines and normalize whitespace.

    Reduces sequences of 3+ blank lines to 1, and strips trailing whitespace
    from each line (common in PDF conversions).
    """
    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in content.splitlines(keepends=False)]
    # Collapse 3+ consecutive blank lines to 1
    text = "\n".join(lines)
    text = re.sub(r"\n\n\n+", "\n\n", text)
    return text


_ORIGIN_COMMENT_PREFIX = "<!-- source:"


def _origin_comment(source_path: Path) -> str:
    """HTML-comment header recording the original file a Markdown file was converted from.

    Read back by `genai_graph.kg.query.markdown_tree_tui` to let the TUI offer opening the
    original document (PDF/DOCX/...) alongside the converted Markdown. Invisible when rendered.
    """
    return f"{_ORIGIN_COMMENT_PREFIX} {source_path.resolve()} -->\n\n"
