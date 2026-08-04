"""Prefect-powered Markdown conversion driven by a single ``markdownize`` profile.

See :func:`genai_tk.workflow.markdownize.flow.markdownize_flow` for the full
docstring and usage examples.
"""

from __future__ import annotations

from genai_tk.workflow.markdownize.config import (
    DEFAULT_PROFILE,
    DocConverter,
    ExcelConverter,
    MarkdownizeProfile,
    PdfConverter,
    PptConverter,
    get_markdownize_profile,
)
from genai_tk.workflow.markdownize.flow import markdownize_flow
from genai_tk.workflow.markdownize.manifest import MarkdownizeManifest, MarkdownizeManifestEntry

__all__ = [
    "DEFAULT_PROFILE",
    "DocConverter",
    "ExcelConverter",
    "MarkdownizeManifest",
    "MarkdownizeManifestEntry",
    "MarkdownizeProfile",
    "PdfConverter",
    "PptConverter",
    "get_markdownize_profile",
    "markdownize_flow",
]
