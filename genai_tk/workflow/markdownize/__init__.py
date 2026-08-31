"""Prefect-powered Markdown conversion driven by markdownize profiles."""

from __future__ import annotations

from genai_tk.extra.markdownize.selector import MarkdownizeProfile
from genai_tk.workflow.markdownize.config import (
    DEFAULT_PROFILE,
    get_markdownize_profile,
)
from genai_tk.workflow.markdownize.flow import markdownize_flow
from genai_tk.workflow.markdownize.manifest import MarkdownizeManifest, MarkdownizeManifestEntry

__all__ = [
    "DEFAULT_PROFILE",
    "MarkdownizeManifest",
    "MarkdownizeManifestEntry",
    "MarkdownizeProfile",
    "get_markdownize_profile",
    "markdownize_flow",
]
