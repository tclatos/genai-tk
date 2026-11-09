"""Textual screens and view layouts."""

from __future__ import annotations

from genai_tk.ui.textual.screens.document_modal import DocumentViewerModal
from genai_tk.ui.textual.screens.document_resolver import open_document_modal, resolve_document_or_section

__all__ = [
    "DocumentViewerModal",
    "open_document_modal",
    "resolve_document_or_section",
]
