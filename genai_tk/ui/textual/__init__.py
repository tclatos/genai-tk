"""Textual Terminal User Interface (TUI) layer."""

from __future__ import annotations

from genai_tk.ui.textual.base import BaseTuiApp
from genai_tk.ui.textual.constants import QUIT_COMMANDS, STANDARD_QUIT_BINDINGS
from genai_tk.ui.textual.links import format_document_link, format_rfq_reference, format_section_link
from genai_tk.ui.textual.screens.document_modal import DocumentViewerModal
from genai_tk.ui.textual.screens.document_resolver import open_document_modal, resolve_document_or_section

__all__ = [
    "BaseTuiApp",
    "DocumentViewerModal",
    "QUIT_COMMANDS",
    "STANDARD_QUIT_BINDINGS",
    "format_document_link",
    "format_rfq_reference",
    "format_section_link",
    "open_document_modal",
    "resolve_document_or_section",
]
