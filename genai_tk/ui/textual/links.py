"""Utilities for formatting valid, clickable Markdown links for Textual TUIs."""

from __future__ import annotations

import re
from urllib.parse import quote


def format_document_link(doc_name: str, label: str | None = None) -> str:
    """Format a clickable Markdown link for a document with valid percent-encoding.

    Args:
        doc_name: Document filename or content hash (e.g. 'Alko RFP Digital.pdf.md').
        label: Display text. If omitted, uses doc_name with brackets sanitized.

    Returns:
        Formatted markdown link string like `[Alko RFP Digital.pdf.md](doc://Alko%20RFP%20Digital.pdf.md)`
    """
    clean_doc = doc_name.strip()
    safe_label = (label or clean_doc).replace("[", "(").replace("]", ")")
    quoted_href = "doc://" + quote(clean_doc)
    return f"[{safe_label}]({quoted_href})"


def format_section_link(section_ref: str, label: str | None = None) -> str:
    """Format a clickable Markdown link for a document section with valid percent-encoding.

    Detects if section_ref contains an explicit section_id (e.g. `file_20d0d01e51d9c515::15`),
    and formats a clean human-readable label and percent-encoded target URL.

    Args:
        section_ref: Section reference string (e.g. 'On-Site service [file_20d0d01e51d9c515::15]'
            or 'file_20d0d01e51d9c515::15' or '3.1 Support services').
        label: Custom display text.

    Returns:
        Formatted markdown link string.
    """
    clean_sec = section_ref.strip()
    m = re.search(r"\[?(file_[a-f0-9]+::\d+)\]?", clean_sec)
    if m:
        sec_id = m.group(1)
        if not label:
            title_part = re.sub(r"\[?file_[a-f0-9]+::\d+\]?", "", clean_sec).strip(" :·-[]()")
            label = f"{title_part} ({sec_id})" if title_part else sec_id
        target = sec_id
    else:
        target = clean_sec

    safe_label = (label or clean_sec).replace("[", "(").replace("]", ")")
    quoted_href = "section://" + quote(target)
    return f"[{safe_label}]({quoted_href})"


def format_rfq_reference(ref: str) -> str:
    """Format an RFQ document and section reference line as valid clickable Markdown links.

    Handles composite references like:
    `Alko RFP Digital workplace and IT-Infra services 2024_pdf.md · On-Site service [file_20d0d01e51d9c515::15]`
    as well as single document or section references.

    Args:
        ref: Reference string.

    Returns:
        Formatted Markdown bullet line.
    """
    clean_ref = ref.strip()
    if " · " in clean_ref:
        doc_part, sec_part = clean_ref.split(" · ", 1)
        sec_clean = sec_part.lstrip("§ ").strip()
        return f"- 📄 {format_document_link(doc_part)} · {format_section_link(sec_clean)}"

    return f"- 📄 {format_document_link(clean_ref)}"
