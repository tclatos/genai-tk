"""Named ``markdownize`` conversion profiles.

A profile describes the *whole* conversion path for each source-document family,
so callers pick one name (``fast`` / ``medium`` / ``best``) instead of wiring
low-level converter flags. The ``via_pdf`` steps (LibreOffice → PDF → OCR) are an
implementation detail hidden behind the profile.

Per-family converter choices:

- ``ppt_converter`` — PowerPoint/Impress (``.ppt``/``.pptx``/``.odp``):
  ``via_pdf`` or ``markitdown``.
- ``doc_converter`` — Word/Writer (``.doc``/``.docx``/``.odt``/``.rtf``):
  ``via_pdf`` or ``markitdown``.
- ``excel_converter`` — spreadsheets (``.xls``/``.xlsx``/``.ods``):
  ``via_pdf``, ``markitdown``, or ``md_parser``.
- ``pdf_converter`` — how PDFs (native *and* the ones produced by ``via_pdf``)
  become Markdown: ``mistral``, ``markitdown``, or ``edgeparse``.

Built-in profiles (always available, no configuration required):

- ``fast`` — everything local: ``markitdown`` + ``md_parser``.
- ``medium`` — Office via LibreOffice → Mistral OCR, spreadsheets via ``md_parser``.
- ``best`` — everything via LibreOffice → Mistral OCR (highest fidelity, slowest).
- ``default`` — alias for ``medium``.

Projects may add or override profiles under a ``markdownize_profiles`` config key;
those entries take precedence over the built-ins of the same name.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from genai_tk.config_mgmt.config_mngr import global_config

PptConverter = Literal["via_pdf", "markitdown"]
DocConverter = Literal["via_pdf", "markitdown"]
ExcelConverter = Literal["via_pdf", "markitdown", "md_parser"]
PdfConverter = Literal["mistral", "markitdown", "edgeparse"]


class MarkdownizeProfile(BaseModel):
    """Whole-path converter choices per source-document family for markdownize_flow."""

    ppt_converter: PptConverter = "markitdown"
    doc_converter: DocConverter = "markitdown"
    excel_converter: ExcelConverter = "md_parser"
    pdf_converter: PdfConverter = "markitdown"

    model_config = {"extra": "forbid"}

    def fingerprint(self) -> str:
        """Stable cache key: any converter change invalidates cached conversions."""
        return f"{self.ppt_converter}:{self.doc_converter}:{self.excel_converter}:{self.pdf_converter}"


BUILTIN_PROFILES: dict[str, MarkdownizeProfile] = {
    "fast": MarkdownizeProfile(
        ppt_converter="markitdown",
        doc_converter="markitdown",
        excel_converter="md_parser",
        pdf_converter="markitdown",
    ),
    "medium": MarkdownizeProfile(
        ppt_converter="via_pdf",
        doc_converter="via_pdf",
        excel_converter="md_parser",
        pdf_converter="mistral",
    ),
    "best": MarkdownizeProfile(
        ppt_converter="via_pdf",
        doc_converter="via_pdf",
        excel_converter="via_pdf",
        pdf_converter="mistral",
    ),
}

DEFAULT_PROFILE = "medium"


def get_markdownize_profile(name: str = "default") -> MarkdownizeProfile:
    """Resolve a profile by name.

    Resolution order: a ``markdownize_profiles`` config entry of that name wins;
    otherwise the built-in ``fast``/``medium``/``best`` profile is used. ``default``
    resolves to ``medium`` when not explicitly configured.

    Args:
        name: Profile name (``fast``, ``medium``, ``best``, ``default``, or a
            custom key configured under ``markdownize_profiles``).

    Returns:
        The resolved profile.

    Example:
        ```python
        profile = get_markdownize_profile("medium")
        markdownize_flow(base_dir=..., output_dir=..., profile=profile)
        ```
    """
    configured = global_config().section_dict("markdownize_profiles", MarkdownizeProfile, inject_name=False)
    if name in configured:
        return configured[name]
    if name in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[name]
    if name == "default":
        return BUILTIN_PROFILES[DEFAULT_PROFILE]
    available = sorted(set(configured) | set(BUILTIN_PROFILES) | {"default"})
    raise KeyError(f"Unknown markdownize profile '{name}'. Available: {available}")
