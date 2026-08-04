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
  ``via_pdf``, ``markitdown``, or ``messy_xls_parser`` (deterministic, handles
  merged cells, grouped headers, and multi-table sheets — see
  :mod:`genai_tk.workflow.markdownize.excel`).
- ``pdf_converter`` — how PDFs (native *and* the ones produced by ``via_pdf``)
  become Markdown: ``mistral``, ``markitdown``, or ``edgeparse``.

Built-in profiles (always available, no configuration required) are shipped in
``genai_tk/default_config/markdownize.yaml``:

- ``fast`` — everything local: ``markitdown`` + ``messy_xls_parser``.
- ``medium`` — Office via LibreOffice → Mistral OCR, spreadsheets via ``messy_xls_parser``.
- ``best`` — everything via LibreOffice → Mistral OCR (highest fidelity, slowest).
- ``default`` — alias for ``medium``.

Projects may add or override profiles under a ``markdownize_profiles`` config key;
those entries take precedence over the built-ins of the same name.
"""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel

from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.utils.singleton import once

PptConverter = Literal["via_pdf", "markitdown"]
DocConverter = Literal["via_pdf", "markitdown"]
ExcelConverter = Literal["via_pdf", "markitdown", "messy_xls_parser"]
PdfConverter = Literal["mistral", "markitdown", "edgeparse"]


class MarkdownizeProfile(BaseModel):
    """Whole-path converter choices per source-document family for markdownize_flow."""

    ppt_converter: PptConverter = "markitdown"
    doc_converter: DocConverter = "markitdown"
    excel_converter: ExcelConverter = "messy_xls_parser"
    pdf_converter: PdfConverter = "markitdown"

    model_config = {"extra": "forbid"}

    def fingerprint(self) -> str:
        """Stable cache key: any converter change invalidates cached conversions."""
        return f"{self.ppt_converter}:{self.doc_converter}:{self.excel_converter}:{self.pdf_converter}"


DEFAULT_PROFILE = "medium"


@once
def _builtin_profiles() -> dict[str, MarkdownizeProfile]:
    """Load the packaged fast/medium/best profiles from default_config/markdownize.yaml."""
    from importlib.resources import files as _pkg_files

    try:
        src = _pkg_files("genai_tk") / "default_config" / "markdownize.yaml"
        yaml_text = src.read_text(encoding="utf-8")
    except Exception:
        # Fallback for editable installs where the symlink may not resolve via importlib
        from pathlib import Path

        config_path = Path(__file__).parent.parent.parent / "default_config" / "markdownize.yaml"
        yaml_text = config_path.read_text(encoding="utf-8")

    data = yaml.safe_load(yaml_text)
    return {name: MarkdownizeProfile.model_validate(raw) for name, raw in data["markdownize_profiles"].items()}


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
        markdownize_flow(sources=..., md_output_dir=..., profile=profile)
        ```
    """
    builtin = _builtin_profiles()
    configured = global_config().section_dict("markdownize_profiles", MarkdownizeProfile, inject_name=False)
    if name in configured:
        return configured[name]
    if name in builtin:
        return builtin[name]
    if name == "default":
        return builtin[DEFAULT_PROFILE]
    available = sorted(set(configured) | set(builtin) | {"default"})
    raise KeyError(f"Unknown markdownize profile '{name}'. Available: {available}")
