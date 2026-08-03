"""Named ``markdownize`` conversion profiles.

Bundles the ``pdf_converter`` / ``excel_converter`` choice for
:func:`genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow` under
a single name, so callers (CLI commands, webapp pages, workflow YAML) pass one
profile name instead of duplicating both converter fields everywhere.

YAML configuration schema::

    markdownize_profiles:
      default:
        pdf_converter: markitdown
        excel_converter: md_parser
      mistral:
        pdf_converter: mistral
        excel_converter: md_parser
"""

from __future__ import annotations

from pydantic import BaseModel

from genai_tk.config_mgmt.config_mngr import global_config


class MarkdownizeProfile(BaseModel):
    """A named pdf_converter/excel_converter combination for markdownize_flow."""

    pdf_converter: str = "markitdown"
    excel_converter: str = "md_parser"


def get_markdownize_profile(name: str = "default") -> MarkdownizeProfile:
    """Resolve a named profile from the ``markdownize_profiles`` config section.

    Falls back to :class:`MarkdownizeProfile` defaults when *name* is
    ``"default"`` and not explicitly configured.

    Args:
        name: Profile key in the ``markdownize_profiles`` config section.

    Returns:
        The resolved profile.

    Raises:
        KeyError: If *name* is not configured and is not ``"default"``.
    """
    profiles = global_config().section_dict("markdownize_profiles", MarkdownizeProfile, inject_name=False)
    if name in profiles:
        return profiles[name]
    if name == "default":
        return MarkdownizeProfile()
    raise KeyError(f"Unknown markdownize profile '{name}'. Available: {list(profiles)}")
