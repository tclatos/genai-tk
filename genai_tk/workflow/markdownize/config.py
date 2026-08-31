"""Named ``markdownize`` conversion profiles."""

from __future__ import annotations

import yaml

from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.extra.markdownize.selector import (
    DocConverter,
    ExcelConverter,
    MarkdownizeProfile,
    PdfConverter,
    PptConverter,
)
from genai_tk.utils.singleton import once

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
    return {name: MarkdownizeProfile.model_validate(raw) for name, raw in data.get("markdownize_profiles", {}).items()}


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


__all__ = [
    "DEFAULT_PROFILE",
    "DocConverter",
    "ExcelConverter",
    "MarkdownizeProfile",
    "PdfConverter",
    "PptConverter",
    "get_markdownize_profile",
]
