"""Unit tests for the markdownize profile config helper."""

from __future__ import annotations

import pytest

from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.config_mgmt.markdownize_config import MarkdownizeProfile, get_markdownize_profile


def test_get_markdownize_profile_default_falls_back_when_unconfigured() -> None:
    profile = get_markdownize_profile("default")
    assert profile == MarkdownizeProfile(pdf_converter="markitdown", excel_converter="md_parser")


def test_get_markdownize_profile_resolves_configured_entry() -> None:
    global_config().set("markdownize_profiles.mistral.pdf_converter", "mistral")
    global_config().set("markdownize_profiles.mistral.excel_converter", "md_parser")

    profile = get_markdownize_profile("mistral")

    assert profile.pdf_converter == "mistral"
    assert profile.excel_converter == "md_parser"


def test_get_markdownize_profile_unknown_name_raises() -> None:
    with pytest.raises(KeyError, match="Unknown markdownize profile"):
        get_markdownize_profile("does_not_exist")
