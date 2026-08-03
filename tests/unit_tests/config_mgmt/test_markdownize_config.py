"""Unit tests for the markdownize profile config helper."""

from __future__ import annotations

import pytest

from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.config_mgmt.markdownize_config import (
    BUILTIN_PROFILES,
    MarkdownizeProfile,
    get_markdownize_profile,
)


def test_default_resolves_to_medium() -> None:
    assert get_markdownize_profile("default") == BUILTIN_PROFILES["medium"]


@pytest.mark.parametrize("name", ["fast", "medium", "best"])
def test_builtin_profiles_available_without_config(name: str) -> None:
    assert get_markdownize_profile(name) == BUILTIN_PROFILES[name]


def test_fast_profile_is_all_local() -> None:
    fast = get_markdownize_profile("fast")
    assert fast.ppt_converter == "markitdown"
    assert fast.doc_converter == "markitdown"
    assert fast.excel_converter == "md_parser"
    assert fast.pdf_converter == "markitdown"


def test_best_profile_is_all_via_pdf_mistral() -> None:
    best = get_markdownize_profile("best")
    assert best.ppt_converter == "via_pdf"
    assert best.doc_converter == "via_pdf"
    assert best.excel_converter == "via_pdf"
    assert best.pdf_converter == "mistral"


def test_config_entry_overrides_builtin() -> None:
    global_config().set("markdownize_profiles.fast.pdf_converter", "mistral")
    assert get_markdownize_profile("fast").pdf_converter == "mistral"


def test_fingerprint_changes_with_any_field() -> None:
    a = MarkdownizeProfile(pdf_converter="markitdown")
    b = MarkdownizeProfile(pdf_converter="mistral")
    assert a.fingerprint() != b.fingerprint()


def test_unknown_name_raises() -> None:
    with pytest.raises(KeyError, match="Unknown markdownize profile"):
        get_markdownize_profile("does_not_exist")
