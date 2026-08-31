"""Unit tests for the markdownize profile config helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.extra.markdownize.selector import ConverterRule
from genai_tk.workflow.markdownize.config import (
    MarkdownizeProfile,
    _builtin_profiles,
    get_markdownize_profile,
)


def test_default_resolves_to_medium() -> None:
    assert get_markdownize_profile("default") == _builtin_profiles()["medium"]


@pytest.mark.parametrize("name", ["fast", "medium", "best", "lighton", "anydoc", "llm"])
def test_builtin_profiles_available_without_config(name: str) -> None:
    assert get_markdownize_profile(name) == _builtin_profiles()[name]


def test_fast_profile_is_all_local() -> None:
    fast = get_markdownize_profile("fast")
    assert fast.select_route(Path("sheet.xlsx")) == "messy_xls"
    assert fast.select_route(Path("doc.pdf")) == "markitdown"
    assert fast.select_route(Path("report.docx")) == "markitdown"


def test_best_profile_is_all_via_pdf_mistral() -> None:
    best = get_markdownize_profile("best")
    assert best.select_route(Path("deck.pptx")) == "via_pdf"
    assert best.select_route(Path("report.docx")) == "via_pdf"
    assert best.select_route(Path("sheet.xlsx")) == "via_pdf"
    assert best.select_route(Path("scan.pdf")) == "mistral_ocr"


def test_config_entry_overrides_builtin() -> None:
    try:
        global_config().set(
            "markdownize_profiles.fast.rules",
            [{"pathspec": "**/*", "converter": "mistral_ocr"}],
        )
        assert get_markdownize_profile("fast").select_route(Path("file.txt")) == "mistral_ocr"
    finally:
        global_config_reload = getattr(global_config(), "singleton", None)
        if global_config_reload:
            global_config_reload.invalidate()


def test_fingerprint_changes_with_any_rule() -> None:
    a = MarkdownizeProfile(rules=[ConverterRule(pathspec="**/*.pdf", converter="markitdown")])
    b = MarkdownizeProfile(rules=[ConverterRule(pathspec="**/*.pdf", converter="mistral_ocr")])
    assert a.fingerprint() != b.fingerprint()


def test_unknown_name_raises() -> None:
    with pytest.raises(KeyError, match="Unknown markdownize profile"):
        get_markdownize_profile("does_not_exist")
