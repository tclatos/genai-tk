"""Unit tests for ingest routing models (IngestRule / IngestRouteTable)."""

from __future__ import annotations

import pytest

from genai_tk.workflow.routing.models import IngestRouteTable, IngestRoutingError, IngestRule, is_url


def test_file_extension_routing() -> None:
    rule = IngestRule(pathspec="**/*.{ppt,pptx,odp}", workflow="office_via_pdf")
    assert rule.matches("/data/docs/deck.pptx")
    assert rule.matches("deck.ppt")
    assert not rule.matches("/data/docs/report.pdf")


def test_url_routing() -> None:
    youtube = IngestRule(pathspec="https://www.youtube.com/**", workflow="youtube_transcript")
    web = IngestRule(pathspec="https://**", workflow="web_page")
    assert youtube.matches("https://www.youtube.com/watch?v=abc")
    assert not youtube.matches("https://example.com/page")
    assert web.matches("https://example.com/page")
    assert web.matches("http://example.com/page") is False  # scheme mismatch


def test_first_match_wins() -> None:
    table = IngestRouteTable(
        routes=[
            IngestRule(pathspec="**/special.pdf", workflow="special_flow"),
            IngestRule(pathspec="**/*.pdf", workflow="pdf_flow"),
        ],
        default="fallback_flow",
    )
    assert table.select("/data/special.pdf") == ("special_flow", {})
    assert table.select("/data/other.pdf") == ("pdf_flow", {})
    assert table.select("/data/readme.txt") == ("fallback_flow", {})


def test_route_params_forwarded() -> None:
    rule = IngestRule(pathspec="**/*.docx", workflow="office_via_pdf", params={"profile": "best"})
    assert rule.matches("/data/doc.docx")
    table = IngestRouteTable(routes=[rule])
    workflow_name, params = table.select("/data/doc.docx")
    assert workflow_name == "office_via_pdf"
    assert params == {"profile": "best"}


def test_with_yaml_key_maps_to_params() -> None:
    rule = IngestRule.model_validate({"pathspec": "**/*.pdf", "workflow": "pdf", "with": {"lang": "en"}})
    assert rule.params == {"lang": "en"}


def test_no_match_and_no_default_raises() -> None:
    table = IngestRouteTable(routes=[IngestRule(pathspec="**/*.pdf", workflow="pdf_flow")])
    with pytest.raises(IngestRoutingError):
        table.select("/data/readme.txt")


def test_fingerprint_changes_with_rules() -> None:
    a = IngestRouteTable(default="wf", routes=[IngestRule(pathspec="**/*.pdf", workflow="pdf")])
    b = IngestRouteTable(default="wf", routes=[IngestRule(pathspec="**/*.pdf", workflow="other")])
    c = IngestRouteTable(default="wf2", routes=[IngestRule(pathspec="**/*.pdf", workflow="pdf")])
    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint() != c.fingerprint()
    assert a.fingerprint() == a.fingerprint()


def test_is_url() -> None:
    assert is_url("https://x.org")
    assert is_url("http://x.org/a")
    assert not is_url("/data/file.pdf")
    assert not is_url("relative/file.pdf")
