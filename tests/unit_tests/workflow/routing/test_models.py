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
    assert is_url("git@github.com:org/repo.git")
    assert is_url("git://example.com/repo")
    assert is_url("ssh://git@example.com/repo")
    assert is_url("repo.git")
    assert not is_url("/data/file.pdf")
    assert not is_url("relative/file.pdf")


def test_git_url_routing() -> None:
    github_rule = IngestRule(pathspec="https://github.com/**", workflow="git_repo_flow")
    gitlab_rule = IngestRule(pathspec="https://gitlab.com/**", workflow="git_repo_flow")
    dotgit_rule = IngestRule(pathspec="**/*.git", workflow="git_repo_flow")
    git_ssh_rule = IngestRule(pathspec="git@**", workflow="git_repo_flow")
    web_rule = IngestRule(pathspec="https://**", workflow="web_page_flow")

    table = IngestRouteTable(
        routes=[github_rule, gitlab_rule, dotgit_rule, git_ssh_rule, web_rule],
        default="files_flow",
    )

    assert table.select("https://github.com/tclatos/prefect-yaml") == ("git_repo_flow", {})
    assert table.select("https://gitlab.com/group/repo") == ("git_repo_flow", {})
    assert table.select("https://example.com/repo.git") == ("git_repo_flow", {})
    assert table.select("git@github.com:tclatos/prefect-yaml.git") == ("git_repo_flow", {})
    assert table.select("https://example.com/blog/article") == ("web_page_flow", {})
    assert table.select("/data/docs/file.pdf") == ("files_flow", {})
