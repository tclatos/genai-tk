"""Unit tests for web page fetchers (both implementations mocked — no network)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from genai_tk.web.fetchers import BsFetcher, TavilyFetcher, get_web_page_fetcher


def test_get_web_page_fetcher_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown web page fetcher"):
        get_web_page_fetcher("does_not_exist")


def test_get_web_page_fetcher_aliases() -> None:
    assert isinstance(get_web_page_fetcher("bs"), BsFetcher)
    assert isinstance(get_web_page_fetcher("bs4"), BsFetcher)
    assert isinstance(get_web_page_fetcher("beautifulsoup"), BsFetcher)
    assert isinstance(get_web_page_fetcher("tavily"), TavilyFetcher)


def test_tavily_fetcher_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    fetcher = TavilyFetcher()
    with patch("tavily.TavilyClient") as _:
        with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
            fetcher.fetch("https://example.com")


def test_tavily_fetcher_extracts_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    mock_client = MagicMock()
    mock_client.extract.return_value = {
        "results": [{"url": "https://example.com", "raw_content": "# Title\n\nBody text"}],
        "failed_results": [],
    }

    with patch("tavily.TavilyClient", return_value=mock_client) as mock_cls:
        content = TavilyFetcher().fetch("https://example.com")

    assert content == "# Title\n\nBody text"
    mock_cls.assert_called_once_with(api_key="test-key")
    mock_client.extract.assert_called_once_with(["https://example.com"], extract_depth="basic")


def test_tavily_fetcher_empty_result_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    mock_client = MagicMock()
    mock_client.extract.return_value = {"results": [], "failed_results": [{"url": "https://example.com"}]}

    with patch("tavily.TavilyClient", return_value=mock_client):
        with pytest.raises(RuntimeError, match="no content"):
            TavilyFetcher().fetch("https://example.com")


def test_bs_fetcher_converts_html_to_markdown() -> None:
    html = """
    <html><body>
        <script>alert('x')</script>
        <main><h1>Hello</h1><p>World</p></main>
        <footer>copyright</footer>
    </body></html>
    """

    class _FakeResponse:
        text = html

        def raise_for_status(self) -> None: ...

    with patch("httpx.get", return_value=_FakeResponse()) as mock_get:
        markdown = BsFetcher().fetch("https://example.com/page")

    assert "# Hello" in markdown
    assert "World" in markdown
    assert "alert" not in markdown
    assert "copyright" not in markdown
    mock_get.assert_called_once()


def test_bs_fetcher_http_error_propagates() -> None:
    import httpx

    with patch("httpx.get", side_effect=httpx.HTTPStatusError("404", request=MagicMock(), response=MagicMock())):
        with pytest.raises(httpx.HTTPStatusError):
            BsFetcher().fetch("https://example.com/missing")
