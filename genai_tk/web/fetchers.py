"""Simple web-page-to-Markdown fetchers.

One abstraction, two implementations:

- ``tavily`` — the Tavily Extract API (needs ``TAVILY_API_KEY``; robust against
  JS-heavy pages and returns clean article content).
- ``bs`` — a plain HTTP GET via httpx followed by BeautifulSoup DOM cleanup and
  markdownify HTML→Markdown conversion (no API key, fully local).

Select an implementation by name with :func:`get_web_page_fetcher`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from pydantic import BaseModel


class WebPageFetcher(BaseModel, ABC):
    """Fetch a web page and return its content as Markdown."""

    @abstractmethod
    def fetch(self, url: str) -> str:
        """Return the content of *url* as Markdown."""


class TavilyFetcher(WebPageFetcher):
    """Fetch pages through the Tavily Extract API."""

    api_key: str | None = None
    extract_depth: str = "basic"

    def model_post_init(self, context: object) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("TAVILY_API_KEY")

    def fetch(self, url: str) -> str:
        from tavily import TavilyClient

        if not self.api_key:
            raise RuntimeError("TavilyFetcher requires an API key: set TAVILY_API_KEY or pass api_key=")
        client = TavilyClient(api_key=self.api_key)
        response = client.extract([url], extract_depth=self.extract_depth)
        results = response.get("results", [])
        if not results:
            failed = response.get("failed_results") or []
            raise RuntimeError(f"Tavily extract returned no content for {url} (failed: {failed})")
        content = results[0].get("raw_content")
        if not content or not content.strip():
            raise RuntimeError(f"Tavily extract returned empty content for {url}")
        return str(content)


class BsFetcher(WebPageFetcher):
    """Fetch pages with httpx, clean the DOM with BeautifulSoup and convert to Markdown."""

    timeout: float = 30.0
    user_agent: str = "Mozilla/5.0 (compatible; genai-tk web fetcher)"

    def fetch(self, url: str) -> str:
        import httpx
        from bs4 import BeautifulSoup
        from markdownify import markdownify

        response = httpx.get(url, follow_redirects=True, timeout=self.timeout, headers={"User-Agent": self.user_agent})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "iframe"]):
            tag.decompose()
        body = soup.body or soup
        markdown = markdownify(str(body), heading_style="AT").strip()
        if not markdown:
            raise RuntimeError(f"No content extracted from {url}")
        return markdown


_FETCHERS: dict[str, type[WebPageFetcher]] = {
    "tavily": TavilyFetcher,
    "bs": BsFetcher,
    "bs4": BsFetcher,
    "beautifulsoup": BsFetcher,
}


def get_web_page_fetcher(name: str = "bs", **kwargs: object) -> WebPageFetcher:
    """Return the named fetcher implementation."""
    try:
        return _FETCHERS[name](**kwargs)  # type: ignore[arg-type]
    except KeyError:
        available = ", ".join(sorted({"tavily", "bs"}))
        raise KeyError(f"Unknown web page fetcher '{name}'. Available: {available}") from None
