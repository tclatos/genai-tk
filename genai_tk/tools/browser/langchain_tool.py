"""LangChain ``BaseTool`` wrapping an authenticated web scraper.

The tool loads a ``WebScraperConfig`` by name, runs ``ScraperSession`` in a
thread-safe async context, and returns the extracted page content for the LLM
to analyse.  Credentials are resolved from environment variables inside the
tool — they never appear in the tool's name, description, or return value.

Example:
    ```python
    from genai_tk.tools.browser.langchain_tool import AuthenticatedWebScraperTool

    tool = AuthenticatedWebScraperTool(
        config_name="enedis_production",
        target_name="production_daily",
    )
    result = await tool.arun("What was my solar production yesterday?")
    ```
"""

from __future__ import annotations

import asyncio

from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import Field, PrivateAttr, model_validator

from genai_tk.tools.browser.models import WebScraperConfig


class AuthenticatedWebScraperTool(BaseTool):
    """Fetches authenticated web pages and returns extracted text for the LLM.

    Credentials are loaded from environment variables defined in the YAML
    config — they are never passed to or exposed by the LLM.
    """

    config_name: str = Field(..., description="Scraper config name (key under web_scrapers: in YAML)")
    target_name: str = Field(..., description="Target name within the scraper config")
    config_path: str | None = Field(None, description="Path to YAML config file (None = auto-detect)")

    _scraped_config: WebScraperConfig | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _set_tool_identity(self) -> AuthenticatedWebScraperTool:
        self.name = f"scrape_{self.config_name}_{self.target_name}"
        self._build_description()
        return self

    def _build_description(self) -> None:
        """Build a rich tool description from config metadata (lazy)."""
        try:
            cfg = self._load_config()
            target = cfg.get_target(self.target_name)
            base = cfg.description or f"Scrape {self.config_name}"
            target_desc = target.description or f"Target: {self.target_name}"
            self.description = (
                f"{base}. {target_desc}. "
                f"Authentication is handled automatically — provide a natural language query "
                f"about the page content and the extracted text will be returned."
            )
        except Exception:
            self.description = (
                f"Scrape authenticated page '{self.config_name}/{self.target_name}'. "
                f"Provide a query about the content to retrieve."
            )

    def _load_config(self) -> WebScraperConfig:
        if self._scraped_config is None:
            from genai_tk.tools.browser.config_loader import load_web_scraper_config

            self._scraped_config = load_web_scraper_config(self.config_name, self.config_path)
        return self._scraped_config

    def _run(self, query: str) -> str:
        """Synchronous wrapper — runs the async scraper in an event loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already inside an async context (e.g. Jupyter/LangGraph)
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, self._ascrape(query))
                    return future.result()
            else:
                return loop.run_until_complete(self._ascrape(query))
        except Exception as exc:
            logger.error(f"[{self.config_name}/{self.target_name}] Scrape failed: {exc}")
            return f"Error scraping {self.config_name}/{self.target_name}: {exc}"

    async def _arun(self, query: str) -> str:
        """Async implementation — preferred path."""
        return await self._ascrape(query)

    async def _ascrape(self, query: str) -> str:
        from genai_tk.tools.browser.scraper_session import run_scraper

        logger.info(f"[{self.config_name}/{self.target_name}] Scraping for query: {query!r}")
        config = self._load_config()
        content = await run_scraper(config, target_name=self.target_name)
        logger.info(f"[{self.config_name}/{self.target_name}] Extracted {len(content)} characters")
        return content
