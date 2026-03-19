"""Generic authenticated web scraper tools for LangChain agents.

Provides Playwright-based browser automation with support for multiple
authentication mechanisms (form login, OAuth redirect, OAuth popup,
storage state reuse), automatic cookie consent handling, and human-like
anti-bot behaviour.

Configuration is YAML-driven; credentials are resolved exclusively from
environment variables or files — they are never passed to the LLM.

Example:
    ```python
    from genai_tk.tools.browser.factory import create_web_scraper_tool

    tools = create_web_scraper_tool("enedis_production")
    # Use in any LangChain agent profile or directly
    result = await tools[0].arun("what was my solar production yesterday?")
    ```
"""

from genai_tk.tools.browser.factory import create_web_scraper_tool
from genai_tk.tools.browser.langchain_tool import AuthenticatedWebScraperTool
from genai_tk.tools.browser.models import WebScraperConfig

__all__ = ["AuthenticatedWebScraperTool", "WebScraperConfig", "create_web_scraper_tool"]
