"""Factory function for creating authenticated web scraper tools.

Compatible with the ``factory:`` tool-spec pattern used in
``config/basic/agents/langchain.yaml``::

    tools:
      - factory: genai_tk.tools.browser.factory:create_web_scraper_tool
        config_name: enedis_production
        # target_name is optional; defaults to all targets when omitted
        # config_path is optional; defaults to auto-detect

Example:
    ```python
    from genai_tk.tools.browser.factory import create_web_scraper_tool

    tools = create_web_scraper_tool("enedis_production")
    # or with an explicit target:
    tools = create_web_scraper_tool("enedis_production", target_name="production_daily")
    ```
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from loguru import logger

from genai_tk.tools.browser.langchain_tool import AuthenticatedWebScraperTool


def create_web_scraper_tool(
    config_name: str,
    target_name: str | None = None,
    config_path: str | None = None,
) -> list[BaseTool]:
    """Create one ``AuthenticatedWebScraperTool`` per target in the config.

    When ``target_name`` is provided, returns a single-element list containing
    only that target's tool.  When ``None``, returns one tool per target
    declared in the scraper config.

    Args:
        config_name: Scraper config name (key under ``web_scrapers:`` in YAML).
        target_name: If set, only create a tool for this target.
        config_path: Path to the YAML config.  ``None`` = auto-detect from
            ``config/basic/web_scrapers/<config_name>.yaml``.

    Returns:
        List of ``AuthenticatedWebScraperTool`` instances.
    """
    if target_name is not None:
        return [
            AuthenticatedWebScraperTool(
                config_name=config_name,
                target_name=target_name,
                config_path=config_path,
            )
        ]

    # Load config to enumerate all targets
    try:
        from genai_tk.tools.browser.config_loader import load_web_scraper_config

        cfg = load_web_scraper_config(config_name, config_path)
        tools: list[BaseTool] = [
            AuthenticatedWebScraperTool(
                config_name=config_name,
                target_name=t.name,
                config_path=config_path,
            )
            for t in cfg.targets
        ]
        logger.info(f"Created {len(tools)} scraper tool(s) for '{config_name}'")
        return tools
    except Exception as exc:
        logger.warning(f"Cannot enumerate targets for '{config_name}': {exc} — returning empty list")
        return []
