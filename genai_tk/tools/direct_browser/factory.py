"""Factory function for creating direct (host-local) browser tools.

Compatible with the ``factory:`` tool-spec pattern used in
``config/basic/agents/langchain.yaml``::

    tools:
      - factory: genai_tk.tools.direct_browser.factory.create_direct_browser_tools
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from loguru import logger

from genai_tk.tools.direct_browser.models import DirectBrowserConfig
from genai_tk.tools.direct_browser.session import DirectBrowserSession
from genai_tk.tools.direct_browser.tools import ALL_BROWSER_TOOLS


def _load_browser_config() -> DirectBrowserConfig:
    """Load browser config from the ``direct_browser`` key in sandbox.yaml."""
    try:
        from omegaconf import OmegaConf  # noqa: PLC0415

        from genai_tk.utils.config_mngr import global_config  # noqa: PLC0415

        raw = global_config().get("direct_browser", {})
        if not raw:
            return DirectBrowserConfig()
        if hasattr(raw, "_metadata"):
            raw = OmegaConf.to_container(raw, resolve=True)
        return DirectBrowserConfig.model_validate(raw)
    except Exception:
        return DirectBrowserConfig()


def create_direct_browser_tools() -> list[BaseTool]:
    """Create all direct browser LangChain tools sharing a single session.

    Returns:
        List of ``BaseTool`` instances for host-local browser automation.
    """
    config = _load_browser_config()
    session = DirectBrowserSession(config=config)
    tools: list[BaseTool] = [tool_cls(session=session) for tool_cls in ALL_BROWSER_TOOLS]
    logger.info(f"Created {len(tools)} direct browser tools (host-local Playwright)")
    return tools
