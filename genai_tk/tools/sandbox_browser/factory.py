"""Factory function for creating sandbox browser tools.

Compatible with the ``factory:`` tool-spec pattern used in
``config/agents/langchain.yaml``::

    tools:
      - factory: genai_tk.tools.sandbox_browser.factory.create_sandbox_browser_tools
        sandbox_url: http://localhost:8080  # optional, defaults to sandbox config

Example:
    ```python
    from genai_tk.tools.sandbox_browser.factory import create_sandbox_browser_tools

    tools = create_sandbox_browser_tools()
    ```
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from loguru import logger

from genai_tk.tools.sandbox_browser.models import SandboxBrowserConfig
from genai_tk.tools.sandbox_browser.session import SandboxBrowserSession
from genai_tk.tools.sandbox_browser.tools import ALL_BROWSER_TOOLS


def _load_browser_config() -> SandboxBrowserConfig:
    """Load browser config from the sandbox.yaml file."""
    try:
        from omegaconf import OmegaConf  # noqa: PLC0415

        from genai_tk.utils.config_mngr import global_config  # noqa: PLC0415

        raw = global_config().get("sandbox_browser", {})
        if not raw:
            return SandboxBrowserConfig()
        if hasattr(raw, "_metadata"):
            raw = OmegaConf.to_container(raw, resolve=True)
        return SandboxBrowserConfig.model_validate(raw)
    except Exception:
        return SandboxBrowserConfig()


def create_sandbox_browser_tools(
    sandbox_url: str | None = None,
) -> list[BaseTool]:
    """Create all sandbox browser LangChain tools sharing a single session.

    Args:
        sandbox_url: AIO sandbox base URL.  Defaults to the ``opensandbox_server_url``
            from ``config/basic/sandbox.yaml``.

    Returns:
        List of ``BaseTool`` instances for browser automation.
    """
    config = _load_browser_config()

    if sandbox_url is None:
        try:
            from genai_tk.agents.sandbox.config import get_docker_aio_settings  # noqa: PLC0415

            sandbox_url = get_docker_aio_settings().opensandbox_server_url
        except Exception:
            sandbox_url = "http://localhost:8080"

    session = SandboxBrowserSession(sandbox_url=sandbox_url, config=config)
    tools: list[BaseTool] = [tool_cls(session=session) for tool_cls in ALL_BROWSER_TOOLS]

    logger.info(f"Created {len(tools)} sandbox browser tools (sandbox: {sandbox_url})")
    return tools
