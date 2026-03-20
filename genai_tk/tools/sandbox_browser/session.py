"""Persistent Playwright session connected to an AIO sandbox browser via CDP.

Manages the browser context and page lifecycle across multiple tool invocations
within a single agent run.  Created lazily on first tool use, closed when the
agent finishes.

Example:
    ```python
    session = SandboxBrowserSession(sandbox_url="http://localhost:8080")
    await session.connect()
    page = session.page
    await page.goto("https://example.com")
    await session.close()
    ```
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from genai_tk.tools.sandbox_browser.models import SandboxBrowserConfig

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page

# Anti-bot JS injected via add_init_script before any page JS runs.
_ANTI_BOT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3]});
Object.defineProperty(navigator, 'languages', {get: () => ['fr-FR', 'fr', 'en-US', 'en']});
window.chrome = {runtime: {}};
"""


class SandboxBrowserSession:
    """Manages a Playwright connection to an AIO sandbox's Chromium via CDP."""

    def __init__(
        self,
        sandbox_url: str = "http://localhost:8080",
        config: SandboxBrowserConfig | None = None,
    ) -> None:
        self._sandbox_url = sandbox_url.rstrip("/")
        self.config = config or SandboxBrowserConfig()
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._playwright: object | None = None
        self._connected = False

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser session not connected — call connect() first")
        return self._page

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def vnc_url(self) -> str:
        """Return the noVNC URL for visual debugging of the sandbox browser."""
        return f"{self._sandbox_url}/vnc/index.html?autoconnect=true"

    async def connect(self) -> None:
        """Connect to the sandbox browser via CDP and create a page."""
        if self._connected:
            return

        try:
            from agent_sandbox import Sandbox  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("agent-sandbox is required: uv add agent-sandbox") from exc
        try:
            from playwright.async_api import async_playwright  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("playwright is required: uv add playwright") from exc

        # Get CDP URL from sandbox API
        client = Sandbox(base_url=self._sandbox_url)
        browser_info = client.browser.get_info().data
        cdp_url = browser_info.cdp_url
        logger.info(f"Connecting to sandbox browser via CDP: {cdp_url}")

        # Connect Playwright over CDP
        pw = await async_playwright().start()
        self._playwright = pw
        self._browser = await pw.chromium.connect_over_cdp(cdp_url)

        # Create context with anti-bot mitigations and locale
        width = self.config.viewport_width + random.randint(-30, 30)
        height = self.config.viewport_height + random.randint(-30, 30)
        self._context = await self._browser.new_context(
            viewport={"width": width, "height": height},
            locale=self.config.locale,
            ignore_https_errors=self.config.ignore_https_errors,
        )

        if self.config.anti_bot_js:
            await self._context.add_init_script(_ANTI_BOT_SCRIPT)

        self._page = await self._context.new_page()
        self._connected = True
        logger.info("Sandbox browser session connected")

    async def close(self) -> None:
        """Close the browser session and release resources."""
        if not self._connected:
            return
        try:
            if self._context:
                await self._context.close()
        except Exception as exc:
            logger.warning(f"Error closing browser context: {exc}")
        try:
            if self._browser:
                await self._browser.close()
        except Exception as exc:
            logger.warning(f"Error closing browser: {exc}")
        try:
            if self._playwright:
                await self._playwright.stop()  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning(f"Error stopping playwright: {exc}")
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._connected = False
        logger.info("Sandbox browser session closed")

    async def save_cookies(self, name: str) -> str:
        """Save the current browser context storage state to a JSON file.

        Args:
            name: Cookie file identifier (used as filename stem).

        Returns:
            Absolute path to the saved file.
        """
        if not self._context:
            raise RuntimeError("No active browser context")
        cookies_dir = Path(self.config.cookies_dir)
        cookies_dir.mkdir(parents=True, exist_ok=True)
        file_path = cookies_dir / f"{name}_session.json"
        state = await self._context.storage_state()
        file_path.write_text(json.dumps(state, indent=2))
        logger.info(f"Cookies saved to {file_path}")
        return str(file_path)

    async def load_cookies(self, name: str) -> bool:
        """Load a previously saved storage state into a new browser context.

        Replaces the current context and page with a new one using the saved state.

        Args:
            name: Cookie file identifier (must match a previous save).

        Returns:
            True if cookies were loaded successfully.
        """
        file_path = Path(self.config.cookies_dir) / f"{name}_session.json"
        if not file_path.is_file():
            logger.warning(f"Cookie file not found: {file_path}")
            return False

        if not self._browser:
            raise RuntimeError("Browser not connected")

        state = json.loads(file_path.read_text())

        # Close current context and create a new one with saved state
        if self._context:
            await self._context.close()

        width = self.config.viewport_width + random.randint(-30, 30)
        height = self.config.viewport_height + random.randint(-30, 30)
        self._context = await self._browser.new_context(
            viewport={"width": width, "height": height},
            locale=self.config.locale,
            ignore_https_errors=self.config.ignore_https_errors,
            storage_state=state,
        )
        if self.config.anti_bot_js:
            await self._context.add_init_script(_ANTI_BOT_SCRIPT)

        self._page = await self._context.new_page()
        logger.info(f"Cookies loaded from {file_path}")
        return True

    async def __aenter__(self) -> SandboxBrowserSession:
        await self.connect()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()
