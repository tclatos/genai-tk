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

# Minimal anti-bot JS: only clean up CDP artifacts that leak automation.
# The sandbox runs *headful* Chromium with --disable-blink-features=AutomationControlled,
# so navigator.webdriver is already false, plugins/languages/UA are all genuine.
# Heavy-handed overrides (faking plugins, UA, etc.) create detectable INCONSISTENCIES
# that sophisticated bot detectors flag.
_ANTI_BOT_SCRIPT = """
// Remove CDP (Chrome DevTools Protocol) artifacts left by automation drivers
for (const key of Object.keys(window)) {
  if (key.startsWith('cdc_')) delete window[key];
}
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
        self._owns_context = True  # False when reusing the browser's default context
        self._event_log: list[str] = []  # in-memory log surviving browser disconnect

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

        # Reuse the browser's DEFAULT context so that all launch flags
        # (--user-agent, --disable-blink-features=AutomationControlled, etc.)
        # are inherited.  Creating a new_context() over CDP does NOT inherit
        # Chromium launch flags — that causes bot-detection redirects.
        contexts = self._browser.contexts
        if contexts:
            self._context = contexts[0]
            self._owns_context = False
            pages = self._context.pages
            if pages:
                self._page = pages[0]
            else:
                self._page = await self._context.new_page()
        else:
            # Fallback: create a fresh context (shouldn't happen with sandbox)
            width = self.config.viewport_width + random.randint(-30, 30)
            height = self.config.viewport_height + random.randint(-30, 30)
            self._context = await self._browser.new_context(
                viewport={"width": width, "height": height},
                locale=self.config.locale,
                ignore_https_errors=self.config.ignore_https_errors,
            )
            self._owns_context = True
            self._page = await self._context.new_page()

        # Set viewport on whichever page we ended up with
        width = self.config.viewport_width + random.randint(-30, 30)
        height = self.config.viewport_height + random.randint(-30, 30)
        await self._page.set_viewport_size({"width": width, "height": height})

        if self.config.anti_bot_js:
            await self._context.add_init_script(_ANTI_BOT_SCRIPT)

        self._connected = True

        # Attach browser-level event listeners for debugging
        self._attach_debug_listeners()
        logger.info("Sandbox browser session connected")

    async def close(self) -> None:
        """Close the browser session and release resources."""
        if not self._connected:
            return
        # Detach event listeners before closing to avoid callbacks on dead objects
        self._detach_debug_listeners()
        try:
            # Only close the context if we created it; the browser's default
            # context must not be closed — that would tear down Chromium.
            if self._context and self._owns_context:
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

    # ------------------------------------------------------------------
    # Debug event listeners
    # ------------------------------------------------------------------

    def _attach_debug_listeners(self) -> None:
        """Wire Playwright event listeners for browser-level console and error logging."""
        if not self.config.log_browser_console:
            return
        if self._page is not None:
            self._page.on("console", self._on_console)
            self._page.on("pageerror", self._on_page_error)
            self._page.on("crash", self._on_page_crash)
            self._page.on("close", self._on_page_close)
            self._page.on("request", self._on_request)
            self._page.on("response", self._on_response)
        if self._browser is not None:
            self._browser.on("disconnected", self._on_browser_disconnected)

    def _detach_debug_listeners(self) -> None:
        """Remove previously attached event listeners."""
        _page_handlers = {
            "console": self._on_console,
            "pageerror": self._on_page_error,
            "crash": self._on_page_crash,
            "close": self._on_page_close,
            "request": self._on_request,
            "response": self._on_response,
        }
        try:
            if self._page is not None:
                for evt, handler in _page_handlers.items():
                    self._page.remove_listener(evt, handler)
        except Exception:
            pass
        try:
            if self._browser is not None:
                self._browser.remove_listener("disconnected", self._on_browser_disconnected)
        except Exception:
            pass

    def _log_event(self, level: str, message: str) -> None:
        """Append a timestamped entry to the in-memory event log and emit via loguru."""
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._event_log.append(f"[{ts}] {level}: {message}")
        # Keep last 200 entries to bound memory
        if len(self._event_log) > 200:
            self._event_log = self._event_log[-200:]
        getattr(logger, level.lower(), logger.debug)(message)

    def get_event_log(self, last_n: int = 50) -> str:
        """Return the last *last_n* browser events as a newline-separated string.

        Survives browser disconnects — useful for post-mortem analysis.
        """
        entries = self._event_log[-last_n:]
        if not entries:
            return "(no browser events recorded)"
        return "\n".join(entries)

    def _on_console(self, msg: object) -> None:
        """Log browser console messages (console.log, console.warn, console.error)."""
        msg_type = getattr(msg, "type", "log")
        text = str(getattr(msg, "text", msg))
        if msg_type in ("error", "warning"):
            self._log_event("WARNING", f"Browser console [{msg_type}]: {text[:500]}")
        else:
            self._log_event("DEBUG", f"Browser console [{msg_type}]: {text[:300]}")

    def _on_page_error(self, error: object) -> None:
        """Log uncaught JavaScript exceptions."""
        self._log_event("ERROR", f"Browser page error: {error}")

    def _on_page_crash(self, _: object) -> None:
        """Log page (renderer) crash events."""
        self._log_event("ERROR", "Browser page CRASHED — renderer process died")

    def _on_page_close(self, _: object) -> None:
        """Log when a page is closed (possibly by navigation or site JS)."""
        self._log_event("WARNING", "Browser page closed")

    def _on_browser_disconnected(self, _: object) -> None:
        """Log when the CDP connection to the browser drops."""
        self._log_event("ERROR", "Browser DISCONNECTED — CDP connection lost (container may have died)")

    def _on_request(self, request: object) -> None:
        """Log main-frame navigations."""
        url = getattr(request, "url", "")
        frame = getattr(request, "frame", None)
        is_navigation = getattr(request, "is_navigation_request", lambda: False)()
        if is_navigation and frame and getattr(frame, "parent_frame", None) is None:
            self._log_event("INFO", f"Navigation request → {url}")

    def _on_response(self, response: object) -> None:
        """Log non-2xx responses on main-frame navigations."""
        status = getattr(response, "status", 0)
        url = getattr(response, "url", "")
        request = getattr(response, "request", None)
        is_navigation = request and getattr(request, "is_navigation_request", lambda: False)()
        frame = request and getattr(request, "frame", None)
        is_main = frame and getattr(frame, "parent_frame", None) is None
        if is_navigation and is_main and status >= 400:
            self._log_event("WARNING", f"HTTP {status} on navigation → {url}")

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
