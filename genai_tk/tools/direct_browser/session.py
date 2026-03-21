"""Host-local Playwright browser session for direct browser control.

Launches a fresh Chromium on the host machine using Playwright.  Uses the real
host GPU, real platform UA, and real network stack — no Docker container.  This
avoids the SwiftShader / platform-mismatch / CDP-attach fingerprints that cause
bot-detection failures in the AIO sandbox path.

Example:
    ```python
    session = DirectBrowserSession()
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

from genai_tk.tools.direct_browser.models import DirectBrowserConfig

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page


class DirectBrowserSession:
    """Manages a host-local Playwright Chromium browser."""

    def __init__(self, config: DirectBrowserConfig | None = None) -> None:
        self.config = config or DirectBrowserConfig()
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._playwright: object | None = None
        self._connected = False
        self._event_log: list[str] = []

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser session not connected — call connect() first")
        return self._page

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Launch a local Chromium and create a page."""
        if self._connected:
            return

        try:
            from playwright.async_api import async_playwright  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "playwright is required: uv add playwright && uv run playwright install chromium"
            ) from exc

        pw = await async_playwright().start()
        self._playwright = pw

        launch_args = [
            "--disable-blink-features=AutomationControlled",
            f"--lang={self.config.locale}",
            f"--window-size={self.config.viewport_width},{self.config.viewport_height}",
        ]
        launch_args.extend(self.config.extra_args)

        self._browser = await pw.chromium.launch(
            headless=self.config.headless,
            args=launch_args,
        )

        width = self.config.viewport_width + random.randint(-30, 30)
        height = self.config.viewport_height + random.randint(-30, 30)
        self._context = await self._browser.new_context(
            viewport={"width": width, "height": height},
            locale=self.config.locale,
            timezone_id=self.config.timezone_id,
            ignore_https_errors=self.config.ignore_https_errors,
        )
        self._page = await self._context.new_page()
        self._connected = True

        self._attach_debug_listeners()
        logger.info("Direct browser session connected (host-local Playwright)")

    async def close(self) -> None:
        """Close the browser and release resources."""
        if not self._connected:
            return
        self._detach_debug_listeners()
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
        logger.info("Direct browser session closed")

    # ------------------------------------------------------------------
    # Cookie persistence
    # ------------------------------------------------------------------

    async def save_cookies(self, name: str) -> str:
        """Save the current browser context storage state to a JSON file."""
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
        """Load a previously saved storage state into the browser context."""
        file_path = Path(self.config.cookies_dir) / f"{name}_session.json"
        if not file_path.is_file():
            logger.warning(f"Cookie file not found: {file_path}")
            return False
        if not self._context or not self._page:
            raise RuntimeError("Browser session missing active context or page")

        state = json.loads(file_path.read_text())
        await self._context.clear_cookies()
        cookies = state.get("cookies", [])
        if cookies:
            await self._context.add_cookies(cookies)

        origins = state.get("origins", [])
        if origins:
            temp_page = await self._context.new_page()
            try:
                for origin_state in origins:
                    origin = origin_state.get("origin")
                    local_storage = origin_state.get("localStorage", [])
                    if not origin:
                        continue
                    await temp_page.goto(
                        origin,
                        wait_until="domcontentloaded",
                        timeout=self.config.default_timeout_ms,
                    )
                    await temp_page.evaluate(
                        """
                        storageItems => {
                          window.localStorage.clear();
                          for (const item of storageItems) {
                            window.localStorage.setItem(item.name, item.value);
                          }
                        }
                        """,
                        local_storage,
                    )
            finally:
                await temp_page.close()
        logger.info(f"Cookies loaded from {file_path}")
        return True

    # ------------------------------------------------------------------
    # Debug event listeners (same approach as SandboxBrowserSession)
    # ------------------------------------------------------------------

    def get_event_log(self, last_n: int = 50) -> str:
        """Return the last *last_n* browser events as a newline-separated string."""
        entries = self._event_log[-last_n:]
        if not entries:
            return "(no browser events recorded)"
        return "\n".join(entries)

    def _log_event(self, level: str, message: str) -> None:
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._event_log.append(f"[{ts}] {level}: {message}")
        if len(self._event_log) > 200:
            self._event_log = self._event_log[-200:]
        getattr(logger, level.lower(), logger.debug)(message)

    def _attach_debug_listeners(self) -> None:
        if not self.config.log_browser_console:
            return
        if self._page is not None:
            self._page.on("console", self._on_console)
            self._page.on("pageerror", self._on_page_error)
            self._page.on("crash", self._on_page_crash)
            self._page.on("close", self._on_page_close)
            self._page.on("request", self._on_request)
            self._page.on("requestfailed", self._on_request_failed)
            self._page.on("response", self._on_response)
            self._page.on("framenavigated", self._on_frame_navigated)
        if self._browser is not None:
            self._browser.on("disconnected", self._on_browser_disconnected)

    def _detach_debug_listeners(self) -> None:
        try:
            if self._page is not None:
                for evt in (
                    "console",
                    "pageerror",
                    "crash",
                    "close",
                    "request",
                    "requestfailed",
                    "response",
                    "framenavigated",
                ):
                    handler = getattr(self, f"_on_{evt}" if evt != "requestfailed" else "_on_request_failed", None)
                    if handler:
                        self._page.remove_listener(evt, handler)
        except Exception:
            pass
        try:
            if self._browser is not None:
                self._browser.remove_listener("disconnected", self._on_browser_disconnected)
        except Exception:
            pass

    def _on_console(self, msg: object) -> None:
        msg_type = getattr(msg, "type", "log")
        text = str(getattr(msg, "text", msg))
        if msg_type in ("error", "warning"):
            self._log_event("WARNING", f"Browser console [{msg_type}]: {text[:500]}")
        else:
            self._log_event("DEBUG", f"Browser console [{msg_type}]: {text[:300]}")

    def _on_page_error(self, error: object) -> None:
        self._log_event("ERROR", f"Browser page error: {error}")

    def _on_page_crash(self, _: object) -> None:
        self._log_event("ERROR", "Browser page CRASHED — renderer process died")

    def _on_page_close(self, _: object) -> None:
        self._log_event("WARNING", "Browser page closed")

    def _on_browser_disconnected(self, _: object) -> None:
        self._log_event("ERROR", "Browser DISCONNECTED")

    def _on_request(self, request: object) -> None:
        url = getattr(request, "url", "")
        resource_type = getattr(request, "resource_type", "")
        frame = getattr(request, "frame", None)
        is_navigation = getattr(request, "is_navigation_request", lambda: False)()
        if is_navigation and frame and getattr(frame, "parent_frame", None) is None:
            self._log_event("INFO", f"Navigation request → {url}")
            return
        if resource_type in {"xhr", "fetch"}:
            method = getattr(request, "method", "GET")
            self._log_event("INFO", f"{resource_type.upper()} request → {method} {url}")

    def _on_request_failed(self, request: object) -> None:
        url = getattr(request, "url", "")
        resource_type = getattr(request, "resource_type", "unknown")
        failure = getattr(request, "failure", None)
        error_text = ""
        if callable(failure):
            details = failure()
            if isinstance(details, dict):
                error_text = str(details.get("errorText", ""))
        elif isinstance(failure, dict):
            error_text = str(failure.get("errorText", ""))
        message = f"Request failed [{resource_type}] → {url}"
        if error_text:
            message += f" ({error_text})"
        self._log_event("WARNING", message)

    def _on_response(self, response: object) -> None:
        status = getattr(response, "status", 0)
        url = getattr(response, "url", "")
        request = getattr(response, "request", None)
        is_navigation = request and getattr(request, "is_navigation_request", lambda: False)()
        if is_navigation:
            self._log_event("INFO", f"Navigation response ← {status} {url}")
        elif status >= 400:
            resource_type = getattr(request, "resource_type", "unknown") if request else "unknown"
            self._log_event("WARNING", f"HTTP {status} [{resource_type}] ← {url}")

    def _on_frame_navigated(self, frame: object) -> None:
        parent = getattr(frame, "parent_frame", None)
        if parent is None:
            url = getattr(frame, "url", "")
            self._log_event("INFO", f"Main frame navigated ⇒ {url}")

    async def __aenter__(self) -> DirectBrowserSession:
        await self.connect()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()
