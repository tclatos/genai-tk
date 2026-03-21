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
        """Connect to the sandbox browser via CDP and create a page.

        When ``config.launch_mode`` is ``"fresh"``, kills the pre-launched
        sandbox browser and launches a new Chromium instance inside the
        container with bot-detection-resistant flags. The agent then
        connects to this fresh instance via CDP.
        """
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

        if self.config.launch_mode == "fresh":
            await self._connect_fresh(Sandbox, async_playwright)
        else:
            await self._connect_cdp(Sandbox, async_playwright)

        # Set viewport on whichever page we ended up with
        width = self.config.viewport_width + random.randint(-30, 30)
        height = self.config.viewport_height + random.randint(-30, 30)
        await self._page.set_viewport_size({"width": width, "height": height})

        self._connected = True

        # Attach browser-level event listeners for debugging
        self._attach_debug_listeners()
        logger.info("Sandbox browser session connected")

    async def _connect_cdp(self, sandbox_cls: type, async_playwright_fn: object) -> None:
        """Attach to the pre-launched sandbox browser over CDP (default mode)."""
        # Get CDP URL from sandbox API
        client = sandbox_cls(base_url=self._sandbox_url)
        browser_info = client.browser.get_info().data
        cdp_url = browser_info.cdp_url
        logger.info(f"Connecting to sandbox browser via CDP: {cdp_url}")

        # Connect Playwright over CDP
        pw = await async_playwright_fn().start()
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

    async def _connect_fresh(self, sandbox_cls: type, async_playwright_fn: object) -> None:
        """Kill the pre-launched browser, launch a fresh Chromium, and connect via CDP.

        This bypasses the pre-launched browser + CDP attach path that some
        sites detect as bot traffic.  The fresh browser runs with explicit
        anti-detection flags and the agent reconnects via CDP to control it.

        The method first discovers the CDP port used by the pre-launched browser
        (via the sandbox API) and reuses it so the sandbox nginx proxy continues
        to route ``/cdp/`` traffic correctly.
        """
        import asyncio as _aio  # noqa: PLC0415
        from urllib.parse import urlparse  # noqa: PLC0415

        from agent_sandbox import AsyncSandbox  # noqa: PLC0415

        logger.info("Fresh-launch mode: killing pre-launched browser and starting a new one")

        # Discover the CDP port the pre-launched browser uses so we reuse it.
        # The sandbox nginx proxy routes /cdp/ to this port — we must keep it.
        sync_client = sandbox_cls(base_url=self._sandbox_url)
        try:
            browser_info = sync_client.browser.get_info().data
            original_cdp_url = browser_info.cdp_url
            parsed = urlparse(original_cdp_url)
            cdp_port = parsed.port or 9222
        except Exception:
            cdp_port = 9222
        logger.info(f"Pre-launched browser CDP port: {cdp_port}")

        client = AsyncSandbox(base_url=self._sandbox_url)

        # Kill the pre-launched Chromium to free the CDP port
        kill_cmd = "pkill -f chromium || pkill -f chrome || true"
        await client.shell.exec_command(command=kill_cmd)
        await _aio.sleep(2)  # let the process die

        # Launch a fresh Chromium with anti-detection flags and remote debugging
        locale_tag = self.config.locale
        tz = self.config.timezone_id
        launch_cmd = (
            "nohup /usr/bin/chromium-browser"
            " --no-first-run --no-default-browser-check"
            " --disable-blink-features=AutomationControlled"
            f" --lang={locale_tag}"
            f" --time-zone-for-testing={tz}"
            " --remote-debugging-address=0.0.0.0"
            f" --remote-debugging-port={cdp_port}"
            " --headless=new"
            f" --window-size={self.config.viewport_width},{self.config.viewport_height}"
            " about:blank"
            " > /tmp/chromium-fresh.log 2>&1 &"
        )
        await client.shell.exec_command(command=launch_cmd)

        # Wait for the new browser to be ready
        cdp_url = None
        for _attempt in range(15):
            await _aio.sleep(1)
            try:
                import httpx  # noqa: PLC0415

                async with httpx.AsyncClient(trust_env=False) as hc:
                    resp = await hc.get(f"{self._sandbox_url}/cdp/json/version", timeout=3.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        cdp_url = data.get("webSocketDebuggerUrl", "")
                        if cdp_url:
                            break
            except Exception:
                continue

        if not cdp_url:
            raise RuntimeError("Fresh Chromium did not start within 15s — check container logs")

        logger.info(f"Fresh browser ready, connecting via CDP: {cdp_url}")

        # Connect Playwright over CDP to the fresh browser
        pw = await async_playwright_fn().start()
        self._playwright = pw
        self._browser = await pw.chromium.connect_over_cdp(cdp_url)

        # With a fresh browser we own the context fully
        width = self.config.viewport_width + random.randint(-30, 30)
        height = self.config.viewport_height + random.randint(-30, 30)
        self._context = await self._browser.new_context(
            viewport={"width": width, "height": height},
            locale=self.config.locale,
            timezone_id=self.config.timezone_id,
            ignore_https_errors=self.config.ignore_https_errors,
        )
        self._owns_context = True
        self._page = await self._context.new_page()

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
            self._attach_page_debug_listeners(self._page)
        if self._browser is not None:
            self._browser.on("disconnected", self._on_browser_disconnected)

    def _detach_debug_listeners(self) -> None:
        """Remove previously attached event listeners."""
        try:
            if self._page is not None:
                self._detach_page_debug_listeners(self._page)
        except Exception:
            pass
        try:
            if self._browser is not None:
                self._browser.remove_listener("disconnected", self._on_browser_disconnected)
        except Exception:
            pass

    def _attach_page_debug_listeners(self, page: Page) -> None:
        """Attach debug listeners to a page instance."""
        _page_handlers = {
            "console": self._on_console,
            "pageerror": self._on_page_error,
            "crash": self._on_page_crash,
            "close": self._on_page_close,
            "request": self._on_request,
            "requestfailed": self._on_request_failed,
            "response": self._on_response,
            "framenavigated": self._on_frame_navigated,
        }
        for evt, handler in _page_handlers.items():
            page.on(evt, handler)

    def _detach_page_debug_listeners(self, page: Page) -> None:
        """Detach debug listeners from a page instance."""
        _page_handlers = {
            "console": self._on_console,
            "pageerror": self._on_page_error,
            "crash": self._on_page_crash,
            "close": self._on_page_close,
            "request": self._on_request,
            "requestfailed": self._on_request_failed,
            "response": self._on_response,
            "framenavigated": self._on_frame_navigated,
        }
        for evt, handler in _page_handlers.items():
            page.remove_listener(evt, handler)

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
        """Log main-frame navigations and XHR/fetch requests."""
        url = getattr(request, "url", "")
        method = getattr(request, "method", "GET")
        resource_type = getattr(request, "resource_type", "")
        frame = getattr(request, "frame", None)
        is_navigation = getattr(request, "is_navigation_request", lambda: False)()
        if is_navigation and frame and getattr(frame, "parent_frame", None) is None:
            self._log_event("INFO", f"Navigation request → {url}")
            return
        if resource_type in {"xhr", "fetch"}:
            self._log_event("INFO", f"{resource_type.upper()} request → {method} {url}")

    def _on_request_failed(self, request: object) -> None:
        """Log network request failures."""
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
        """Log navigation responses and failing XHR/fetch requests."""
        status = getattr(response, "status", 0)
        url = getattr(response, "url", "")
        request = getattr(response, "request", None)
        is_navigation = request and getattr(request, "is_navigation_request", lambda: False)()
        frame = request and getattr(request, "frame", None)
        is_main = frame and getattr(frame, "parent_frame", None) is None
        resource_type = getattr(request, "resource_type", "") if request else ""
        if is_navigation and is_main:
            level = "WARNING" if status >= 400 else "INFO"
            self._log_event(level, f"Navigation response ← HTTP {status} {url}")
            return
        if resource_type in {"xhr", "fetch"}:
            level = "WARNING" if status >= 400 else "INFO"
            self._log_event(level, f"{resource_type.upper()} response ← HTTP {status} {url}")

    def _on_frame_navigated(self, frame: object) -> None:
        """Log completed main-frame navigations, including client-side route changes."""
        if getattr(frame, "parent_frame", None) is not None:
            return
        url = getattr(frame, "url", "")
        if url:
            self._log_event("INFO", f"Main frame navigated ⇒ {url}")

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
        """Load a previously saved storage state into the active browser context.

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

    async def __aenter__(self) -> SandboxBrowserSession:
        await self.connect()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()
