"""LangChain tools for host-local browser automation via direct Playwright.

Same tool names and interface as ``genai_tk.tools.sandbox_browser.tools`` so
that SKILL.md files work unchanged with either suite.  The only difference is
the session backend: ``DirectBrowserSession`` launches a host-local Chromium
instead of connecting to an AIO sandbox container.
"""

from __future__ import annotations

import asyncio
import base64
import random
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

from genai_tk.tools.direct_browser.session import DirectBrowserSession
from genai_tk.tools.sandbox_browser.models import CredentialRef, PageSummary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _page_summary(session: DirectBrowserSession, max_text: int = 2000) -> str:
    page = session.page
    title = await page.title()
    url = page.url
    try:
        text = await page.inner_text("body")
        text = text[:max_text].strip()
    except Exception:
        text = ""
    summary = PageSummary(url=url, title=title, text_snippet=text)
    return f"URL: {summary.url}\nTitle: {summary.title}\n\n{summary.text_snippet}"


async def _human_type(session: DirectBrowserSession, selector: str, text: str) -> None:
    page = session.page
    delay = session.config.slow_type_ms
    timeout = session.config.default_timeout_ms
    locator = page.locator(selector).first
    await locator.wait_for(state="visible", timeout=timeout)
    await locator.scroll_into_view_if_needed(timeout=timeout)
    await locator.click(timeout=timeout)
    # Clear existing content — try fill first, fall back to select-all+delete
    try:
        await locator.fill("", timeout=5000)
    except Exception:
        await page.keyboard.press("Control+a")
        await page.keyboard.press("Backspace")
    # Type with human-like delays
    await locator.press_sequentially(text, delay=delay)


# ---------------------------------------------------------------------------
# Tool base
# ---------------------------------------------------------------------------


class _BrowserTool(BaseTool):
    """Base class for direct browser tools sharing a session."""

    session: DirectBrowserSession = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def _ensure_connected(self) -> None:
        if not self.session.connected:
            await self.session.connect()
            return
        page = self.session.page
        try:
            await page.evaluate("1")
        except Exception as exc:
            from loguru import logger  # noqa: PLC0415

            message = str(exc).lower()
            transient = (
                "execution context was destroyed",
                "cannot find context with specified id",
                "inspected target navigated or closed",
                "navigation",
            )
            is_closed = getattr(page, "is_closed", None)
            page_closed = is_closed() if callable(is_closed) else False
            if not page_closed and any(m in message for m in transient):
                await asyncio.sleep(0.1)
                return
            logger.warning("Browser context appears dead — reconnecting")
            try:
                await self.session.close()
                await self.session.connect()
            except Exception as reconn_exc:
                raise RuntimeError(
                    f"Browser connection lost and reconnect failed ({reconn_exc}). "
                    "The browser may have been closed or crashed."
                ) from reconn_exc


# ---------------------------------------------------------------------------
# Tools — same names as sandbox_browser for SKILL.md compatibility
# ---------------------------------------------------------------------------


class BrowserNavigateTool(_BrowserTool):
    name: str = "browser_navigate"
    description: str = (
        "Navigate the browser to a URL. The 'url' argument is required. "
        "Returns the page title, URL, and a text snippet. "
        "Use this to go to websites, follow links, or load specific pages."
    )

    def _run(self, url: str = "", wait_until: str = "domcontentloaded") -> str:
        return asyncio.get_event_loop().run_until_complete(self._arun(url=url, wait_until=wait_until))

    async def _arun(self, url: str = "", wait_until: str = "domcontentloaded") -> str:
        if not url or not url.strip():
            return "Error: 'url' argument is required. Provide the full URL to navigate to."
        await self._ensure_connected()
        page = self.session.page
        try:
            await page.goto(url.strip(), wait_until=wait_until, timeout=self.session.config.default_timeout_ms)
        except Exception as exc:
            return f"Navigation failed: {exc}"
        return await _page_summary(self.session)


class BrowserClickTool(_BrowserTool):
    name: str = "browser_click"
    description: str = (
        "Click an element on the page using a CSS selector or text content. "
        "Examples: '#submit-btn', 'button:has-text(\"Login\")', 'a:has-text(\"Next\")'. "
        "Returns the page state after clicking."
    )

    def _run(self, selector: str = "") -> str:
        return asyncio.get_event_loop().run_until_complete(self._arun(selector))

    async def _arun(self, selector: str = "") -> str:
        if not selector:
            return "Error: 'selector' argument is required."
        await self._ensure_connected()
        page = self.session.page
        try:
            await page.click(selector, timeout=self.session.config.default_timeout_ms)
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception as exc:
            return f"Click failed on '{selector}': {exc}"
        await asyncio.sleep(0.5)
        return await _page_summary(self.session)


class BrowserTypeTool(_BrowserTool):
    name: str = "browser_type"
    description: str = (
        "Type text into a visible form field identified by CSS selector. "
        "The element must be visible on screen. Use specific selectors like "
        "input[name='username'], input.txt-answer-box, textarea#comment — avoid generic 'input'. "
        "Uses human-like typing delays. For credentials, use browser_fill_credential instead. "
        "To submit after typing, set press_enter=true."
    )

    def _run(self, selector: str = "", text: str = "", press_enter: bool = False) -> str:
        return asyncio.get_event_loop().run_until_complete(
            self._arun(selector=selector, text=text, press_enter=press_enter)
        )

    async def _arun(self, selector: str = "", text: str = "", press_enter: bool = False) -> str:
        if not selector:
            return "Error: 'selector' argument is required."
        if not text:
            return "Error: 'text' argument is required."
        await self._ensure_connected()
        try:
            await _human_type(self.session, selector, text)
            if press_enter:
                await self.session.page.keyboard.press("Enter")
                await asyncio.sleep(0.5)
        except Exception as exc:
            return f"Typing failed on '{selector}': {exc}"
        return f"Typed into '{selector}' successfully."


class BrowserFillCredentialTool(_BrowserTool):
    name: str = "browser_fill_credential"
    description: str = (
        "Fill a form field with a credential stored in an environment variable. "
        "The actual credential value is never visible to you — only a confirmation is returned. "
        "Args: selector (CSS selector for the input field), credential_env (env var name, "
        "e.g. 'ENEDIS_USERNAME'). Only pre-approved env vars from the allowlist can be used."
    )

    def _run(self, selector: str, credential_env: str) -> str:
        return asyncio.get_event_loop().run_until_complete(self._arun(selector=selector, credential_env=credential_env))

    async def _arun(self, selector: str, credential_env: str) -> str:
        await self._ensure_connected()
        ref = CredentialRef(env=credential_env)
        try:
            value = ref.resolve(allowlist=self.session.config.allowed_credential_envs)
        except (PermissionError, ValueError) as exc:
            return f"Credential error: {exc}"
        try:
            await _human_type(self.session, selector, value)
        except Exception as exc:
            return f"Failed to fill credential in '{selector}': {exc}"
        return f"Credential from ${credential_env} filled into '{selector}'."


class BrowserScreenshotTool(_BrowserTool):
    name: str = "browser_screenshot"
    description: str = (
        "Take a screenshot of the current browser page. Returns a base64-encoded PNG image. "
        "Use this when you need visual verification, when text extraction is insufficient, "
        "or when dealing with charts/images. This costs extra tokens — use sparingly."
    )

    def _run(self) -> str:
        return asyncio.get_event_loop().run_until_complete(self._arun())

    async def _arun(self, **kwargs: Any) -> str:
        await self._ensure_connected()
        page = self.session.page
        try:
            screenshot_bytes = await page.screenshot(type="png", full_page=False)
            b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        except Exception as exc:
            return f"Screenshot failed: {exc}"


class BrowserReadPageTool(_BrowserTool):
    name: str = "browser_read_page"
    description: str = (
        "Read text content from the current page. Optionally provide a CSS selector "
        "to read only a specific element. Returns the current URL, title, and text content. "
        "Without a selector, returns the full page body text (truncated to ~4000 chars)."
    )

    def _run(self, selector: str | None = None) -> str:
        return asyncio.get_event_loop().run_until_complete(self._arun(selector=selector))

    async def _arun(self, selector: str | None = None, **kwargs: Any) -> str:
        await self._ensure_connected()
        page = self.session.page
        try:
            if selector:
                element = await page.query_selector(selector)
                if not element:
                    return f"Element not found: '{selector}'"
                text = await element.inner_text()
            else:
                text = await page.inner_text("body")
            summary = PageSummary(
                url=page.url,
                title=await page.title(),
                text_snippet=text,
            )
            return f"URL: {summary.url}\nTitle: {summary.title}\n\n{summary.text_snippet}"
        except Exception as exc:
            return f"Read page failed: {exc}"


class BrowserScrollTool(_BrowserTool):
    name: str = "browser_scroll"
    description: str = (
        "Scroll the page. Direction: 'down' or 'up'. Amount is in pixels (default 500). "
        "Returns a brief summary of what's now visible."
    )

    def _run(self, direction: str = "down", amount: int = 500) -> str:
        return asyncio.get_event_loop().run_until_complete(self._arun(direction=direction, amount=amount))

    async def _arun(self, direction: str = "down", amount: int = 500, **kwargs: Any) -> str:
        await self._ensure_connected()
        page = self.session.page
        delta = amount if direction == "down" else -amount
        try:
            await page.evaluate(f"window.scrollBy(0, {delta})")
        except Exception as exc:
            return f"Scroll failed: {exc}"
        await asyncio.sleep(0.3)
        return await _page_summary(self.session, max_text=1000)


class BrowserWaitTool(_BrowserTool):
    name: str = "browser_wait"
    description: str = (
        "Wait for an element to appear on the page (by CSS selector), or wait a fixed number "
        "of milliseconds. You can also wait for a browser load state like 'domcontentloaded', "
        "'load', or 'networkidle'. Useful after navigation or clicks that trigger async loading."
    )

    def _run(self, selector: str | None = None, timeout_ms: int = 10000, load_state: str | None = None) -> str:
        return asyncio.get_event_loop().run_until_complete(
            self._arun(selector=selector, timeout_ms=timeout_ms, load_state=load_state)
        )

    async def _arun(
        self,
        selector: str | None = None,
        timeout_ms: int = 10000,
        load_state: str | None = None,
        **kwargs: Any,
    ) -> str:
        await self._ensure_connected()
        page = self.session.page
        if selector:
            try:
                await page.wait_for_selector(selector, timeout=timeout_ms)
                return f"Element '{selector}' appeared."
            except Exception:
                return f"Timeout: element '{selector}' did not appear within {timeout_ms}ms."
        if load_state:
            try:
                await page.wait_for_load_state(load_state, timeout=timeout_ms)
                return f"Load state '{load_state}' reached."
            except Exception:
                return f"Timeout: load state '{load_state}' not reached within {timeout_ms}ms."
        await asyncio.sleep(timeout_ms / 1000)
        return f"Waited {timeout_ms}ms."


class BrowserSaveCookiesTool(_BrowserTool):
    name: str = "browser_save_cookies"
    description: str = (
        "Save the current browser session (cookies, localStorage) to a file. "
        "Use a descriptive name like 'enedis' or 'sharepoint'. "
        "The session can be restored later with browser_load_cookies to skip re-authentication."
    )

    def _run(self, name: str) -> str:
        return asyncio.get_event_loop().run_until_complete(self._arun(name=name))

    async def _arun(self, name: str, **kwargs: Any) -> str:
        await self._ensure_connected()
        try:
            path = await self.session.save_cookies(name)
            return f"Session saved to {path}"
        except Exception as exc:
            return f"Failed to save cookies: {exc}"


class BrowserLoadCookiesTool(_BrowserTool):
    name: str = "browser_load_cookies"
    description: str = (
        "Load a previously saved browser session (cookies, localStorage) from disk. "
        "Use the same name that was used with browser_save_cookies. "
        "Returns success or failure. If successful, subsequent navigations will use the restored session."
    )

    def _run(self, name: str) -> str:
        return asyncio.get_event_loop().run_until_complete(self._arun(name=name))

    async def _arun(self, name: str, **kwargs: Any) -> str:
        await self._ensure_connected()
        try:
            loaded = await self.session.load_cookies(name)
            if loaded:
                return f"Session '{name}' loaded successfully."
            return f"No saved session found for '{name}'."
        except Exception as exc:
            return f"Failed to load cookies: {exc}"


class BrowserGetLogsTool(_BrowserTool):
    name: str = "browser_get_logs"
    description: str = (
        "Retrieve recent browser event logs (console messages, JS errors, page crashes, "
        "navigations, HTTP errors). Use this to diagnose why a page failed to load, "
        "why the browser disconnected, or what happened before an error. "
        "The log survives browser crashes. Optional arg: last_n (default 50)."
    )

    def _run(self, last_n: int = 50) -> str:
        return self.session.get_event_log(last_n=last_n)

    async def _arun(self, last_n: int = 50, **kwargs: Any) -> str:
        return self.session.get_event_log(last_n=last_n)


class BrowserEvaluateTool(_BrowserTool):
    name: str = "browser_evaluate"
    description: str = (
        "Execute a JavaScript expression in the current page and return the result. "
        "The expression must return a JSON-serializable value (string, number, object, array). "
        "Useful for inspecting page state, checking element attributes, reading JS variables, "
        "or diagnosing browser fingerprint (e.g. navigator.userAgent). "
        "Example: 'document.querySelector(\"h1\").textContent' or 'navigator.userAgent'."
    )

    def _run(self, expression: str) -> str:
        return asyncio.get_event_loop().run_until_complete(self._arun(expression=expression))

    async def _arun(self, expression: str, **kwargs: Any) -> str:
        await self._ensure_connected()
        page = self.session.page
        try:
            result = await page.evaluate(expression)
            if isinstance(result, str):
                return result[:4000]
            import json  # noqa: PLC0415

            return json.dumps(result, ensure_ascii=False, default=str)[:4000]
        except Exception as exc:
            return f"Evaluate failed: {exc}"


_DIAGNOSE_JS = """
(() => {
    const d = {};
    d.userAgent = navigator.userAgent;
    d.platform = navigator.platform;
    d.languages = navigator.languages;
    d.webdriver = navigator.webdriver;
    d.timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    d.uaDataPlatform = navigator.userAgentData?.platform || 'N/A';
    d.uaDataBrands = navigator.userAgentData?.brands?.map(b => b.brand + '/' + b.version) || [];
    try {
        const c = document.createElement('canvas');
        const gl = c.getContext('webgl') || c.getContext('experimental-webgl');
        const ext = gl?.getExtension('WEBGL_debug_renderer_info');
        d.webglRenderer = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'N/A';
    } catch(e) { d.webglRenderer = 'error'; }
    d.screen = screen.width + 'x' + screen.height;
    d.colorDepth = screen.colorDepth;
    d.hardwareConcurrency = navigator.hardwareConcurrency;
    d.deviceMemory = navigator.deviceMemory || 'N/A';
    d.cookieEnabled = navigator.cookieEnabled;
    d.maxTouchPoints = navigator.maxTouchPoints;
    return d;
})()
"""


class BrowserDiagnoseTool(_BrowserTool):
    name: str = "browser_diagnose"
    description: str = (
        "Collect browser fingerprint diagnostics: user agent, platform, WebGL renderer, "
        "timezone, webdriver flag, screen size, and client hints. Returns a JSON object. "
        "Use this to check what a website's bot-detection might see, or to debug why "
        "a site is blocking the browser. Also returns recent event logs."
    )

    def _run(self) -> str:
        return asyncio.get_event_loop().run_until_complete(self._arun())

    async def _arun(self, **kwargs: Any) -> str:
        await self._ensure_connected()
        page = self.session.page
        import json as _json  # noqa: PLC0415

        parts = []
        try:
            fp = await page.evaluate(_DIAGNOSE_JS)
            parts.append("Fingerprint:\n" + _json.dumps(fp, indent=2, ensure_ascii=False))
        except Exception as exc:
            parts.append(f"Fingerprint collection failed: {exc}")

        parts.append(f"\nURL: {page.url}")
        try:
            parts.append(f"Title: {await page.title()}")
        except Exception:
            pass

        logs = self.session.get_event_log(last_n=20)
        parts.append(f"\nRecent events:\n{logs}")

        return "\n".join(parts)[:4000]


class BrowserWaitForUserTool(_BrowserTool):
    name: str = "browser_wait_for_user"
    description: str = (
        "Pause and wait for the user to complete a manual action in the browser "
        "(e.g. entering credentials, solving a CAPTCHA, completing SSO login). "
        "The tool polls the page every few seconds and returns once the URL changes "
        "(indicating a redirect after login) or the maximum wait time is reached. "
        "Use this when the user needs to interact with the browser manually."
    )

    def _run(self, message: str = "", timeout_seconds: int = 120, poll_interval_seconds: int = 5) -> str:
        return asyncio.get_event_loop().run_until_complete(
            self._arun(message=message, timeout_seconds=timeout_seconds, poll_interval_seconds=poll_interval_seconds)
        )

    async def _arun(
        self,
        message: str = "",
        timeout_seconds: int = 120,
        poll_interval_seconds: int = 5,
        **kwargs: Any,
    ) -> str:
        await self._ensure_connected()
        page = self.session.page
        initial_url = page.url
        display_msg = message or "Waiting for user to complete action in the browser..."
        from loguru import logger  # noqa: PLC0415

        logger.info("⏳ {} (timeout={}s, polling every {}s)", display_msg, timeout_seconds, poll_interval_seconds)

        elapsed = 0
        while elapsed < timeout_seconds:
            await asyncio.sleep(poll_interval_seconds)
            elapsed += poll_interval_seconds
            try:
                current_url = page.url
            except Exception:
                return f"Browser connection lost after {elapsed}s while waiting."
            if current_url != initial_url:
                return (
                    f"Page URL changed after {elapsed}s.\n"
                    f"Old URL: {initial_url}\n"
                    f"New URL: {current_url}\n"
                    + await _page_summary(self.session)
                )

        return (
            f"Timeout reached ({timeout_seconds}s) — URL did not change.\n"
            f"Current URL: {page.url}\n"
            + await _page_summary(self.session)
        )


ALL_BROWSER_TOOLS: list[type[_BrowserTool]] = [
    BrowserNavigateTool,
    BrowserClickTool,
    BrowserTypeTool,
    BrowserFillCredentialTool,
    BrowserScreenshotTool,
    BrowserReadPageTool,
    BrowserScrollTool,
    BrowserWaitTool,
    BrowserWaitForUserTool,
    BrowserSaveCookiesTool,
    BrowserLoadCookiesTool,
    BrowserGetLogsTool,
    BrowserEvaluateTool,
    BrowserDiagnoseTool,
]
