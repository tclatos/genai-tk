"""LangChain tools for agent-driven browser automation inside an AIO sandbox.

Each tool is a thin wrapper around Playwright operations on a shared
``SandboxBrowserSession``.  The agent calls these primitive tools — guided by
site-specific SKILL.md files — to navigate, interact, and extract data from
websites running in the sandbox's real Chromium.

Security:
    ``browser_fill_credential`` resolves credentials from environment variables
    and types them into form fields.  The actual credential value is **never**
    returned to the LLM — only a confirmation message.
"""

from __future__ import annotations

import asyncio
import base64
import random
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

from genai_tk.tools.sandbox_browser.models import CredentialRef, PageSummary
from genai_tk.tools.sandbox_browser.session import SandboxBrowserSession

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _page_summary(session: SandboxBrowserSession, max_text: int = 2000) -> str:
    """Return a short textual summary of the current page state."""
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


async def _human_type(session: SandboxBrowserSession, selector: str, text: str) -> None:
    """Type text into a field with human-like per-character delays."""
    page = session.page
    delay = session.config.slow_type_ms
    await page.click(selector)
    await page.fill(selector, "")
    for char in text:
        await page.type(selector, char, delay=delay + random.randint(-15, 15))


# ---------------------------------------------------------------------------
# Tool base
# ---------------------------------------------------------------------------


class _BrowserTool(BaseTool):
    """Base class for sandbox browser tools sharing a session."""

    session: SandboxBrowserSession = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    async def _ensure_connected(self) -> None:
        if not self.session.connected:
            await self.session.connect()
            return
        # Detect stale connection (browser/context closed by remote crash or timeout)
        try:
            page = self.session.page
            # Quick liveness check — if the browser is dead this throws
            await page.evaluate("1")
        except Exception:
            from loguru import logger  # noqa: PLC0415

            logger.warning("Browser context appears dead — reconnecting")
            try:
                await self.session.close()
                await self.session.connect()
            except Exception as reconn_exc:
                raise RuntimeError(
                    f"Browser connection lost and reconnect failed ({reconn_exc}). "
                    "The sandbox container may have been terminated. "
                    "Check 'cli sandbox list' or VNC for status."
                ) from reconn_exc


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class BrowserNavigateTool(_BrowserTool):
    """Navigate the browser to a URL."""

    name: str = "browser_navigate"
    description: str = (
        "Navigate the browser to a URL. Returns the page title, URL, and a text snippet. "
        "Use this to go to websites, follow links, or load specific pages."
    )

    def _run(self, url: str) -> str:
        return asyncio.get_event_loop().run_until_complete(self._arun(url))

    async def _arun(self, url: str) -> str:
        await self._ensure_connected()
        page = self.session.page
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=self.session.config.default_timeout_ms)
        except Exception as exc:
            return f"Navigation failed: {exc}"
        return await _page_summary(self.session)


class BrowserClickTool(_BrowserTool):
    """Click an element on the page."""

    name: str = "browser_click"
    description: str = (
        "Click an element on the page using a CSS selector or text content. "
        "Examples: '#submit-btn', 'button:has-text(\"Login\")', 'a:has-text(\"Next\")'. "
        "Returns the page state after clicking."
    )

    def _run(self, selector: str) -> str:
        return asyncio.get_event_loop().run_until_complete(self._arun(selector))

    async def _arun(self, selector: str) -> str:
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
    """Type text into a form field."""

    name: str = "browser_type"
    description: str = (
        "Type text into a form field identified by CSS selector. "
        "Uses human-like typing delays. For credentials, use browser_fill_credential instead."
    )

    def _run(self, selector: str, text: str) -> str:
        return asyncio.get_event_loop().run_until_complete(self._arun(selector=selector, text=text))

    async def _arun(self, selector: str, text: str) -> str:
        await self._ensure_connected()
        try:
            await _human_type(self.session, selector, text)
        except Exception as exc:
            return f"Typing failed on '{selector}': {exc}"
        return f"Typed into '{selector}' successfully."


class BrowserFillCredentialTool(_BrowserTool):
    """Securely fill a credential from an environment variable into a form field.

    The credential value is NEVER returned to the LLM.
    """

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
    """Take a screenshot of the current page."""

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
    """Extract text content from the current page or a specific element."""

    name: str = "browser_read_page"
    description: str = (
        "Read text content from the current page. Optionally provide a CSS selector "
        "to read only a specific element. Returns the text content. "
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
            return PageSummary(
                url=page.url,
                title=await page.title(),
                text_snippet=text,
            ).text_snippet
        except Exception as exc:
            return f"Read page failed: {exc}"


class BrowserScrollTool(_BrowserTool):
    """Scroll the page."""

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
    """Wait for an element to appear or a fixed duration."""

    name: str = "browser_wait"
    description: str = (
        "Wait for an element to appear on the page (by CSS selector), or wait a fixed number "
        "of milliseconds. Useful after navigation or clicks that trigger async loading."
    )

    def _run(self, selector: str | None = None, timeout_ms: int = 10000) -> str:
        return asyncio.get_event_loop().run_until_complete(self._arun(selector=selector, timeout_ms=timeout_ms))

    async def _arun(self, selector: str | None = None, timeout_ms: int = 10000, **kwargs: Any) -> str:
        await self._ensure_connected()
        page = self.session.page
        if selector:
            try:
                await page.wait_for_selector(selector, timeout=timeout_ms)
                return f"Element '{selector}' appeared."
            except Exception:
                return f"Timeout: element '{selector}' did not appear within {timeout_ms}ms."
        else:
            await asyncio.sleep(timeout_ms / 1000)
            return f"Waited {timeout_ms}ms."


class BrowserSaveCookiesTool(_BrowserTool):
    """Save the current session cookies to disk."""

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
    """Load a previously saved session from disk."""

    name: str = "browser_load_cookies"
    description: str = (
        "Load a previously saved browser session (cookies, localStorage) from disk. "
        "Use the same name that was used with browser_save_cookies. "
        "Returns success or failure. If successful, subsequent navigations will use the restored session."
    )

    def _run(self, name: str) -> str:
        return asyncio.get_event_loop().run_until_complete(self._arun(name=name))

    async def _arun(self, name: str, **kwargs: Any) -> str:
        # Connect first so the browser is available when cookies exist
        await self._ensure_connected()
        try:
            loaded = await self.session.load_cookies(name)
            if loaded:
                return f"Session '{name}' loaded successfully."
            return f"No saved session found for '{name}'."
        except Exception as exc:
            return f"Failed to load cookies: {exc}"


class BrowserGetLogsTool(_BrowserTool):
    """Retrieve the in-memory browser event log for self-diagnosis.

    The event log survives browser disconnects, making it useful for
    post-mortem analysis when the browser crashes or the container dies.
    """

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


# Registry for easy access
ALL_BROWSER_TOOLS: list[type[_BrowserTool]] = [
    BrowserNavigateTool,
    BrowserClickTool,
    BrowserTypeTool,
    BrowserFillCredentialTool,
    BrowserScreenshotTool,
    BrowserReadPageTool,
    BrowserScrollTool,
    BrowserWaitTool,
    BrowserSaveCookiesTool,
    BrowserLoadCookiesTool,
    BrowserGetLogsTool,
]
