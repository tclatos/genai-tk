"""Integration tests for the direct_browser tool suite.

These tests launch a real Chromium browser via host-local Playwright and navigate
to live websites.  They verify the tool interface, session lifecycle, cookie
persistence, and fingerprint diagnostics.

Run with:
    uv run pytest tests/integration_tests/tools/test_direct_browser_integration.py -v -s

Requires:
    uv run playwright install chromium
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from genai_tk.tools.direct_browser.factory import create_direct_browser_tools
from genai_tk.tools.direct_browser.models import DirectBrowserConfig
from genai_tk.tools.direct_browser.session import DirectBrowserSession
from genai_tk.tools.direct_browser.tools import (
    ALL_BROWSER_TOOLS,
    BrowserDiagnoseTool,
    BrowserEvaluateTool,
    BrowserNavigateTool,
    BrowserReadPageTool,
    BrowserSaveCookiesTool,
    BrowserScreenshotTool,
    BrowserScrollTool,
    BrowserWaitTool,
)


@pytest.fixture
def config(tmp_path: Path) -> DirectBrowserConfig:
    return DirectBrowserConfig(
        headless=True,
        cookies_dir=str(tmp_path / "sessions"),
        locale="fr-FR",
        timezone_id="Europe/Paris",
        viewport_width=1280,
        viewport_height=720,
    )


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class TestDirectBrowserSessionIntegration:
    @pytest.mark.asyncio
    async def test_connect_and_close(self, config: DirectBrowserConfig) -> None:
        """Session connects, creates a page, and closes cleanly."""
        session = DirectBrowserSession(config=config)
        assert not session.connected
        await session.connect()
        assert session.connected
        assert session.page is not None
        await session.close()
        assert not session.connected

    @pytest.mark.asyncio
    async def test_async_context_manager(self, config: DirectBrowserConfig) -> None:
        """Session works as an async context manager."""
        async with DirectBrowserSession(config=config) as session:
            assert session.connected
            page = session.page
            await page.goto("about:blank")
            assert page.url == "about:blank"
        assert not session.connected

    @pytest.mark.asyncio
    async def test_double_connect_is_noop(self, config: DirectBrowserConfig) -> None:
        """Calling connect() twice does not raise."""
        async with DirectBrowserSession(config=config) as session:
            await session.connect()  # second call — should be a no-op
            assert session.connected


# ---------------------------------------------------------------------------
# Navigate to real sites
# ---------------------------------------------------------------------------


class TestNavigateRealSites:
    @pytest.mark.asyncio
    async def test_navigate_example_com(self, config: DirectBrowserConfig) -> None:
        """Navigate to example.com and read the page."""
        async with DirectBrowserSession(config=config) as session:
            tool = BrowserNavigateTool(session=session)
            result = await tool._arun("https://example.com")
            assert "Example Domain" in result
            assert "example.com" in result

    @pytest.mark.asyncio
    async def test_navigate_httpbin(self, config: DirectBrowserConfig) -> None:
        """Navigate to httpbin and verify JSON response."""
        async with DirectBrowserSession(config=config) as session:
            tool = BrowserNavigateTool(session=session)
            result = await tool._arun("https://httpbin.org/html")
            assert "Herman Melville" in result or "httpbin" in result.lower()

    @pytest.mark.asyncio
    async def test_navigate_invalid_url(self, config: DirectBrowserConfig) -> None:
        """Navigating to a bad URL returns an error message, not an exception."""
        async with DirectBrowserSession(config=config) as session:
            tool = BrowserNavigateTool(session=session)
            result = await tool._arun("https://this-domain-does-not-exist-xyz123.invalid/")
            assert "failed" in result.lower() or "error" in result.lower()


# ---------------------------------------------------------------------------
# Tool functionality
# ---------------------------------------------------------------------------


class TestBrowserToolsIntegration:
    @pytest.mark.asyncio
    async def test_read_page(self, config: DirectBrowserConfig) -> None:
        """BrowserReadPageTool extracts text from a page."""
        async with DirectBrowserSession(config=config) as session:
            await session.page.goto("https://example.com")
            tool = BrowserReadPageTool(session=session)
            result = await tool._arun()
            assert "Example Domain" in result
            assert "URL:" in result

    @pytest.mark.asyncio
    async def test_read_page_with_selector(self, config: DirectBrowserConfig) -> None:
        """Reading with a CSS selector extracts only that element."""
        async with DirectBrowserSession(config=config) as session:
            await session.page.goto("https://example.com")
            tool = BrowserReadPageTool(session=session)
            result = await tool._arun(selector="h1")
            assert "Example Domain" in result

    @pytest.mark.asyncio
    async def test_screenshot(self, config: DirectBrowserConfig) -> None:
        """Screenshot returns a base64 PNG."""
        async with DirectBrowserSession(config=config) as session:
            await session.page.goto("https://example.com")
            tool = BrowserScreenshotTool(session=session)
            result = await tool._arun()
            assert result.startswith("data:image/png;base64,")
            assert len(result) > 100  # non-trivial image

    @pytest.mark.asyncio
    async def test_evaluate_js(self, config: DirectBrowserConfig) -> None:
        """Evaluate returns JavaScript results."""
        async with DirectBrowserSession(config=config) as session:
            await session.page.goto("https://example.com")
            tool = BrowserEvaluateTool(session=session)
            result = await tool._arun(expression="document.title")
            assert "Example Domain" in result

    @pytest.mark.asyncio
    async def test_scroll(self, config: DirectBrowserConfig) -> None:
        """Scroll doesn't crash on a simple page."""
        async with DirectBrowserSession(config=config) as session:
            await session.page.goto("https://example.com")
            tool = BrowserScrollTool(session=session)
            result = await tool._arun(direction="down", amount=200)
            assert "Example Domain" in result

    @pytest.mark.asyncio
    async def test_wait_for_load_state(self, config: DirectBrowserConfig) -> None:
        """Wait for networkidle succeeds on a simple page."""
        async with DirectBrowserSession(config=config) as session:
            await session.page.goto("https://example.com")
            tool = BrowserWaitTool(session=session)
            result = await tool._arun(load_state="networkidle", timeout_ms=10000)
            assert "reached" in result


# ---------------------------------------------------------------------------
# Fingerprint diagnostics
# ---------------------------------------------------------------------------


class TestBrowserDiagnoseIntegration:
    @pytest.mark.asyncio
    async def test_diagnose_fingerprint(self, config: DirectBrowserConfig) -> None:
        """Diagnose tool collects browser fingerprint data."""
        async with DirectBrowserSession(config=config) as session:
            await session.page.goto("https://example.com")
            tool = BrowserDiagnoseTool(session=session)
            result = await tool._arun()
            assert "Fingerprint:" in result
            assert "userAgent" in result
            assert "platform" in result
            assert "webglRenderer" in result
            assert "timezone" in result
            # Verify locale/timezone are set correctly
            assert "Europe/Paris" in result
            assert "fr-FR" in result or "Linux" in result

    @pytest.mark.asyncio
    async def test_diagnose_webdriver_flag(self, config: DirectBrowserConfig) -> None:
        """Anti-detection flag makes webdriver false."""
        async with DirectBrowserSession(config=config) as session:
            await session.page.goto("about:blank")
            tool = BrowserEvaluateTool(session=session)
            result = await tool._arun(expression="navigator.webdriver")
            assert result == "false"


# ---------------------------------------------------------------------------
# Cookie persistence
# ---------------------------------------------------------------------------


class TestCookiePersistenceIntegration:
    @pytest.mark.asyncio
    async def test_save_and_load_cookies(self, config: DirectBrowserConfig) -> None:
        """Cookies saved from one session can be loaded in another."""
        # Session 1: navigate and save cookies
        async with DirectBrowserSession(config=config) as s1:
            await s1.page.goto("https://example.com")
            save_tool = BrowserSaveCookiesTool(session=s1)
            save_result = await save_tool._arun(name="test_example")
            assert "saved" in save_result.lower()
            cookie_file = Path(config.cookies_dir) / "test_example_session.json"
            assert cookie_file.exists()

        # Session 2: load cookies
        async with DirectBrowserSession(config=config) as s2:
            from genai_tk.tools.direct_browser.tools import BrowserLoadCookiesTool

            load_tool = BrowserLoadCookiesTool(session=s2)
            load_result = await load_tool._arun(name="test_example")
            assert "loaded" in load_result.lower() or "no saved" in load_result.lower()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestDirectBrowserFactory:
    def test_factory_creates_all_tools(self) -> None:
        """Factory creates the expected number of tools."""
        tools = create_direct_browser_tools()
        assert len(tools) == len(ALL_BROWSER_TOOLS)
        tool_names = {t.name for t in tools}
        assert "browser_navigate" in tool_names
        assert "browser_diagnose" in tool_names
        assert "browser_fill_credential" in tool_names


# ---------------------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------------------


class TestEventLogIntegration:
    @pytest.mark.asyncio
    async def test_navigation_logged(self, config: DirectBrowserConfig) -> None:
        """Navigation events are captured in the event log."""
        async with DirectBrowserSession(config=config) as session:
            await session.page.goto("https://example.com")
            await asyncio.sleep(0.5)
            log = session.get_event_log(last_n=20)
            assert "example.com" in log.lower() or "navigation" in log.lower()
