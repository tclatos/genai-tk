"""Unit tests for direct browser LangChain tools."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genai_tk.tools.direct_browser.models import DirectBrowserConfig
from genai_tk.tools.direct_browser.session import DirectBrowserSession
from genai_tk.tools.direct_browser.tools import (
    ALL_BROWSER_TOOLS,
    BrowserClickTool,
    BrowserEvaluateTool,
    BrowserFillCredentialTool,
    BrowserNavigateTool,
    BrowserReadPageTool,
    BrowserScreenshotTool,
    BrowserScrollTool,
    BrowserWaitTool,
)


@pytest.fixture
def mock_session() -> DirectBrowserSession:
    """Create a mock session with a mock page."""
    config = DirectBrowserConfig(allowed_credential_envs=["TEST_USER", "TEST_PASS"])
    session = DirectBrowserSession(config=config)
    mock_page = AsyncMock()
    mock_page.url = "https://example.com/dashboard"
    mock_page.title = AsyncMock(return_value="Dashboard")
    mock_page.inner_text = AsyncMock(return_value="Welcome to dashboard. Data: 42kWh")
    mock_page.goto = AsyncMock()
    mock_page.click = AsyncMock()
    mock_page.fill = AsyncMock()
    mock_page.type = AsyncMock()
    mock_page.screenshot = AsyncMock(return_value=b"\x89PNG\r\n")
    mock_page.query_selector = AsyncMock()
    mock_page.evaluate = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.wait_for_selector = AsyncMock()
    mock_page.wait_for_load_state = AsyncMock()
    session._page = mock_page
    session._connected = True
    return session


class TestToolRegistry:
    def test_all_browser_tools_count(self) -> None:
        assert len(ALL_BROWSER_TOOLS) == 13

    def test_tool_names_match_sandbox_browser(self) -> None:
        expected_names = {
            "browser_navigate",
            "browser_click",
            "browser_type",
            "browser_fill_credential",
            "browser_screenshot",
            "browser_read_page",
            "browser_scroll",
            "browser_wait",
            "browser_save_cookies",
            "browser_load_cookies",
            "browser_get_logs",
            "browser_evaluate",
            "browser_diagnose",
        }
        actual_names = {cls.model_fields["name"].default for cls in ALL_BROWSER_TOOLS}
        assert actual_names == expected_names


class TestBrowserToolConnectionHandling:
    @pytest.mark.asyncio
    async def test_transient_error_does_not_reconnect(self, mock_session: DirectBrowserSession) -> None:
        mock_session.page.evaluate.side_effect = Exception(
            "Execution context was destroyed, most likely because of a navigation",
        )
        mock_session.close = AsyncMock()
        mock_session.connect = AsyncMock()
        tool = BrowserNavigateTool(session=mock_session)
        await tool._ensure_connected()
        mock_session.close.assert_not_called()
        mock_session.connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_closed_page_reconnects(self, mock_session: DirectBrowserSession) -> None:
        mock_session.page.evaluate.side_effect = Exception("Target page has been closed")
        mock_session.page.is_closed.return_value = True
        mock_session.close = AsyncMock()
        mock_session.connect = AsyncMock()
        tool = BrowserNavigateTool(session=mock_session)
        await tool._ensure_connected()
        mock_session.close.assert_awaited_once()
        mock_session.connect.assert_awaited_once()


class TestBrowserNavigateTool:
    @pytest.mark.asyncio
    async def test_navigate_success(self, mock_session: DirectBrowserSession) -> None:
        tool = BrowserNavigateTool(session=mock_session)
        result = await tool._arun("https://example.com")
        mock_session.page.goto.assert_called_once()
        assert "Dashboard" in result
        assert "example.com" in result

    @pytest.mark.asyncio
    async def test_navigate_failure(self, mock_session: DirectBrowserSession) -> None:
        mock_session.page.goto.side_effect = TimeoutError("Timeout")
        tool = BrowserNavigateTool(session=mock_session)
        result = await tool._arun("https://bad-url.com")
        assert "Navigation failed" in result


class TestBrowserClickTool:
    @pytest.mark.asyncio
    async def test_click_success(self, mock_session: DirectBrowserSession) -> None:
        tool = BrowserClickTool(session=mock_session)
        result = await tool._arun("#submit-btn")
        mock_session.page.click.assert_called_once()
        assert "Dashboard" in result


class TestBrowserFillCredentialTool:
    @pytest.mark.asyncio
    async def test_fill_credential_success(self, mock_session: DirectBrowserSession) -> None:
        with patch.dict(os.environ, {"TEST_USER": "user@test.com"}):
            tool = BrowserFillCredentialTool(session=mock_session)
            result = await tool._arun(selector="#email", credential_env="TEST_USER")
            assert "Credential from $TEST_USER filled" in result

    @pytest.mark.asyncio
    async def test_fill_credential_blocked_by_allowlist(self, mock_session: DirectBrowserSession) -> None:
        with patch.dict(os.environ, {"SECRET_KEY": "val"}):
            tool = BrowserFillCredentialTool(session=mock_session)
            result = await tool._arun(selector="#field", credential_env="SECRET_KEY")
            assert "not in the allowlist" in result


class TestBrowserReadPageTool:
    @pytest.mark.asyncio
    async def test_read_page_success(self, mock_session: DirectBrowserSession) -> None:
        tool = BrowserReadPageTool(session=mock_session)
        result = await tool._arun()
        assert "Welcome to dashboard" in result
        assert "Dashboard" in result


class TestBrowserScreenshotTool:
    @pytest.mark.asyncio
    async def test_screenshot_success(self, mock_session: DirectBrowserSession) -> None:
        tool = BrowserScreenshotTool(session=mock_session)
        result = await tool._arun()
        assert result.startswith("data:image/png;base64,")


class TestBrowserScrollTool:
    @pytest.mark.asyncio
    async def test_scroll_down(self, mock_session: DirectBrowserSession) -> None:
        tool = BrowserScrollTool(session=mock_session)
        result = await tool._arun(direction="down", amount=500)
        mock_session.page.evaluate.assert_called()
        assert "Dashboard" in result


class TestBrowserWaitTool:
    @pytest.mark.asyncio
    async def test_wait_for_selector(self, mock_session: DirectBrowserSession) -> None:
        tool = BrowserWaitTool(session=mock_session)
        result = await tool._arun(selector="#content")
        mock_session.page.wait_for_selector.assert_called_once()
        assert "appeared" in result

    @pytest.mark.asyncio
    async def test_wait_for_load_state(self, mock_session: DirectBrowserSession) -> None:
        tool = BrowserWaitTool(session=mock_session)
        result = await tool._arun(load_state="networkidle")
        mock_session.page.wait_for_load_state.assert_called_once()
        assert "reached" in result


class TestBrowserEvaluateTool:
    @pytest.mark.asyncio
    async def test_evaluate_success(self, mock_session: DirectBrowserSession) -> None:
        mock_session.page.evaluate.return_value = "Mozilla/5.0"
        tool = BrowserEvaluateTool(session=mock_session)
        result = await tool._arun(expression="navigator.userAgent")
        assert "Mozilla" in result
