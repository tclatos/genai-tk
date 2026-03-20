"""Unit tests for sandbox browser LangChain tools."""

import os
from unittest.mock import AsyncMock, patch

import pytest

from genai_tk.tools.sandbox_browser.models import SandboxBrowserConfig
from genai_tk.tools.sandbox_browser.session import SandboxBrowserSession
from genai_tk.tools.sandbox_browser.tools import (
    BrowserClickTool,
    BrowserFillCredentialTool,
    BrowserLoadCookiesTool,
    BrowserNavigateTool,
    BrowserReadPageTool,
    BrowserSaveCookiesTool,
    BrowserScreenshotTool,
    BrowserScrollTool,
    BrowserTypeTool,
    BrowserWaitTool,
)


@pytest.fixture
def mock_session() -> SandboxBrowserSession:
    """Create a mock session with a mock page."""
    config = SandboxBrowserConfig(allowed_credential_envs=["TEST_USER", "TEST_PASS"])
    session = SandboxBrowserSession(config=config)
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
    mock_page.wait_for_selector = AsyncMock()
    mock_page.wait_for_load_state = AsyncMock()
    session._page = mock_page
    session._connected = True
    return session


class TestBrowserNavigateTool:
    @pytest.mark.asyncio
    async def test_navigate_success(self, mock_session: SandboxBrowserSession) -> None:
        tool = BrowserNavigateTool(session=mock_session)
        result = await tool._arun("https://example.com")
        mock_session.page.goto.assert_called_once()
        assert "Dashboard" in result
        assert "example.com" in result

    @pytest.mark.asyncio
    async def test_navigate_failure(self, mock_session: SandboxBrowserSession) -> None:
        mock_session.page.goto.side_effect = TimeoutError("Timeout")
        tool = BrowserNavigateTool(session=mock_session)
        result = await tool._arun("https://bad-url.com")
        assert "Navigation failed" in result


class TestBrowserClickTool:
    @pytest.mark.asyncio
    async def test_click_success(self, mock_session: SandboxBrowserSession) -> None:
        tool = BrowserClickTool(session=mock_session)
        result = await tool._arun("#submit-btn")
        mock_session.page.click.assert_called_once()
        assert "Dashboard" in result

    @pytest.mark.asyncio
    async def test_click_failure(self, mock_session: SandboxBrowserSession) -> None:
        mock_session.page.click.side_effect = Exception("Element not found")
        tool = BrowserClickTool(session=mock_session)
        result = await tool._arun("#nonexistent")
        assert "Click failed" in result


class TestBrowserTypeTool:
    @pytest.mark.asyncio
    async def test_type_success(self, mock_session: SandboxBrowserSession) -> None:
        tool = BrowserTypeTool(session=mock_session)
        result = await tool._arun(selector="#search", text="hello")
        assert "successfully" in result

    @pytest.mark.asyncio
    async def test_type_failure(self, mock_session: SandboxBrowserSession) -> None:
        mock_session.page.click.side_effect = Exception("Not found")
        tool = BrowserTypeTool(session=mock_session)
        result = await tool._arun(selector="#missing", text="hello")
        assert "Typing failed" in result


class TestBrowserFillCredentialTool:
    @pytest.mark.asyncio
    async def test_fill_credential_success(self, mock_session: SandboxBrowserSession) -> None:
        with patch.dict(os.environ, {"TEST_USER": "myuser@example.com"}):
            tool = BrowserFillCredentialTool(session=mock_session)
            result = await tool._arun(selector="#email", credential_env="TEST_USER")
            assert "Credential from $TEST_USER filled" in result
            # The actual value should NOT appear in the result
            assert "myuser@example.com" not in result

    @pytest.mark.asyncio
    async def test_fill_credential_blocked_by_allowlist(self, mock_session: SandboxBrowserSession) -> None:
        with patch.dict(os.environ, {"SECRET_KEY": "super_secret"}):
            tool = BrowserFillCredentialTool(session=mock_session)
            result = await tool._arun(selector="#field", credential_env="SECRET_KEY")
            assert "Credential error" in result
            assert "not in the allowlist" in result
            # Value should NEVER appear
            assert "super_secret" not in result

    @pytest.mark.asyncio
    async def test_fill_credential_missing_env(self, mock_session: SandboxBrowserSession) -> None:
        tool = BrowserFillCredentialTool(session=mock_session)
        result = await tool._arun(selector="#field", credential_env="TEST_USER")
        assert "Credential error" in result


class TestBrowserScreenshotTool:
    @pytest.mark.asyncio
    async def test_screenshot(self, mock_session: SandboxBrowserSession) -> None:
        tool = BrowserScreenshotTool(session=mock_session)
        result = await tool._arun()
        assert result.startswith("data:image/png;base64,")


class TestBrowserReadPageTool:
    @pytest.mark.asyncio
    async def test_read_full_page(self, mock_session: SandboxBrowserSession) -> None:
        tool = BrowserReadPageTool(session=mock_session)
        result = await tool._arun()
        assert "42kWh" in result

    @pytest.mark.asyncio
    async def test_read_with_selector(self, mock_session: SandboxBrowserSession) -> None:
        mock_element = AsyncMock()
        mock_element.inner_text = AsyncMock(return_value="Production: 42kWh")
        mock_session.page.query_selector.return_value = mock_element
        tool = BrowserReadPageTool(session=mock_session)
        result = await tool._arun(selector=".production-data")
        assert "42kWh" in result

    @pytest.mark.asyncio
    async def test_read_selector_not_found(self, mock_session: SandboxBrowserSession) -> None:
        mock_session.page.query_selector.return_value = None
        tool = BrowserReadPageTool(session=mock_session)
        result = await tool._arun(selector="#missing")
        assert "not found" in result


class TestBrowserScrollTool:
    @pytest.mark.asyncio
    async def test_scroll_down(self, mock_session: SandboxBrowserSession) -> None:
        tool = BrowserScrollTool(session=mock_session)
        result = await tool._arun(direction="down", amount=500)
        mock_session.page.evaluate.assert_called_once_with("window.scrollBy(0, 500)")
        assert "Dashboard" in result

    @pytest.mark.asyncio
    async def test_scroll_up(self, mock_session: SandboxBrowserSession) -> None:
        tool = BrowserScrollTool(session=mock_session)
        await tool._arun(direction="up", amount=300)
        mock_session.page.evaluate.assert_called_once_with("window.scrollBy(0, -300)")


class TestBrowserWaitTool:
    @pytest.mark.asyncio
    async def test_wait_for_selector(self, mock_session: SandboxBrowserSession) -> None:
        tool = BrowserWaitTool(session=mock_session)
        result = await tool._arun(selector="#data-table", timeout_ms=5000)
        assert "appeared" in result

    @pytest.mark.asyncio
    async def test_wait_timeout(self, mock_session: SandboxBrowserSession) -> None:
        mock_session.page.wait_for_selector.side_effect = TimeoutError("timeout")
        tool = BrowserWaitTool(session=mock_session)
        result = await tool._arun(selector="#missing", timeout_ms=1000)
        assert "Timeout" in result


class TestBrowserSaveCookiesTool:
    @pytest.mark.asyncio
    async def test_save_cookies(self, mock_session: SandboxBrowserSession) -> None:
        mock_session.save_cookies = AsyncMock(return_value="/tmp/test_session.json")
        tool = BrowserSaveCookiesTool(session=mock_session)
        result = await tool._arun(name="test")
        assert "saved" in result.lower()


class TestBrowserLoadCookiesTool:
    @pytest.mark.asyncio
    async def test_load_cookies_success(self, mock_session: SandboxBrowserSession) -> None:
        mock_session.load_cookies = AsyncMock(return_value=True)
        tool = BrowserLoadCookiesTool(session=mock_session)
        result = await tool._arun(name="test")
        assert "loaded" in result.lower()

    @pytest.mark.asyncio
    async def test_load_cookies_not_found(self, mock_session: SandboxBrowserSession) -> None:
        mock_session.load_cookies = AsyncMock(return_value=False)
        tool = BrowserLoadCookiesTool(session=mock_session)
        result = await tool._arun(name="nonexistent")
        assert "No saved session" in result


class TestCredentialSecurity:
    """Verify credentials never leak into tool outputs."""

    @pytest.mark.asyncio
    async def test_credential_value_never_in_output(self, mock_session: SandboxBrowserSession) -> None:
        secret = "my_super_secret_password_123!"
        with patch.dict(os.environ, {"TEST_PASS": secret}):
            tool = BrowserFillCredentialTool(session=mock_session)
            result = await tool._arun(selector="#password", credential_env="TEST_PASS")
            assert secret not in result

    @pytest.mark.asyncio
    async def test_credential_env_name_in_output(self, mock_session: SandboxBrowserSession) -> None:
        with patch.dict(os.environ, {"TEST_PASS": "secret"}):
            tool = BrowserFillCredentialTool(session=mock_session)
            result = await tool._arun(selector="#password", credential_env="TEST_PASS")
            assert "TEST_PASS" in result
