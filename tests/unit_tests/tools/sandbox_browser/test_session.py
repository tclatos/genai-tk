"""Unit tests for sandbox browser session management."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from genai_tk.tools.sandbox_browser.models import SandboxBrowserConfig
from genai_tk.tools.sandbox_browser.session import SandboxBrowserSession


@pytest.fixture
def config(tmp_path: Path) -> SandboxBrowserConfig:
    return SandboxBrowserConfig(cookies_dir=str(tmp_path / "sessions"))


class TestSandboxBrowserSession:
    def test_initial_state(self, config: SandboxBrowserConfig) -> None:
        session = SandboxBrowserSession(sandbox_url="http://localhost:8080", config=config)
        assert not session.connected
        assert session._sandbox_url == "http://localhost:8080"

    def test_trailing_slash_stripped(self, config: SandboxBrowserConfig) -> None:
        session = SandboxBrowserSession(sandbox_url="http://localhost:8080/", config=config)
        assert session._sandbox_url == "http://localhost:8080"

    def test_page_raises_when_not_connected(self, config: SandboxBrowserConfig) -> None:
        session = SandboxBrowserSession(config=config)
        with pytest.raises(RuntimeError, match="not connected"):
            _ = session.page


class TestCookiePersistence:
    @pytest.mark.asyncio
    async def test_save_cookies(self, config: SandboxBrowserConfig) -> None:
        session = SandboxBrowserSession(config=config)
        mock_context = AsyncMock()
        mock_context.storage_state.return_value = {"cookies": [{"name": "sid", "value": "abc"}]}
        session._context = mock_context

        path = await session.save_cookies("test_site")
        assert Path(path).exists()
        saved = json.loads(Path(path).read_text())
        assert saved["cookies"][0]["name"] == "sid"

    @pytest.mark.asyncio
    async def test_load_cookies_missing_file(self, config: SandboxBrowserConfig) -> None:
        session = SandboxBrowserSession(config=config)
        session._browser = MagicMock()
        result = await session.load_cookies("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_load_cookies_success(self, config: SandboxBrowserConfig, tmp_path: Path) -> None:
        # Create a cookie file
        cookies_dir = Path(config.cookies_dir)
        cookies_dir.mkdir(parents=True, exist_ok=True)
        cookie_file = cookies_dir / "mysite_session.json"
        state = {"cookies": [{"name": "token", "value": "xyz"}], "origins": []}
        cookie_file.write_text(json.dumps(state))

        session = SandboxBrowserSession(config=config)
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        temp_page = AsyncMock()
        mock_context.new_page.return_value = temp_page
        session._browser = AsyncMock()
        session._context = mock_context
        session._page = mock_page

        result = await session.load_cookies("mysite")
        assert result is True
        mock_context.clear_cookies.assert_called_once()
        mock_context.add_cookies.assert_called_once_with(state["cookies"])
        mock_context.new_page.assert_not_called()
        temp_page.close.assert_not_called()
        assert session._page is mock_page

    @pytest.mark.asyncio
    async def test_load_cookies_raises_without_browser(self, config: SandboxBrowserConfig) -> None:
        cookies_dir = Path(config.cookies_dir)
        cookies_dir.mkdir(parents=True, exist_ok=True)
        cookie_file = cookies_dir / "test_session.json"
        cookie_file.write_text('{"cookies": []}')

        session = SandboxBrowserSession(config=config)
        with pytest.raises(RuntimeError, match="not connected"):
            await session.load_cookies("test")
