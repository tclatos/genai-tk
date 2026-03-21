"""Unit tests for direct browser session management."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from genai_tk.tools.direct_browser.models import DirectBrowserConfig
from genai_tk.tools.direct_browser.session import DirectBrowserSession


@pytest.fixture
def config(tmp_path: Path) -> DirectBrowserConfig:
    return DirectBrowserConfig(cookies_dir=str(tmp_path / "sessions"))


class TestDirectBrowserSession:
    def test_initial_state(self, config: DirectBrowserConfig) -> None:
        session = DirectBrowserSession(config=config)
        assert not session.connected
        assert session._browser is None

    def test_page_raises_when_not_connected(self, config: DirectBrowserConfig) -> None:
        session = DirectBrowserSession(config=config)
        with pytest.raises(RuntimeError, match="not connected"):
            _ = session.page

    def test_event_log_empty_initially(self, config: DirectBrowserConfig) -> None:
        session = DirectBrowserSession(config=config)
        assert session.get_event_log() == "(no browser events recorded)"


class TestCookiePersistence:
    @pytest.mark.asyncio
    async def test_save_cookies(self, config: DirectBrowserConfig) -> None:
        session = DirectBrowserSession(config=config)
        mock_context = AsyncMock()
        mock_context.storage_state.return_value = {"cookies": [{"name": "sid", "value": "abc"}]}
        session._context = mock_context

        path = await session.save_cookies("test_site")
        assert Path(path).exists()
        saved = json.loads(Path(path).read_text())
        assert saved["cookies"][0]["name"] == "sid"

    @pytest.mark.asyncio
    async def test_load_cookies_missing_file(self, config: DirectBrowserConfig) -> None:
        session = DirectBrowserSession(config=config)
        result = await session.load_cookies("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_load_cookies_success(self, config: DirectBrowserConfig) -> None:
        cookies_dir = Path(config.cookies_dir)
        cookies_dir.mkdir(parents=True, exist_ok=True)
        cookie_file = cookies_dir / "mysite_session.json"
        state = {"cookies": [{"name": "token", "value": "xyz"}], "origins": []}
        cookie_file.write_text(json.dumps(state))

        session = DirectBrowserSession(config=config)
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        session._context = mock_context
        session._page = mock_page

        result = await session.load_cookies("mysite")
        assert result is True
        mock_context.clear_cookies.assert_called_once()
        mock_context.add_cookies.assert_called_once_with(state["cookies"])

    @pytest.mark.asyncio
    async def test_load_cookies_raises_without_context(self, config: DirectBrowserConfig) -> None:
        cookies_dir = Path(config.cookies_dir)
        cookies_dir.mkdir(parents=True, exist_ok=True)
        cookie_file = cookies_dir / "test_session.json"
        cookie_file.write_text('{"cookies": []}')

        session = DirectBrowserSession(config=config)
        with pytest.raises(RuntimeError, match="missing active context"):
            await session.load_cookies("test")
