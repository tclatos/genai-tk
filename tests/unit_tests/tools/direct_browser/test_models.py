"""Unit tests for direct browser Pydantic models."""

from genai_tk.tools.direct_browser.models import DirectBrowserConfig


class TestDirectBrowserConfig:
    def test_defaults(self) -> None:
        cfg = DirectBrowserConfig()
        assert cfg.locale == "fr-FR"
        assert cfg.timezone_id == "Europe/Paris"
        assert cfg.headless is False
        assert cfg.viewport_width == 1920
        assert cfg.viewport_height == 1080
        assert cfg.default_timeout_ms == 30_000
        assert cfg.slow_type_ms == 60
        assert cfg.cookies_dir == "data/sessions"
        assert cfg.user_data_dir is None
        assert cfg.extra_args == []
        assert cfg.allowed_credential_envs == []

    def test_custom_values(self) -> None:
        cfg = DirectBrowserConfig(
            locale="en-US",
            timezone_id="UTC",
            headless=True,
            viewport_width=1280,
            extra_args=["--no-sandbox"],
            allowed_credential_envs=["MY_USER", "MY_PASS"],
        )
        assert cfg.locale == "en-US"
        assert cfg.headless is True
        assert cfg.viewport_width == 1280
        assert cfg.extra_args == ["--no-sandbox"]
        assert cfg.allowed_credential_envs == ["MY_USER", "MY_PASS"]
