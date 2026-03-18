"""Unit tests for genai_tk.utils.logger_factory."""

import pytest

from genai_tk.utils.logger_factory import LoggingConfig, logging_config, setup_logging


class TestLoggingConfig:
    def test_default_values(self) -> None:
        cfg = LoggingConfig()
        assert cfg.level == "INFO"
        assert cfg.format is None
        assert cfg.backtrace is False

    def test_custom_level(self) -> None:
        cfg = LoggingConfig(level="DEBUG")
        assert cfg.level == "DEBUG"

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(Exception):
            LoggingConfig(level="INVALID")  # type: ignore

    def test_extra_fields_allowed(self) -> None:
        cfg = LoggingConfig(level="WARNING", extra_field="value")
        assert cfg.level == "WARNING"


class TestLoggingConfigFunction:
    def test_returns_logging_config(self) -> None:
        result = logging_config()
        assert isinstance(result, LoggingConfig)

    def test_returns_default_on_missing_config(self) -> None:

        # Config key logging may not exist - should return default
        result = logging_config()
        assert result.level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class TestSetupLogging:
    def test_setup_logging_default(self) -> None:
        # Should not raise
        setup_logging()

    def test_setup_logging_with_level(self) -> None:
        # Should not raise
        setup_logging("DEBUG")

    def test_setup_logging_with_trace(self) -> None:
        # TRACE is supported by loguru
        setup_logging("TRACE")

    def test_setup_logging_with_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        custom_format = "{time} {level} {message}"
        monkeypatch.setenv("LOGURU_FORMAT", custom_format)
        # Should pick up the environment variable
        setup_logging()
