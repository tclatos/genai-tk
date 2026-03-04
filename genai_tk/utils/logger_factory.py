"""Logging configuration factory using Loguru.

Provides a centralized way to configure logging across the application
"""

# For more control, see  https://medium.com/python-in-plain-english/mastering-logging-in-python-with-loguru-and-pydantic-settings-a-complete-guide-for-cross-module-a6205a987251

import os
import sys

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal

from genai_tk.utils.config_mngr import global_config


class LoggingConfig(BaseModel):
    """Typed view of the ``logging`` config section."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field("INFO")
    format: str | None = Field(None)
    backtrace: bool = Field(False)
    model_config = ConfigDict(extra="allow")


def logging_config() -> LoggingConfig:
    """Return typed logging settings from config."""
    try:
        raw = global_config().get_dict("logging")
        return LoggingConfig.model_validate(raw)
    except Exception:
        return LoggingConfig()


def setup_logging(level: str | None = None) -> None:
    """Configure the application logger with Loguru.

    Sets up logging with a default format. It can be overridden by setting the LOGURU_FORMAT environment variable.
    """
    LOGURU_FORMAT = "<cyan>{time:HH:mm:ss}</cyan>-<level>{level: <7}</level> | <magenta>{file.name}</magenta>:<green>{line} <italic>{function}</italic></green>- <level>{message}</level>"
    cfg = logging_config()
    format_str = os.environ.get("LOGURU_FORMAT") or cfg.format or LOGURU_FORMAT
    level = level or cfg.level
    backtrace = cfg.backtrace
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        level=level.upper(),
        format=format_str,
        backtrace=backtrace,
        diagnose=True,
    )
