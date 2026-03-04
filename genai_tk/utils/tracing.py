"""Tracing utilities for LangSmith integration."""

from contextlib import contextmanager
from typing import Generator

from langsmith.utils import tracing_is_enabled
from pydantic import BaseModel, ConfigDict, Field

from genai_tk.utils.config_mngr import global_config


class MonitoringConfig(BaseModel):
    """Typed view of the ``monitoring`` config section."""

    langsmith: bool = Field(False)
    project: str | None = Field(None)
    model_config = ConfigDict(extra="allow")


def monitoring_config() -> MonitoringConfig:
    """Return typed monitoring settings from config."""
    try:
        raw = global_config().get_dict("monitoring")
        return MonitoringConfig.model_validate(raw)
    except Exception:
        return MonitoringConfig()


def should_enable_tracing() -> bool:
    """Check if LangSmith tracing should be enabled.

    Returns:
        True if both config and API key are available, False otherwise
    """
    langsmith_enabled = monitoring_config().langsmith
    return langsmith_enabled and tracing_is_enabled()


@contextmanager
def tracing_context() -> Generator[None, None, None]:
    """Context manager for backwards compatibility.

    This doesn't capture trace URLs anymore. Just use it as a context manager
    that yields None. Tracing is controlled by environment variables.
    """
    yield None
