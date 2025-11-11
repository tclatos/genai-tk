"""Tracing utilities for LangSmith integration."""

from contextlib import contextmanager
from typing import Generator

from langsmith.utils import tracing_is_enabled

from genai_tk.utils.config_mngr import global_config


def should_enable_tracing() -> bool:
    """Check if LangSmith tracing should be enabled.

    Returns:
        True if both config and API key are available, False otherwise
    """
    langsmith_enabled = global_config().get_bool("monitoring.langsmith", False)
    return langsmith_enabled and tracing_is_enabled()


@contextmanager
def tracing_context() -> Generator[None, None, None]:
    """Context manager for backwards compatibility.

    This doesn't capture trace URLs anymore. Just use it as a context manager
    that yields None. Tracing is controlled by environment variables.
    """
    yield None
