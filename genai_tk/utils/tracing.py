"""Tracing utilities for LangSmith integration."""

import os
from contextlib import contextmanager
from typing import Any, Generator

from genai_tk.utils.config_mngr import global_config


class NoOpTraceCallback:
    """No-op callback for when tracing is disabled."""

    def get_run_url(self) -> None:
        """Return None when tracing is disabled."""
        return None


@contextmanager
def tracing_context() -> Generator[Any, None, None]:
    """Context manager that conditionally enables LangSmith tracing.

    Enables tracing if:
    1. monitoring.langsmith config is True
    2. LANGCHAIN_API_KEY environment variable is set

    Otherwise returns a no-op callback that returns None for get_run_url().

    Yields:
        Either a LangSmith tracing callback or a no-op callback
    """
    # Check if LangSmith is configured and API key is available
    langsmith_enabled = global_config().get_bool("monitoring.langsmith", False)
    api_key_available = bool(os.getenv("LANGCHAIN_API_KEY"))

    if langsmith_enabled and api_key_available:
        try:
            from langchain_core.callbacks import tracing_v2_enabled

            with tracing_v2_enabled() as cb:
                yield cb
        except ImportError:
            # If langchain tracing is not available, fall back to no-op
            yield NoOpTraceCallback()
    else:
        # Return no-op callback when tracing is disabled
        yield NoOpTraceCallback()
