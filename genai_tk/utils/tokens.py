"""Shared tiktoken-based token counting.

Centralizes tiktoken encoding lookup so callers across the codebase (RAG
chunking, document summarization, etc.) share one cached encoding instance
instead of each re-implementing `tiktoken.get_encoding()` + caching.
"""

from __future__ import annotations

import re

import tiktoken
from loguru import logger

from genai_tk.utils.singleton import once

DEFAULT_ENCODING = "o200k_base"


@once
def get_tiktoken_encoding(encoding_name: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    """Return a cached tiktoken encoding, loaded once per encoding name.

    Example:
        ```python
        encoding = get_tiktoken_encoding()
        ```
    """
    return tiktoken.get_encoding(encoding_name)


def _estimate_token_count(text: str) -> int:
    """Rough token-count estimate (word + punctuation split) — no tokenizer dependency."""
    return len(re.findall(r"\w+|[^\w\s]", text))


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    """Count tokens in text, preferring a real tiktoken encoding.

    Falls back to a regex-based estimate (word + punctuation split) when tiktoken
    has no local encoding cache and no network access to fetch one, so this never
    fails outright on an offline machine.

    Example:
        ```python
        n = count_tokens("Hello world")
        ```
    """
    try:
        return len(get_tiktoken_encoding(encoding_name).encode(text))
    except Exception as exc:  # noqa: BLE001
        logger.debug("tiktoken unavailable, falling back to regex token estimate: {}", exc)
        return _estimate_token_count(text)
