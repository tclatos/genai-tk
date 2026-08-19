"""Unit tests for genai_tk.utils.tokens."""

from __future__ import annotations

import pytest

from genai_tk.utils.tokens import count_tokens, get_tiktoken_encoding


class TestGetTiktokenEncoding:
    def test_returns_cached_encoding(self) -> None:
        first = get_tiktoken_encoding()
        second = get_tiktoken_encoding()
        assert first is second  # cached via @once

    def test_supports_alternate_encoding_name(self) -> None:
        encoding = get_tiktoken_encoding("cl100k_base")
        assert encoding.name == "cl100k_base"


class TestCountTokens:
    def test_counts_real_tokens(self) -> None:
        assert count_tokens("Hello world") > 0

    def test_empty_string_is_zero_tokens(self) -> None:
        assert count_tokens("") == 0

    def test_longer_text_has_more_tokens(self) -> None:
        short = count_tokens("Hello")
        long = count_tokens("Hello " * 50)
        assert long > short

    def test_falls_back_to_regex_estimate_when_tiktoken_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(encoding_name: str = "o200k_base"):
            raise RuntimeError("no network access to fetch tiktoken encoding")

        monkeypatch.setattr("genai_tk.utils.tokens.get_tiktoken_encoding", _boom)
        assert count_tokens("Hello, world!") == 4  # "Hello" "," "world" "!"
