"""Unit tests for sandbox browser Pydantic models."""

import os
from unittest.mock import patch

import pytest

from genai_tk.tools.sandbox_browser.models import CredentialRef, PageSummary, SandboxBrowserConfig


class TestSandboxBrowserConfig:
    def test_defaults(self) -> None:
        cfg = SandboxBrowserConfig()
        assert cfg.locale == "fr-FR"
        assert cfg.viewport_width == 1920
        assert cfg.viewport_height == 1080
        assert cfg.default_timeout_ms == 30_000
        assert cfg.slow_type_ms == 60
        assert cfg.cookies_dir == "data/sessions"
        assert cfg.allowed_credential_envs == []

    def test_custom_values(self) -> None:
        cfg = SandboxBrowserConfig(
            locale="en-US",
            viewport_width=1280,
            allowed_credential_envs=["MY_USER", "MY_PASS"],
        )
        assert cfg.locale == "en-US"
        assert cfg.viewport_width == 1280
        assert cfg.allowed_credential_envs == ["MY_USER", "MY_PASS"]


class TestCredentialRef:
    def test_resolve_from_env(self) -> None:
        with patch.dict(os.environ, {"TEST_CRED": "secret123"}):
            ref = CredentialRef(env="TEST_CRED")
            assert ref.resolve() == "secret123"

    def test_resolve_missing_env_raises(self) -> None:
        ref = CredentialRef(env="NONEXISTENT_CRED_XYZ")
        with pytest.raises(ValueError, match="not set or empty"):
            ref.resolve()

    def test_resolve_empty_env_raises(self) -> None:
        with patch.dict(os.environ, {"EMPTY_CRED": ""}):
            ref = CredentialRef(env="EMPTY_CRED")
            with pytest.raises(ValueError, match="not set or empty"):
                ref.resolve()

    def test_allowlist_permits(self) -> None:
        with patch.dict(os.environ, {"ALLOWED_CRED": "val"}):
            ref = CredentialRef(env="ALLOWED_CRED")
            assert ref.resolve(allowlist=["ALLOWED_CRED", "OTHER"]) == "val"

    def test_allowlist_blocks(self) -> None:
        with patch.dict(os.environ, {"BLOCKED_CRED": "val"}):
            ref = CredentialRef(env="BLOCKED_CRED")
            with pytest.raises(PermissionError, match="not in the allowlist"):
                ref.resolve(allowlist=["OTHER_CRED"])

    def test_empty_allowlist_permits_all(self) -> None:
        with patch.dict(os.environ, {"ANY_CRED": "val"}):
            ref = CredentialRef(env="ANY_CRED")
            assert ref.resolve(allowlist=[]) == "val"

    def test_none_allowlist_permits_all(self) -> None:
        with patch.dict(os.environ, {"ANY_CRED": "val"}):
            ref = CredentialRef(env="ANY_CRED")
            assert ref.resolve(allowlist=None) == "val"


class TestPageSummary:
    def test_truncation(self) -> None:
        long_text = "x" * 5000
        summary = PageSummary(url="http://test.com", title="Test", text_snippet=long_text)
        assert len(summary.text_snippet) < 5000
        assert summary.text_snippet.endswith("…[truncated]")

    def test_short_text_not_truncated(self) -> None:
        summary = PageSummary(url="http://test.com", title="Test", text_snippet="Hello")
        assert summary.text_snippet == "Hello"

    def test_empty_defaults(self) -> None:
        summary = PageSummary()
        assert summary.url == ""
        assert summary.title == ""
        assert summary.text_snippet == ""
