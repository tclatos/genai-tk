"""Unit tests for genai_tk.tools.browser.models."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from genai_tk.tools.browser.models import (
    AuthConfig,
    BrowserConfig,
    CookieConsentConfig,
    CredentialRef,
    ExtractConfig,
    SessionConfig,
    TargetConfig,
    ViewportConfig,
    WebScraperConfig,
)

# ---------------------------------------------------------------------------
# CredentialRef
# ---------------------------------------------------------------------------


class TestCredentialRef:
    def test_env_source_resolves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_CRED_VAR", "secret123")
        ref = CredentialRef(env="TEST_CRED_VAR")
        assert ref.resolve() == "secret123"

    def test_env_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NO_SUCH_VAR_XYZ", raising=False)
        ref = CredentialRef(env="NO_SUCH_VAR_XYZ")
        with pytest.raises(ValueError, match="NO_SUCH_VAR_XYZ"):
            ref.resolve()

    def test_file_source_resolves(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("filepassword\n")
            fname = f.name
        try:
            ref = CredentialRef(file=fname)
            assert ref.resolve() == "filepassword"
        finally:
            os.unlink(fname)

    def test_file_missing_raises(self) -> None:
        ref = CredentialRef(file="/nonexistent/secret.txt")
        with pytest.raises(ValueError, match="does not exist"):
            ref.resolve()

    def test_both_sources_raises(self) -> None:
        with pytest.raises(Exception):
            CredentialRef(env="A", file="/some/file")

    def test_no_source_raises(self) -> None:
        with pytest.raises(Exception):
            CredentialRef()

    def test_repr_hides_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SECRET_KEY", "topsecret")
        ref = CredentialRef(env="SECRET_KEY")
        r = repr(ref)
        assert "topsecret" not in r
        assert "SECRET_KEY" in r


# ---------------------------------------------------------------------------
# SessionConfig
# ---------------------------------------------------------------------------


class TestSessionConfig:
    def test_resolve_path_substitutes_name(self) -> None:
        cfg = SessionConfig(storage_state_path="data/sessions/{name}_session.json")
        resolved = cfg.resolve_path("my_scraper")
        assert resolved == Path("data/sessions/my_scraper_session.json")

    def test_resolve_path_no_placeholder(self) -> None:
        cfg = SessionConfig(storage_state_path="data/sessions/fixed.json")
        resolved = cfg.resolve_path("anything")
        assert resolved == Path("data/sessions/fixed.json")

    def test_defaults(self) -> None:
        cfg = SessionConfig()
        assert cfg.check_validity is True
        assert "{name}" in cfg.storage_state_path


# ---------------------------------------------------------------------------
# BrowserConfig
# ---------------------------------------------------------------------------


class TestBrowserConfig:
    def test_defaults(self) -> None:
        bc = BrowserConfig()
        assert bc.headless is True
        assert bc.user_agent == "rotating"
        assert bc.locale == "fr-FR"
        assert bc.viewport_jitter is True

    def test_viewport_defaults(self) -> None:
        bc = BrowserConfig()
        assert bc.viewport.width == 1920
        assert bc.viewport.height == 1080

    def test_custom_viewport(self) -> None:
        bc = BrowserConfig(viewport=ViewportConfig(width=1280, height=720))
        assert bc.viewport.width == 1280


# ---------------------------------------------------------------------------
# CookieConsentConfig
# ---------------------------------------------------------------------------


class TestCookieConsentConfig:
    def test_defaults(self) -> None:
        cfg = CookieConsentConfig()
        assert cfg.enabled is True
        assert cfg.strategy == "auto"
        assert cfg.selectors == []

    def test_custom_strategy(self) -> None:
        cfg = CookieConsentConfig(
            strategy="selectors",
            selectors=["#accept-btn", ".accept-all"],
        )
        assert len(cfg.selectors) == 2


# ---------------------------------------------------------------------------
# ExtractConfig
# ---------------------------------------------------------------------------


class TestExtractConfig:
    def test_text_default(self) -> None:
        ec = ExtractConfig()
        assert ec.type == "text"
        assert ec.selector is None

    def test_screenshot_type(self) -> None:
        ec = ExtractConfig(type="screenshot")
        assert ec.type == "screenshot"


# ---------------------------------------------------------------------------
# TargetConfig
# ---------------------------------------------------------------------------


class TestTargetConfig:
    def test_required_fields(self) -> None:
        t = TargetConfig(name="main", url="https://example.com/data")
        assert t.name == "main"
        assert t.wait_for == "networkidle"
        assert t.extract.type == "text"


# ---------------------------------------------------------------------------
# WebScraperConfig
# ---------------------------------------------------------------------------


class TestWebScraperConfig:
    def _make_config(self) -> WebScraperConfig:
        return WebScraperConfig(
            name="test_scraper",
            description="Test",
            targets=[
                TargetConfig(name="page_a", url="https://example.com/a"),
                TargetConfig(name="page_b", url="https://example.com/b"),
            ],
        )

    def test_get_target_found(self) -> None:
        cfg = self._make_config()
        t = cfg.get_target("page_a")
        assert t.url == "https://example.com/a"

    def test_get_target_not_found(self) -> None:
        cfg = self._make_config()
        with pytest.raises(ValueError, match="page_c"):
            cfg.get_target("page_c")

    def test_auth_none_type(self) -> None:
        cfg = WebScraperConfig(
            name="open_site",
            auth=AuthConfig(type="none"),
        )
        assert cfg.auth.type == "none"
