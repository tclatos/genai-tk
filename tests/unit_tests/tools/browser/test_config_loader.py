"""Unit tests for genai_tk.tools.browser.config_loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture()
def scraper_dir(tmp_path: Path) -> Path:
    """Create a temporary web_scrapers directory with two YAML files."""
    d = tmp_path / "web_scrapers"
    d.mkdir()

    # File with a key that does NOT match the filename
    (d / "site_a.yaml").write_text(
        yaml.dump(
            {
                "web_scrapers": {
                    "my_scraper": {
                        "description": "Test scraper",
                        "browser": {"headless": True},
                        "auth": {"type": "none"},
                        "cookie_consent": {"enabled": False},
                        "targets": [{"name": "home", "url": "https://example.com"}],
                    }
                }
            }
        )
    )

    # Second file with a different key
    (d / "site_b.yaml").write_text(
        yaml.dump(
            {
                "web_scrapers": {
                    "other_scraper": {
                        "description": "Other scraper",
                        "browser": {"headless": True},
                        "auth": {"type": "none"},
                        "cookie_consent": {"enabled": False},
                        "targets": [{"name": "page", "url": "https://other.example.com"}],
                    }
                }
            }
        )
    )
    return d


def test_load_by_exact_filename_stem(scraper_dir: Path) -> None:
    """When config_path is supplied, name is used as the lookup key within that file."""
    from genai_tk.tools.browser.config_loader import load_web_scraper_config

    cfg = load_web_scraper_config("my_scraper", config_path=scraper_dir / "site_a.yaml")
    assert cfg.name == "my_scraper"
    assert cfg.auth.type == "none"


def test_load_by_key_found_in_different_file(scraper_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Key 'my_scraper' lives in site_a.yaml, not my_scraper.yaml — loader must find it."""
    from genai_tk.tools.browser import config_loader

    monkeypatch.setattr(config_loader, "_default_config_dir", lambda: scraper_dir)

    cfg = config_loader.load_web_scraper_config("my_scraper")
    assert cfg.name == "my_scraper"
    assert len(cfg.targets) == 1
    assert cfg.targets[0].name == "home"


def test_load_second_file_key(scraper_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Key 'other_scraper' lives in site_b.yaml — loader scans all files."""
    from genai_tk.tools.browser import config_loader

    monkeypatch.setattr(config_loader, "_default_config_dir", lambda: scraper_dir)

    cfg = config_loader.load_web_scraper_config("other_scraper")
    assert cfg.name == "other_scraper"
    assert cfg.targets[0].url == "https://other.example.com"


def test_load_missing_key_raises(scraper_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A key that does not exist in any file raises FileNotFoundError."""
    from genai_tk.tools.browser import config_loader

    monkeypatch.setattr(config_loader, "_default_config_dir", lambda: scraper_dir)

    with pytest.raises(FileNotFoundError, match="not found in any YAML"):
        config_loader.load_web_scraper_config("nonexistent_scraper")


def test_load_with_explicit_config_path(scraper_dir: Path) -> None:
    """Passing config_path bypasses directory scanning."""
    from genai_tk.tools.browser.config_loader import load_web_scraper_config

    cfg = load_web_scraper_config("other_scraper", config_path=scraper_dir / "site_b.yaml")
    assert cfg.name == "other_scraper"
