"""YAML configuration loader for web scraper definitions.

Loads scraper configs from a single YAML file or a directory of YAML files
(``config/basic/web_scrapers/``) using OmegaConf for variable interpolation,
then validates them into
:class:`~genai_tk.tools.browser.models.WebScraperConfig` Pydantic models.

YAML top-level structure::

    web_scrapers:
      my_scraper:
        description: "..."
        browser: { ... }
        auth: { ... }
        cookie_consent: { ... }
        targets:
          - name: main_page
            ...

The scraper name is taken from the dict key (``my_scraper``) and injected
into the resulting ``WebScraperConfig.name`` field.

Example:
    ```python
    from genai_tk.tools.browser.config_loader import load_web_scraper_config

    config = load_web_scraper_config("enedis_production")
    print(config.auth.type)  # "form"
    ```
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from genai_tk.tools.browser.models import WebScraperConfig
from genai_tk.utils.config_mngr import load_yaml_configs


def _default_scrapers_dir() -> Path:
    """Return the default web-scrapers config directory."""
    try:
        from genai_tk.utils.config_mngr import global_config

        return global_config().get_dir_path("paths.config") / "web_scrapers"
    except Exception:
        return Path("config/basic/web_scrapers")


def load_web_scraper_config(
    name: str,
    config_path: str | Path | None = None,
) -> WebScraperConfig:
    """Load and validate a web-scraper configuration by name.

    OmegaConf ``${...}`` interpolations (e.g. ``${paths.project}``) are
    expanded before Pydantic validation.  *config_path* may point to a single
    YAML file **or** a directory; in the directory case every ``*.yaml`` /
    ``*.yml`` file is searched for a ``web_scrapers.<name>`` key.

    Args:
        name: Key under ``web_scrapers:`` that identifies the scraper.
        config_path: Path to a YAML file or directory.  Defaults to the
            ``web_scrapers`` sub-directory of ``paths.config``.

    Returns:
        Validated ``WebScraperConfig`` with ``name`` populated.
    """
    if config_path is not None:
        search_path = Path(config_path)
    else:
        config_dir = _default_scrapers_dir()
        # Try exact-match file first; fall back to full directory search
        candidate = config_dir / f"{name}.yaml"
        search_path = candidate if candidate.exists() else config_dir

    scrapers: dict = load_yaml_configs(search_path, "web_scrapers")  # type: ignore[assignment]

    if name not in scrapers:
        available = list(scrapers.keys())
        raise KeyError(f"Scraper '{name}' not found in '{search_path}'. Available: {available}")

    scraper_raw: dict = scrapers[name]
    scraper_raw.setdefault("name", name)

    try:
        config = WebScraperConfig.model_validate(scraper_raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid web scraper config '{name}' in '{search_path}':\n{exc}") from exc

    logger.debug(f"Loaded web scraper '{name}' with {len(config.targets)} target(s)")
    return config
