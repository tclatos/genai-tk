"""YAML configuration loader for web scraper definitions.

Loads scraper configs from ``config/basic/web_scrapers/<name>.yaml`` (or a
caller-supplied path) using OmegaConf for variable interpolation, then
validates them into :class:`~genai_tk.tools.browser.models.WebScraperConfig`
Pydantic models.

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

The scraper name is taken from the YAML key (``my_scraper`` above) and
injected into the resulting ``WebScraperConfig.name`` field.

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


def _default_config_dir() -> Path:
    """Return the default web-scrapers config directory."""
    try:
        from genai_tk.utils.config_mngr import global_config

        config_dir = global_config().get_dir_path("paths.config")
        return config_dir / "web_scrapers"
    except Exception:
        # Fallback when config manager is not initialised (e.g. in tests)
        return Path("config/basic/web_scrapers")


def _find_yaml_containing_key(config_dir: Path, name: str) -> Path | None:
    """Search all YAML files in *config_dir* for one that contains *name* under ``web_scrapers:``."""
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        try:
            from omegaconf import OmegaConf  # noqa: PLC0415

            with open(yaml_file) as fh:
                raw = fh.read()
            node = OmegaConf.create(raw)
            cfg = OmegaConf.to_container(node, resolve=False)
            if isinstance(cfg, dict) and name in cfg.get("web_scrapers", {}):
                return yaml_file
        except Exception:
            continue
    return None


def load_web_scraper_config(
    name: str,
    config_path: str | Path | None = None,
) -> WebScraperConfig:
    """Load and validate a web-scraper configuration by name.

    The YAML file is resolved via OmegaConf so ``${...}`` interpolations
    (e.g. ``${paths.project}``) are expanded before Pydantic validation.

    Args:
        name: Key under ``web_scrapers:`` in the YAML file.
        config_path: Path to the YAML file.  When ``None``, looks for
            ``config/basic/web_scrapers/<name>.yaml`` relative to the project root.

    Returns:
        Validated ``WebScraperConfig`` with ``name`` populated.
    """
    from omegaconf import OmegaConf

    # Resolve file path
    if config_path is not None:
        path = Path(config_path)
    else:
        config_dir = _default_config_dir()
        # 1. Try exact match: <name>.yaml
        candidate = config_dir / f"{name}.yaml"
        if candidate.exists():
            path = candidate
        else:
            # 2. Search all YAML files in the directory for a file that contains the key
            path = _find_yaml_containing_key(config_dir, name)
            if path is None:
                raise FileNotFoundError(
                    f"Scraper key '{name}' not found in any YAML file under '{config_dir}'. "
                    f"Add a 'web_scrapers.{name}:' section to a YAML file there."
                )

    if not path.exists():
        raise FileNotFoundError(
            f"Web scraper config not found at '{path}'. Create '{path}' with a 'web_scrapers.{name}:' section."
        )

    logger.debug(f"Loading web scraper config from {path}")

    with open(path) as fh:
        raw_text = fh.read()

    # OmegaConf load + resolve (handles ${paths.project} etc.)
    try:
        cfg_node = OmegaConf.create(raw_text)
        cfg_dict = OmegaConf.to_container(cfg_node, resolve=True)
    except Exception as exc:
        raise ValueError(f"OmegaConf failed to parse '{path}': {exc}") from exc

    if not isinstance(cfg_dict, dict):
        raise ValueError(f"Config file '{path}' must contain a YAML mapping at the top level")

    scrapers: dict = cfg_dict.get("web_scrapers", {})
    if name not in scrapers:
        available = list(scrapers.keys())
        raise KeyError(f"Scraper '{name}' not found in '{path}'. Available keys: {available}")

    scraper_raw: dict = scrapers[name]
    scraper_raw.setdefault("name", name)

    try:
        config = WebScraperConfig.model_validate(scraper_raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid web scraper config '{name}' in '{path}':\n{exc}") from exc

    logger.debug(f"Loaded web scraper '{name}' with {len(config.targets)} target(s)")
    return config
