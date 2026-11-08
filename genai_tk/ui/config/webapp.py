"""Configuration models for the Web application shell and navigation."""

from __future__ import annotations

from pydantic import Field

from genai_tk.config_mgmt.config_model import ConfigModel


class WebappConfig(ConfigModel):
    """Configuration for the main Streamlit webapp shell and page navigation.

    Supports instantiation directly, from a YAML source, or from global config:
        `WebappConfig.from_config("ui")`
    """

    app_name: str | None = None
    logo: str | None = None
    pages_dir: str | None = None
    navigation: dict[str, list[str | dict[str, str]]] = Field(default_factory=dict)
