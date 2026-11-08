"""Configuration models for Textual Terminal User Interfaces (TUI)."""

from __future__ import annotations

from pydantic import Field

from genai_tk.config_mgmt.config_model import ConfigModel


class TuiDisplayConfig(ConfigModel):
    """Configuration for Textual TUI applications.

    Supports instantiation directly, from a YAML source, or from global config:
        `TuiDisplayConfig.from_config("ui.display.tui")`
    """

    theme: str = "slate"
    split_ratio_left: int = 48
    split_ratio_right: int = 52
    refresh_interval_ms: int = 500
    max_table_rows: int = 500
    options: dict[str, str] = Field(default_factory=dict)
