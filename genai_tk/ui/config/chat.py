"""Configuration models for Chat, Workbench, and LLM demo interfaces."""

from __future__ import annotations

from pydantic import Field

from genai_tk.config_mgmt.config_model import ConfigModel


class ChatDisplayConfig(ConfigModel):
    """Configuration for chat interfaces, execution trace panels, and workbench layouts.

    Supports instantiation directly, from a YAML source, or from global config:
        `ChatDisplayConfig.from_config("ui.display.chat")`
    """

    panel_height: int = 650
    chat_height: int = 600
    trace_height: int = 650
    show_trace_panel: bool = True
    show_monitoring: bool = True
    show_artifacts: bool = True
    default_mode: str = "pro"
    kind_order: list[str] = Field(default_factory=lambda: ["React", "DeepAgent", "Custom", "DeerFlow"])
