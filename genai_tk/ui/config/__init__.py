"""Typed Pydantic configuration models for UI components and applications."""

from __future__ import annotations

from genai_tk.ui.config.chat import ChatDisplayConfig
from genai_tk.ui.config.graph import GraphDisplayConfig
from genai_tk.ui.config.tui import TuiDisplayConfig
from genai_tk.ui.config.webapp import WebappConfig

__all__ = [
    "ChatDisplayConfig",
    "GraphDisplayConfig",
    "TuiDisplayConfig",
    "WebappConfig",
]
