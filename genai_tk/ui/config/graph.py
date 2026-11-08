"""Configuration models for Knowledge Graph visualization (HTML and Streamlit)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from genai_tk.config_mgmt.config_model import ConfigModel


class GraphDisplayConfig(ConfigModel):
    """Configuration for Knowledge Graph visualization (D3 and DAG).

    Supports instantiation directly, from a YAML source, or from global config:
        `GraphDisplayConfig.from_config("ui.display.graph")`
    """

    top_node_type: str = "Opportunity"
    default_exclude_node_types: list[str] = Field(default_factory=lambda: ["Image", "MarkdownSection"])
    max_siblings: int = 10
    target_sibling_types: list[str] = Field(default_factory=lambda: ["MarkdownSection", "SectionChunk"])
    default_layout: Literal["dag", "force"] = "dag"
    dag_orientation: Literal["TB", "LR"] = "TB"
    default_limit: int = 2000
    filter_orphan_nodes: bool = True
    custom_colors: dict[str, str] = Field(default_factory=dict)
