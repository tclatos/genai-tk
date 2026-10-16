"""Configuration models for AUIP and AG-UI server endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AuipServerSettings(BaseModel):
    """Network and runtime settings for the AUIP / AG-UI server."""

    host: str = "127.0.0.1"
    port: int = 8000
    default_profile: str = "research"
    allowed_profiles: list[str] = Field(default_factory=lambda: ["*"])
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    title: str = "GenAI Toolkit AUIP Gateway"
    endpoint: str = "/agent"


class AuipConfig(BaseModel):
    """Top-level ``auip:`` section configuration."""

    server: AuipServerSettings = Field(default_factory=AuipServerSettings)


def auip_config() -> AuipConfig:
    """Return the global AUIP configuration (from ``auip:`` YAML section)."""
    from genai_tk.config_mgmt.config_mngr import global_config

    return global_config().section("auip", AuipConfig)
