"""Pydantic configuration models for the sandbox browser tools.

Defines settings for browser session management, anti-bot mitigations,
and the credential allowlist used by ``browser_fill_credential``.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field, model_validator


class SandboxBrowserConfig(BaseModel):
    """Top-level configuration for sandbox browser tools.

    Loaded from the ``sandbox_browser`` key in ``config/basic/sandbox.yaml``.
    """

    locale: str = "fr-FR"
    viewport_width: int = 1920
    viewport_height: int = 1080
    default_timeout_ms: int = 30_000
    slow_type_ms: int = 60
    anti_bot_js: bool = True
    ignore_https_errors: bool = True
    log_browser_console: bool = True
    cookies_dir: str = "data/sessions"
    allowed_credential_envs: list[str] = Field(default_factory=list)


class CredentialRef(BaseModel):
    """Resolve a credential value from an environment variable at use-time.

    The LLM never sees the actual value — only the env var name.
    An optional allowlist restricts which env vars are accessible.
    """

    env: str

    def resolve(self, allowlist: list[str] | None = None) -> str:
        """Return the credential value.

        Args:
            allowlist: If non-empty, only env vars in this list are permitted.
        """
        if allowlist and self.env not in allowlist:
            raise PermissionError(f"Credential env var '{self.env}' is not in the allowlist")
        value = os.environ.get(self.env)
        if not value:
            raise ValueError(f"Environment variable '{self.env}' is not set or empty")
        return value


class PageSummary(BaseModel):
    """Summarised state of the current page returned by browser tools."""

    url: str = ""
    title: str = ""
    text_snippet: str = ""

    @model_validator(mode="after")
    def _truncate_snippet(self) -> PageSummary:
        if len(self.text_snippet) > 4000:
            self.text_snippet = self.text_snippet[:4000] + "\n…[truncated]"
        return self
