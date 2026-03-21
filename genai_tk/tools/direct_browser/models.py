"""Pydantic configuration models for the direct (host-local) browser tools.

Reuses ``CredentialRef`` and ``PageSummary`` from
``genai_tk.tools.sandbox_browser.models`` to maintain a single source of
truth.  Defines ``DirectBrowserConfig`` for host-local Playwright settings.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DirectBrowserConfig(BaseModel):
    """Configuration for the host-local Playwright browser.

    Loaded from the ``direct_browser`` key in ``config/basic/sandbox.yaml``.
    """

    locale: str = "fr-FR"
    timezone_id: str = "Europe/Paris"
    headless: bool = False
    viewport_width: int = 1920
    viewport_height: int = 1080
    default_timeout_ms: int = 30_000
    slow_type_ms: int = 60
    ignore_https_errors: bool = True
    log_browser_console: bool = True
    cookies_dir: str = "data/sessions"
    user_data_dir: str | None = None
    extra_args: list[str] = Field(default_factory=list)
    allowed_credential_envs: list[str] = Field(default_factory=list)
