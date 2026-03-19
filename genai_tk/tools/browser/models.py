"""Pydantic configuration models for the authenticated web scraper.

All credential references are resolved at runtime from environment variables
or files.  Credentials are never stored in plain text in the model, never
logged, and never sent to an LLM.

YAML structure (under ``web_scrapers.<name>``)::

    web_scrapers:
      my_site:
        description: "My protected site scraper"
        browser:
          headless: true
          user_agent: rotating
          viewport: {width: 1920, height: 1080}
          timeout_ms: 30000
          locale: "fr-FR"
          slow_mo_ms: 80
        auth:
          type: form
          login_url: "https://example.com/login"
          credentials:
            username: {env: MY_SITE_USER}
            password: {env: MY_SITE_PASS}
          selectors:
            username_input: "input[name=email]"
            password_input: "input[type=password]"
            submit_button:  "button[type=submit]"
          success_url_pattern: "**/dashboard**"
          session:
            storage_state_path: "data/sessions/{name}_session.json"
            check_validity: true
        cookie_consent:
          enabled: true
          strategy: auto
          timeout_ms: 6000
        targets:
          - name: main_page
            description: "Extract text from the main data page"
            url: "https://example.com/data"
            wait_for: networkidle
            wait_for_selector: ".data-table"
            extract:
              type: text
              selector: null
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

# ---------------------------------------------------------------------------
# Credential reference — resolves from env or file, never stored plain
# ---------------------------------------------------------------------------

CredentialSource = Literal["env", "file"]


class CredentialRef(BaseModel):
    """A credential value resolved at use-time from an env var or file.

    Usage examples (YAML):
    ```yaml
    username: {env: MY_APP_USER}
    password: {env: MY_APP_PASS}
    # Or from a file:
    token: {file: /run/secrets/my_token}
    ```
    """

    env: str | None = None
    file: str | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _exactly_one_source(self) -> CredentialRef:
        if (self.env is None) == (self.file is None):
            raise ValueError("Exactly one of 'env' or 'file' must be set in a CredentialRef")
        return self

    def resolve(self) -> str:
        """Return the credential value, raising ``ValueError`` if missing or empty."""
        if self.env is not None:
            value = os.environ.get(self.env)
            if not value:
                raise ValueError(
                    f"Credential env var '{self.env}' is not set or empty. "
                    f"Export it before running: export {self.env}=<value>"
                )
            return value
        # file source
        assert self.file is not None
        path = Path(self.file)
        if not path.exists():
            raise ValueError(f"Credential file '{self.file}' does not exist")
        value = path.read_text().strip()
        if not value:
            raise ValueError(f"Credential file '{self.file}' is empty")
        return value

    def __repr__(self) -> str:
        if self.env:
            return f"CredentialRef(env={self.env!r})"
        return f"CredentialRef(file={self.file!r})"


# ---------------------------------------------------------------------------
# Authentication selectors
# ---------------------------------------------------------------------------


class AuthSelectors(BaseModel):
    """CSS selectors used to locate form fields during authentication.

    Defaults cover the most common patterns; override in YAML when needed.
    """

    username_input: str = Field(
        'input[type="email"], input[name="email"], input[name="username"], input[name="login"]',
        description="CSS selector for the username/email input",
    )
    password_input: str = Field(
        'input[type="password"]',
        description="CSS selector for the password input",
    )
    submit_button: str = Field(
        'button[type="submit"], input[type="submit"], button:has-text("Se connecter"), button:has-text("Connexion"), button:has-text("Login"), button:has-text("Sign in")',
        description="CSS selector for the submit/login button",
    )

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


class SessionConfig(BaseModel):
    """Controls Playwright storage-state persistence between runs.

    ```yaml
    session:
      storage_state_path: "data/sessions/{name}_session.json"
      check_validity: true
    ```
    The ``{name}`` placeholder in ``storage_state_path`` is replaced with the
    scraper's config name at load time.
    """

    storage_state_path: str = Field(
        "data/sessions/{name}_session.json",
        description="Path to Playwright storage-state JSON; {name} is replaced with the scraper name",
    )
    check_validity: bool = Field(
        True,
        description="When true, skip re-auth if the saved cookies are still valid",
    )

    model_config = ConfigDict(extra="forbid")

    def resolve_path(self, name: str) -> Path:
        """Return the storage-state path with {name} substituted."""
        return Path(self.storage_state_path.replace("{name}", name))


# ---------------------------------------------------------------------------
# Authentication configuration
# ---------------------------------------------------------------------------

AuthType = Literal["form", "oauth_redirect", "oauth_popup", "storage_state", "none", "custom"]


class AuthCredentials(BaseModel):
    """Username/password pair for form or basic OAuth login."""

    username: CredentialRef
    password: CredentialRef

    model_config = ConfigDict(extra="forbid")


class AuthConfig(BaseModel):
    """Full authentication configuration for a scraper.

    ``type`` selects the handler:

    - ``form`` — fills username/password fields and submits
    - ``oauth_redirect`` — follows OAuth redirect to IdP, fills credentials there
    - ``oauth_popup`` — handles IdP login in a browser popup window
    - ``storage_state`` — assumes a pre-existing storage state file; no browser interaction
    - ``none`` — page requires no authentication
    - ``custom`` — delegates to a fully-qualified Python callable
    """

    type: AuthType = Field("form", description="Authentication mechanism to use")
    login_url: str | None = Field(None, description="URL to navigate to for authentication")
    credentials: AuthCredentials | None = Field(None, description="Username/password credentials")
    selectors: AuthSelectors = Field(default_factory=AuthSelectors, description="CSS selectors for form fields")
    success_url_pattern: str | None = Field(
        None,
        description="URL glob pattern to wait for after successful login, e.g. '**/dashboard**'",
    )
    session: SessionConfig = Field(default_factory=SessionConfig, description="Session persistence settings")
    custom_handler: str | None = Field(
        None,
        description="Qualified callable 'module:function' for type=custom. Signature: async (page, config) -> None",
    )
    mfa_handler: str | None = Field(
        None,
        description="Optional qualified callable for MFA step. Signature: async (page, config) -> None",
    )

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Browser configuration
# ---------------------------------------------------------------------------

UserAgentMode = Annotated[str, StringConstraints(min_length=1)]


class ViewportConfig(BaseModel):
    """Browser viewport dimensions."""

    width: int = 1920
    height: int = 1080

    model_config = ConfigDict(extra="forbid")


class BrowserConfig(BaseModel):
    """Playwright browser launch and context configuration.

    ``user_agent`` can be:
    - ``"rotating"`` — cycles through the built-in UA pool (per-process round-robin)
    - ``"random"``   — picks a random UA from the pool on each scrape call
    - any other string — used verbatim as the UA

    Setting ``viewport_jitter: true`` adds ±30 px noise to width/height to
    reduce fingerprint consistency.
    """

    headless: bool = Field(True, description="Run browser in headless mode")
    user_agent: UserAgentMode = Field("rotating", description="UA mode: rotating, random, or a literal UA string")
    viewport: ViewportConfig = Field(default_factory=ViewportConfig, description="Browser viewport size")
    viewport_jitter: bool = Field(True, description="Add ±30px random noise to viewport dimensions")
    timeout_ms: int = Field(30_000, description="Default navigation and element timeout in milliseconds")
    locale: str = Field("fr-FR", description="Browser locale, affects Accept-Language header")
    slow_mo_ms: int = Field(
        60,
        description="Slow-motion delay between Playwright operations (simulates human typing speed)",
    )
    java_script_enabled: bool = Field(True, description="Enable JavaScript (disable only for static pages)")

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Cookie consent
# ---------------------------------------------------------------------------

CookieConsentStrategy = Literal["auto", "selectors", "custom", "skip"]


class CookieConsentConfig(BaseModel):
    """Configuration for automatically dismissing cookie consent banners.

    ``auto`` strategy tries known French/GDPR banner implementations in order:
    Didomi, Axeptio, Tarteaucitron, OneTrust, and generic accept-all buttons.

    ``selectors`` uses the provided CSS selector list directly.

    ``custom`` delegates to a fully-qualified callable.
    """

    enabled: bool = Field(True, description="Whether to attempt cookie consent dismissal")
    strategy: CookieConsentStrategy = Field("auto", description="Strategy: auto, selectors, custom, or skip")
    selectors: list[str] = Field(
        default_factory=list,
        description="CSS selectors to click for strategy=selectors (tried in order)",
    )
    timeout_ms: int = Field(6_000, description="Max time to wait for a consent banner to appear")
    custom_handler: str | None = Field(
        None,
        description="Qualified callable for strategy=custom. Signature: async (page, config) -> None",
    )

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Extraction configuration
# ---------------------------------------------------------------------------

ExtractType = Literal["text", "screenshot", "dom", "custom"]


class ExtractConfig(BaseModel):
    """Controls how content is extracted from the page after navigation.

    - ``text``       — returns ``page.inner_text(selector or 'body')``
    - ``screenshot`` — returns base64-encoded PNG (for multimodal LLMs)
    - ``dom``        — returns ``page.inner_html(selector or 'body')``
    - ``custom``     — calls a qualified Python function; receives ``(page, config)``
    """

    type: ExtractType = Field("text", description="Extraction method")
    selector: str | None = Field(None, description="CSS selector to scope extraction (null = full body)")
    custom_extractor: str | None = Field(
        None,
        description="Qualified callable for type=custom. Signature: async (page, config) -> str",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("custom_extractor")
    @classmethod
    def _custom_requires_extractor(cls, v: str | None, info: object) -> str | None:
        return v


# ---------------------------------------------------------------------------
# Target (page to scrape)
# ---------------------------------------------------------------------------

WaitForOption = Literal["load", "domcontentloaded", "networkidle", "commit"]


class TargetConfig(BaseModel):
    """A single page target within a scraper configuration.

    A scraper can have multiple targets (e.g. daily view, monthly view).
    """

    name: str = Field(..., description="Unique target identifier within the scraper")
    description: str = Field("", description="Human-readable description passed to LangChain tool description")
    url: str = Field(..., description="URL to navigate to after authentication")
    wait_for: WaitForOption = Field("networkidle", description="Playwright wait_until option for page navigation")
    wait_for_selector: str | None = Field(
        None,
        description="Optional CSS selector to wait for before extraction (e.g. '.data-table')",
    )
    wait_for_selector_timeout_ms: int = Field(
        15_000,
        description="Timeout for wait_for_selector in milliseconds",
    )
    extract: ExtractConfig = Field(default_factory=ExtractConfig, description="Extraction configuration")

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Top-level scraper configuration
# ---------------------------------------------------------------------------


class WebScraperConfig(BaseModel):
    """Complete configuration for a single named web scraper.

    Loaded from ``config/basic/web_scrapers/<name>.yaml`` under the
    ``web_scrapers.<name>`` key.
    """

    name: str = Field(..., description="Scraper name (set programmatically from the YAML key)")
    description: str = Field("", description="Human-readable description for LangChain tool docs")
    browser: BrowserConfig = Field(default_factory=BrowserConfig, description="Browser and context settings")
    auth: AuthConfig = Field(default_factory=AuthConfig, description="Authentication configuration")
    cookie_consent: CookieConsentConfig = Field(
        default_factory=CookieConsentConfig, description="Cookie consent handling"
    )
    targets: list[TargetConfig] = Field(default_factory=list, description="Pages to scrape after authentication")

    model_config = ConfigDict(extra="forbid")

    def get_target(self, name: str) -> TargetConfig:
        """Return the target with the given name.

        Args:
            name: Target name to look up.
        """
        for t in self.targets:
            if t.name == name:
                return t
        available = [t.name for t in self.targets]
        raise ValueError(f"Target '{name}' not found in scraper '{self.name}'. Available: {available}")
