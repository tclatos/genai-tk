"""Authentication handlers for Playwright-backed web scraping.

Each handler implements an ``authenticate`` async method that drives
a Playwright ``Page`` through the login flow for a specific mechanism.

Supported types:
- ``form``           — fill username/password fields and submit
- ``oauth_redirect`` — form auth on an OAuth IdP page after redirect
- ``oauth_popup``    — handle IdP login inside a popup window
- ``storage_state``  — no browser interaction; validates file exists
- ``none``           — no authentication required
- ``custom``         — delegates to a user-supplied async callable

Example:
    ```python
    from genai_tk.tools.browser.auth_handlers import get_auth_handler
    from genai_tk.tools.browser.models import AuthConfig

    handler = get_auth_handler("form")
    await handler.authenticate(page, config, scraper_name="my_site")
    ```
"""

from __future__ import annotations

import asyncio
import random

from loguru import logger

from genai_tk.tools.browser.models import AuthConfig, AuthType

# ---------------------------------------------------------------------------
# Protocol / base
# ---------------------------------------------------------------------------


class AuthHandlerProtocol:
    """Interface that all auth handlers implement."""

    async def authenticate(self, page: object, config: AuthConfig, scraper_name: str) -> None:
        """Drive the page through authentication.

        Args:
            page: Playwright ``Page`` to interact with.
            config: Full authentication configuration.
            scraper_name: Scraper name (used for session-state path resolution).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _random_delay(min_ms: int = 50, max_ms: int = 200) -> None:
    """Sleep for a random interval to simulate human behaviour."""
    await asyncio.sleep(random.randint(min_ms, max_ms) / 1000)


async def _type_human(locator: object, text: str, delay_ms: int = 60) -> None:
    """Type text with per-character delays to mimic a human typist.

    Args:
        locator: Playwright ``Locator`` pointing to the input field.
        text: Text to type.
        delay_ms: Base delay between keystrokes in milliseconds (±20% jitter added).
    """
    for char in text:
        jitter = random.randint(-int(delay_ms * 0.2), int(delay_ms * 0.2))
        await locator.type(char, delay=max(10, delay_ms + jitter))  # type: ignore[attr-defined]


async def _fill_form(page: object, config: AuthConfig) -> None:
    """Fill username and password fields on the current page.

    Resolves credentials from env/file — they are never logged or returned.
    """
    creds = config.credentials
    if creds is None:
        raise ValueError("AuthConfig.credentials must be set for form/oauth auth")

    username = creds.username.resolve()
    password = creds.password.resolve()

    selectors = config.selectors

    await _random_delay(200, 500)

    # Username
    username_loc = page.locator(selectors.username_input).first  # type: ignore[attr-defined]
    await username_loc.wait_for(state="visible")
    await username_loc.click()
    await _random_delay(80, 200)
    await _type_human(username_loc, username)
    await _random_delay(100, 300)

    # Password
    password_loc = page.locator(selectors.password_input).first  # type: ignore[attr-defined]
    await password_loc.wait_for(state="visible")
    await password_loc.click()
    await _random_delay(80, 200)
    await _type_human(password_loc, password)
    await _random_delay(200, 500)

    # Submit
    submit_loc = page.locator(selectors.submit_button).first  # type: ignore[attr-defined]
    await submit_loc.click()


async def _wait_for_success(page: object, config: AuthConfig) -> None:
    """Wait for the post-login page to appear."""
    if config.success_url_pattern:
        await page.wait_for_url(config.success_url_pattern, timeout=30_000)  # type: ignore[attr-defined]
    else:
        await page.wait_for_load_state("networkidle")  # type: ignore[attr-defined]


async def _run_mfa(page: object, config: AuthConfig) -> None:
    """Invoke optional MFA handler if configured."""
    if not config.mfa_handler:
        return
    from genai_tk.utils.config_mngr import import_from_qualified

    handler_fn = import_from_qualified(config.mfa_handler)
    logger.info(f"Running MFA handler: {config.mfa_handler}")
    await handler_fn(page, config)


async def _save_session(page: object, config: AuthConfig, scraper_name: str) -> None:
    """Save the authenticated browser context state to disk."""
    from genai_tk.tools.browser.session_manager import SessionManager

    SessionManager.ensure_parent_dir(config, scraper_name)
    path = SessionManager.state_path(config, scraper_name)
    await page.context.storage_state(path=str(path))  # type: ignore[attr-defined]
    logger.info(f"Session saved to {path}")


# ---------------------------------------------------------------------------
# Form handler
# ---------------------------------------------------------------------------


class FormAuthHandler(AuthHandlerProtocol):
    """Fills a username/password login form and submits it."""

    async def authenticate(self, page: object, config: AuthConfig, scraper_name: str) -> None:
        if not config.login_url:
            raise ValueError("AuthConfig.login_url is required for type=form")

        logger.info(f"[form auth] Navigating to {config.login_url}")
        await page.goto(config.login_url, wait_until="domcontentloaded")  # type: ignore[attr-defined]
        await _random_delay(300, 700)

        await _fill_form(page, config)
        await _wait_for_success(page, config)
        await _run_mfa(page, config)
        await _save_session(page, config, scraper_name)
        logger.info("[form auth] Authentication successful")


# ---------------------------------------------------------------------------
# OAuth redirect handler
# ---------------------------------------------------------------------------


class OAuthRedirectHandler(AuthHandlerProtocol):
    """Handles OAuth login where the app redirects to an IdP login form."""

    async def authenticate(self, page: object, config: AuthConfig, scraper_name: str) -> None:
        if not config.login_url:
            raise ValueError("AuthConfig.login_url is required for type=oauth_redirect")

        logger.info(f"[oauth_redirect] Navigating to {config.login_url}")
        # Navigate to the protected URL; the app will redirect to the IdP
        await page.goto(config.login_url, wait_until="domcontentloaded")  # type: ignore[attr-defined]
        await _random_delay(500, 1000)

        # Fill credentials on the IdP page (same selectors)
        await _fill_form(page, config)
        # After submit the IdP redirects back to the app
        await _wait_for_success(page, config)
        await _run_mfa(page, config)
        await _save_session(page, config, scraper_name)
        logger.info("[oauth_redirect] Authentication successful")


# ---------------------------------------------------------------------------
# OAuth popup handler
# ---------------------------------------------------------------------------


class OAuthPopupHandler(AuthHandlerProtocol):
    """Handles OAuth SSO where a popup window opens for the IdP."""

    async def authenticate(self, page: object, config: AuthConfig, scraper_name: str) -> None:
        if not config.login_url:
            raise ValueError("AuthConfig.login_url is required for type=oauth_popup")

        logger.info(f"[oauth_popup] Navigating to {config.login_url}")
        await page.goto(config.login_url, wait_until="domcontentloaded")  # type: ignore[attr-defined]
        await _random_delay(500, 1000)

        # Click the SSO button — must match the submit_button selector
        selectors = config.selectors
        sso_btn = page.locator(selectors.submit_button).first  # type: ignore[attr-defined]

        # Wait for the popup window
        async with page.expect_popup() as popup_info:  # type: ignore[attr-defined]
            await sso_btn.click()

        popup = await popup_info.value
        await popup.wait_for_load_state("domcontentloaded")
        await _random_delay(500, 1000)

        # Fill credentials inside the popup
        await _fill_form(popup, config)
        # Popup closes after successful login; wait for the parent page to update
        await _wait_for_success(page, config)
        await _run_mfa(page, config)
        await _save_session(page, config, scraper_name)
        logger.info("[oauth_popup] Authentication successful")


# ---------------------------------------------------------------------------
# Storage-state handler
# ---------------------------------------------------------------------------


class StorageStateHandler(AuthHandlerProtocol):
    """No-op handler: relies entirely on a pre-existing storage-state file.

    Use this when the storage state was created by an external script
    (e.g. ``scripts/oauth_auth.py``) and you never want this code to
    attempt automated login.
    """

    async def authenticate(self, page: object, config: AuthConfig, scraper_name: str) -> None:
        from genai_tk.tools.browser.session_manager import SessionManager

        path = SessionManager.state_path(config, scraper_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Storage state file '{path}' not found. "
                "Run the auth capture script first: uv run scripts/oauth_auth.py"
            )
        logger.info(f"[storage_state] Using pre-existing session from {path}")


# ---------------------------------------------------------------------------
# No-auth handler
# ---------------------------------------------------------------------------


class NoAuthHandler(AuthHandlerProtocol):
    """Pass-through handler for pages that require no authentication."""

    async def authenticate(self, page: object, config: AuthConfig, scraper_name: str) -> None:
        logger.debug("[no-auth] Authentication skipped")


# ---------------------------------------------------------------------------
# Custom handler
# ---------------------------------------------------------------------------


class CustomAuthHandler(AuthHandlerProtocol):
    """Delegates to a user-supplied async callable.

    The callable must be a fully-qualified Python name (``'module:function'``)
    and have the signature ``async (page, config: AuthConfig) -> None``.
    """

    async def authenticate(self, page: object, config: AuthConfig, scraper_name: str) -> None:
        if not config.custom_handler:
            raise ValueError("AuthConfig.custom_handler must be set for type=custom")

        from genai_tk.utils.config_mngr import import_from_qualified

        handler_fn = import_from_qualified(config.custom_handler)
        logger.info(f"[custom auth] Delegating to {config.custom_handler}")
        await handler_fn(page, config)
        await _save_session(page, config, scraper_name)
        logger.info("[custom auth] Authentication completed")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_HANDLERS: dict[AuthType, AuthHandlerProtocol] = {
    "form": FormAuthHandler(),
    "oauth_redirect": OAuthRedirectHandler(),
    "oauth_popup": OAuthPopupHandler(),
    "storage_state": StorageStateHandler(),
    "none": NoAuthHandler(),
    "custom": CustomAuthHandler(),
}


def get_auth_handler(auth_type: AuthType) -> AuthHandlerProtocol:
    """Return the authentication handler for the given type.

    Args:
        auth_type: One of the supported ``AuthType`` values.
    """
    handler = _HANDLERS.get(auth_type)
    if handler is None:
        raise ValueError(f"Unknown auth type: {auth_type!r}. Valid: {list(_HANDLERS)}")
    return handler
