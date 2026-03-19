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
_COOKIE_ACCEPT_SELECTORS: tuple[str, ...] = (
    "#popin_tc_privacy_button_3",
    "button:has-text('Autoriser tous les cookies')",
    "button:has-text('Accepter tous les cookies')",
    "button:has-text('Tout accepter')",
)

_CAPTCHA_START_SELECTORS: tuple[str, ...] = (
    "#captcha-widget",
    ".frc-captcha",
    "text='Clique ici pour vérifier'",
)

_CAPTCHA_SOLUTION_SELECTOR = "input[name='frc-captcha-solution']"
_LOGIN_ERROR_SELECTORS: tuple[str, ...] = (
    "[role='alert']",
    ".fr-input-message",
    ".error",
    ".alert",
    ".frc-banner",
)


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


async def _locator_is_visible(locator: object) -> bool:
    """Return ``True`` when the locator is currently visible."""
    try:
        return bool(await locator.is_visible())  # type: ignore[attr-defined]
    except Exception:
        return False


async def _click_first_visible(page: object, selectors: tuple[str, ...], timeout_ms: int = 1_500) -> bool:
    """Try clicking the first visible element among selectors."""
    for selector in selectors:
        try:
            loc = page.locator(selector)  # type: ignore[attr-defined]
            count = await loc.count()
        except Exception:
            continue
        for idx in range(count):
            item = loc.nth(idx)
            if not await _locator_is_visible(item):
                continue
            try:
                await item.click(timeout=timeout_ms)  # type: ignore[attr-defined]
                return True
            except Exception:
                continue
    return False


async def _captcha_solution_value(page: object) -> str | None:
    """Return FriendlyCaptcha hidden solution marker/value when present."""
    try:
        loc = page.locator(_CAPTCHA_SOLUTION_SELECTOR)  # type: ignore[attr-defined]
        if await loc.count() == 0:
            return None
        return await loc.first.get_attribute("value")
    except Exception:
        return None

def _captcha_pending(value: str | None) -> bool:
    """Return ``True`` when FriendlyCaptcha indicates unresolved challenge."""
    return value is not None and value.startswith(".")



async def _prepare_challenge_if_needed(page: object) -> None:
    """Dismiss known overlays and trigger anti-robot verification when present."""
    await _click_first_visible(page, _COOKIE_ACCEPT_SELECTORS)
    captcha_value = await _captcha_solution_value(page)
    if _captcha_pending(captcha_value):
        await _click_first_visible(page, _CAPTCHA_START_SELECTORS)


async def _wait_for_captcha_ready(page: object, timeout_ms: int = 120_000) -> None:
    """Wait for FriendlyCaptcha pending state to clear when captcha is present."""
    value = await _captcha_solution_value(page)
    if not _captcha_pending(value):
        return

    logger.info(f"[form auth] Waiting for anti-robot challenge (captcha state={value!r})")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + (timeout_ms / 1000)
    last_value = value
    while loop.time() < deadline:
        await _prepare_challenge_if_needed(page)
        value = await _captcha_solution_value(page)
        if value != last_value:
            logger.info(f"[form auth] captcha state: {value!r}")
            last_value = value
        if not _captcha_pending(value):
            return
        await _random_delay(220, 420)

    if value == ".HEADLESS_ERROR":
        raise TimeoutError(
            "Anti-robot challenge failed in headless mode (captcha=.HEADLESS_ERROR). "
            "Retry with --no-headless and complete the challenge in the browser window."
        )
    raise TimeoutError(
        f"Anti-robot challenge not completed within {timeout_ms}ms (captcha={value!r}). "
        "Complete the captcha in the browser window and retry."
    )

async def _locator_is_enabled(locator: object) -> bool:
    """Return ``True`` when the locator appears enabled for interaction."""
    try:
        if await locator.is_disabled():  # type: ignore[attr-defined]
            return False
    except Exception:
        # Some custom elements may not support is_disabled; fall back to attributes
        pass

    try:
        aria_disabled = (await locator.get_attribute("aria-disabled") or "").strip().lower()  # type: ignore[attr-defined]
        if aria_disabled in {"true", "1"}:
            return False
    except Exception:
        pass

    try:
        disabled_attr = await locator.get_attribute("disabled")  # type: ignore[attr-defined]
        if disabled_attr is not None:
            return False
    except Exception:
        pass

    return True


async def _wait_for_enabled(locator: object, timeout_ms: int = 12_000) -> None:
    """Wait until a locator is enabled or raise ``TimeoutError``."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + (timeout_ms / 1000)
    page = getattr(locator, "page", None)
    observed_captcha_value: str | None = None
    attempts = 0
    while True:
        if await _locator_is_enabled(locator):
            return
        attempts += 1
        if page is not None and (attempts == 1 or attempts % 5 == 0):
            await _prepare_challenge_if_needed(page)
            captcha_value = await _captcha_solution_value(page)
            if captcha_value != observed_captcha_value:
                logger.info(f"[form auth] captcha state: {captcha_value!r}")
                observed_captcha_value = captcha_value
        if loop.time() >= deadline:
            if observed_captcha_value is not None:
                raise TimeoutError(
                    f"Submit control stayed disabled for {timeout_ms}ms "
                    f"(captcha={observed_captcha_value!r}). Complete anti-robot check and retry."
                )
            raise TimeoutError(f"Submit control stayed disabled for {timeout_ms}ms")
        await _random_delay(120, 260)

async def _collect_visible_error_hints(page: object) -> list[str]:
    """Return a small list of visible error/alert text snippets on the page."""
    hints: list[str] = []
    for selector in _LOGIN_ERROR_SELECTORS:
        try:
            loc = page.locator(selector)  # type: ignore[attr-defined]
            count = min(await loc.count(), 5)
        except Exception:
            continue
        for idx in range(count):
            item = loc.nth(idx)
            if not await _locator_is_visible(item):
                continue
            text = ""
            try:
                text = (await item.inner_text()).strip()  # type: ignore[attr-defined]
            except Exception:
                try:
                    text = (await item.text_content() or "").strip()  # type: ignore[attr-defined]
                except Exception:
                    text = ""
            if not text:
                continue
            compact = " ".join(text.split())
            if compact not in hints:
                hints.append(compact)
            if len(hints) >= 4:
                return hints
    return hints


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
    await _prepare_challenge_if_needed(page)

    await _random_delay(200, 500)

    # Username
    username_loc = page.locator(selectors.username_input).first  # type: ignore[attr-defined]
    await username_loc.wait_for(state="visible")
    try:
        await username_loc.click(timeout=3_000)
    except Exception:
        logger.debug("[form auth] Username click failed; continuing with direct typing/fill")
    await _random_delay(80, 200)
    try:
        await _type_human(username_loc, username)
    except Exception:
        await username_loc.fill(username)  # type: ignore[attr-defined]
    await _random_delay(100, 300)
    # Some IdPs use a two-step form: username first, password after an
    # intermediate submit/continue action.
    password_loc = page.locator(selectors.password_input).first  # type: ignore[attr-defined]
    if not await _locator_is_visible(password_loc):
        logger.info("[form auth] Password field not visible yet — submitting username step first")
        await _wait_for_captcha_ready(page, timeout_ms=120_000)
        username_submit_loc = page.locator(selectors.submit_button).first  # type: ignore[attr-defined]
        await username_submit_loc.wait_for(state="visible")
        await _wait_for_enabled(username_submit_loc, timeout_ms=60_000)
        await username_submit_loc.click()
        await _random_delay(350, 800)

    # Password
    await password_loc.wait_for(state="visible")
    try:
        await password_loc.click(timeout=3_000)
    except Exception:
        logger.debug("[form auth] Password click failed; continuing with direct typing/fill")
    await _random_delay(80, 200)
    try:
        await _type_human(password_loc, password)
    except Exception:
        await password_loc.fill(password)  # type: ignore[attr-defined]
    await _random_delay(200, 500)
    # Some IdP forms only enable submit after blur/validation.
    try:
        await password_loc.press("Tab")
    except Exception:
        pass
    await _prepare_challenge_if_needed(page)
    await _wait_for_captcha_ready(page, timeout_ms=120_000)
    await _random_delay(120, 260)

    # Submit
    submit_loc = page.locator(selectors.submit_button).first  # type: ignore[attr-defined]
    await submit_loc.wait_for(state="visible")
    await _wait_for_enabled(submit_loc, timeout_ms=90_000)
    await submit_loc.click()


async def _wait_for_success(page: object, config: AuthConfig) -> None:
    """Wait for the post-login page to appear."""
    if not config.success_url_pattern:
        await page.wait_for_load_state("networkidle")  # type: ignore[attr-defined]
        return

    try:
        await page.wait_for_url(config.success_url_pattern, timeout=30_000)  # type: ignore[attr-defined]
        return
    except Exception as exc:
        if "Timeout" not in str(exc):
            raise
        current_url = str(getattr(page, "url", "<unknown>"))
        logger.warning(
            "[auth] success_url_pattern not reached within timeout "
            f"({config.success_url_pattern!r}); current_url={current_url!r}. "
            "Falling back to login-form visibility check."
        )

    await page.wait_for_load_state("networkidle")  # type: ignore[attr-defined]
    await _random_delay(200, 500)

    selectors = config.selectors
    username_visible = await _locator_is_visible(page.locator(selectors.username_input).first)  # type: ignore[attr-defined]
    password_visible = await _locator_is_visible(page.locator(selectors.password_input).first)  # type: ignore[attr-defined]

    if not username_visible and not password_visible:
        logger.info(
            "[auth] Login form fields are no longer visible after submit; "
            "treating authentication as successful despite URL pattern mismatch."
        )
        return

    current_url = str(getattr(page, "url", "<unknown>"))
    captcha_value = await _captcha_solution_value(page)
    hints = await _collect_visible_error_hints(page)
    details: list[str] = []
    if captcha_value is not None:
        details.append(f"captcha={captcha_value!r}")
    if hints:
        details.append(f"page_hints={hints!r}")
    details_msg = f" ({'; '.join(details)})" if details else ""
    raise TimeoutError(
        "Timeout waiting for authenticated page: expected URL pattern "
        f"{config.success_url_pattern!r} and login form is still visible at {current_url!r}{details_msg}"
    )


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
