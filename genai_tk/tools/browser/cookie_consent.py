"""Automatic cookie-consent banner dismissal for Playwright pages.

Tries a series of well-known GDPR/ePrivacy consent framework selectors
in priority order, catching ``TimeoutError`` for each so a missing banner
never blocks the scraper.

Supported banner implementations (``auto`` strategy):
- Didomi (common on French news/e-commerce sites)
- Axeptio (common on French SMB sites)
- Tarteaucitron (widespread in French public/enterprise sites)
- OneTrust (enterprise global)
- RGPD.io / CookieBot / generic patterns
- Fallback: any button containing "Accepter tout" / "Accept all" / "Tout accepter"

Example:
    ```python
    from playwright.async_api import Page
    from genai_tk.tools.browser.cookie_consent import CookieConsentHandler
    from genai_tk.tools.browser.models import CookieConsentConfig

    config = CookieConsentConfig(enabled=True, strategy="auto")
    await CookieConsentHandler.handle(page, config)
    ```
"""

from __future__ import annotations

from loguru import logger

from genai_tk.tools.browser.models import CookieConsentConfig

# ---------------------------------------------------------------------------
# Auto strategy: ordered list of (banner_name, css_selector) to try
# ---------------------------------------------------------------------------

_AUTO_SELECTORS: list[tuple[str, str]] = [
    # Didomi
    ("Didomi", "#didomi-notice-agree-button"),
    ("Didomi", 'button[id="didomi-notice-agree-button"]'),
    # Axeptio
    ("Axeptio", ".axeptio_btn_acceptAll"),
    ("Axeptio", "#axeptio_btn_acceptAll"),
    # Tarteaucitron
    ("Tarteaucitron", "#tarteaucitronAllAllowed"),
    ("Tarteaucitron", ".tarteaucitronAllow"),
    ("Tarteaucitron", 'button:has-text("Autoriser tous les cookies")'),
    # OneTrust
    ("OneTrust", "#onetrust-accept-btn-handler"),
    ("OneTrust", ".onetrust-accept-btn-handler"),
    # CookieBot
    ("CookieBot", 'button[id="CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll"]'),
    # RGPD.io / generic EU patterns
    ("RGPD.io", 'button[data-testid="allow-all-cookies"]'),
    # Enedis / TC Privacy wrapper
    ("TCPrivacy", "#popin_tc_privacy_button_3"),
    ("TCPrivacy", '#popin_tc_privacy_button_2'),
    ("TCPrivacy", 'button:has-text("Autoriser tous les cookies")'),
    # Generic French (most likely on gov / energy sites)
    ("Generic-fr", 'button:has-text("Tout accepter")'),
    ("Generic-fr", 'button:has-text("Accepter tout")'),
    ("Generic-fr", 'button:has-text("Accepter les cookies")'),
    ("Generic-fr", 'button:has-text("Accepter tous les cookies")'),
    ("Generic-fr", 'a:has-text("Tout accepter")'),
    # Generic English fallbacks
    ("Generic-en", 'button:has-text("Accept all")'),
    ("Generic-en", 'button:has-text("Accept all cookies")'),
    ("Generic-en", 'button:has-text("I agree")'),
    ("Generic-en", 'button:has-text("Allow all")'),
]


class CookieConsentHandler:
    """Handles cookie/privacy consent banners before content is scraped."""

    @staticmethod
    async def handle(page: object, config: CookieConsentConfig) -> None:
        """Dismiss a cookie consent banner according to the configured strategy.

        Never raises — if no banner is found or dismissal fails, scraping
        continues (the page may still work without accepting cookies).

        Args:
            page: Playwright ``Page`` object.
            config: Cookie consent configuration.
        """
        if not config.enabled or config.strategy == "skip":
            return
        page_url = str(getattr(page, "url", "") or "")
        if page_url.startswith("about:blank"):
            logger.debug("Skipping cookie consent on about:blank page")
            return

        if config.strategy == "custom":
            await CookieConsentHandler._run_custom(page, config)
        elif config.strategy == "selectors":
            await CookieConsentHandler._try_selectors(page, config.selectors, config.timeout_ms)
        else:
            # auto
            await CookieConsentHandler._try_auto(page, config.timeout_ms)

    @staticmethod
    async def _try_auto(page: object, timeout_ms: int) -> None:
        """Attempt each selector in the auto list, stopping at first success."""
        for banner_name, selector in _AUTO_SELECTORS:
            try:
                locator = page.locator(selector)  # type: ignore[attr-defined]
                if await locator.count() == 0:
                    continue
                first = locator.first
                if not await first.is_visible():
                    continue
                if await first.is_disabled():
                    continue
                await first.click(timeout=min(timeout_ms, 1_000))
                logger.info(f"Cookie consent dismissed via {banner_name} selector: {selector!r}")
                return
            except Exception:
                continue
        logger.debug("No cookie consent banner found (tried all auto selectors)")

    @staticmethod
    async def _try_selectors(page: object, selectors: list[str], timeout_ms: int) -> None:
        """Try caller-provided selectors in order."""
        for selector in selectors:
            try:
                locator = page.locator(selector)  # type: ignore[attr-defined]
                if await locator.count() == 0:
                    continue
                first = locator.first
                if not await first.is_visible():
                    continue
                if await first.is_disabled():
                    continue
                await first.click(timeout=min(timeout_ms, 1_000))
                logger.info(f"Cookie consent dismissed via custom selector: {selector!r}")
                return
            except Exception:
                continue
        logger.debug("Custom consent selectors: none matched")

    @staticmethod
    async def _run_custom(page: object, config: CookieConsentConfig) -> None:
        """Invoke a user-supplied async handler."""
        if not config.custom_handler:
            logger.warning("Cookie consent strategy=custom but no custom_handler set — skipping")
            return
        try:
            from genai_tk.utils.config_mngr import import_from_qualified

            handler = import_from_qualified(config.custom_handler)
            await handler(page, config)
            logger.info(f"Cookie consent handled by custom callable: {config.custom_handler}")
        except Exception as exc:
            logger.warning(f"Custom cookie consent handler raised: {exc} — continuing")
