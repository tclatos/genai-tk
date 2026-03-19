"""Playwright scraper session — authentication, consent, navigation, extraction.

``ScraperSession`` is an async context manager that launches a Playwright
browser, applies all configured anti-bot mitigations, authenticates (or
reuses a cached session), handles cookie consent, then exposes a ``scrape``
method that navigates to a target page and extracts content.

Anti-bot mitigations applied:
- Removes ``navigator.webdriver`` fingerprint via ``add_init_script``
- Randomises Chrome CDP header to avoid headless detection
- Applies locale matching to ``Accept-Language``
- Optional viewport jitter (±30 px)
- Human-like slow typing via ``AuthHandler``
- Configurable ``slow_mo``

Example:
    ```python
    import asyncio
    from genai_tk.tools.browser.scraper_session import run_scraper
    from genai_tk.tools.browser.config_loader import load_web_scraper_config

    config = load_web_scraper_config("enedis_production")
    result = asyncio.run(run_scraper(config, target_name="production_daily"))
    print(result[:500])
    ```
"""

from __future__ import annotations

import base64
import random

from loguru import logger

from genai_tk.tools.browser.auth_handlers import get_auth_handler
from genai_tk.tools.browser.cookie_consent import CookieConsentHandler
from genai_tk.tools.browser.models import BrowserConfig, ExtractConfig, TargetConfig, WebScraperConfig
from genai_tk.tools.browser.session_manager import SessionManager
from genai_tk.tools.browser.user_agents import get_user_agent

# ---------------------------------------------------------------------------
# Anti-bot init script injected into every page before any JS runs
# ---------------------------------------------------------------------------

_ANTI_BOT_SCRIPT = """
// Remove webdriver flag — the most common headless-detection check
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
// Ensure plugins array is not empty (headless Chrome has 0 plugins)
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3]});
// Spoof languages to match locale setting
Object.defineProperty(navigator, 'languages', {get: () => ['fr-FR', 'fr', 'en-US', 'en']});
// Chrome runtime must exist
window.chrome = {runtime: {}};
"""


def _apply_viewport_jitter(base_width: int, base_height: int) -> tuple[int, int]:
    """Add ±30 px random noise to viewport dimensions."""
    return (
        base_width + random.randint(-30, 30),
        base_height + random.randint(-30, 30),
    )


# ---------------------------------------------------------------------------
# ScraperSession
# ---------------------------------------------------------------------------


class ScraperSession:
    """Async context manager that manages a full browser session lifecycle.

    Args:
        config: The web scraper configuration.
        scraper_name: The config key name (used for session-state path).
    """

    def __init__(self, config: WebScraperConfig) -> None:
        self._config = config
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    async def __aenter__(self) -> ScraperSession:
        try:
            from playwright.async_api import async_playwright  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("playwright is required: uv sync --group browser-control") from exc

        bc: BrowserConfig = self._config.browser
        ua = get_user_agent(bc.user_agent)

        if bc.viewport_jitter:
            width, height = _apply_viewport_jitter(bc.viewport.width, bc.viewport.height)
        else:
            width, height = bc.viewport.width, bc.viewport.height

        self._playwright = await async_playwright().start()

        # Determine if we can load a stored session
        auth = self._config.auth
        name = self._config.name
        use_storage_state = SessionManager.has_valid_session(auth, name)
        storage_state_path = str(SessionManager.state_path(auth, name)) if use_storage_state else None

        context_kwargs: dict = {
            "user_agent": ua,
            "viewport": {"width": width, "height": height},
            "locale": bc.locale,
            "java_script_enabled": bc.java_script_enabled,
        }
        if storage_state_path:
            context_kwargs["storage_state"] = storage_state_path
            logger.info(f"[{name}] Launching browser with cached session")
        else:
            logger.info(f"[{name}] Launching browser (no cached session — will authenticate)")

        self._browser = await self._playwright.chromium.launch(
            headless=bc.headless,
            slow_mo=bc.slow_mo_ms,
        )
        self._context = await self._browser.new_context(**context_kwargs)

        # Inject anti-bot script into every page opened in this context
        await self._context.add_init_script(_ANTI_BOT_SCRIPT)

        self._page = await self._context.new_page()
        self._page.set_default_timeout(bc.timeout_ms)

        if not use_storage_state:
            # Perform authentication
            handler = get_auth_handler(auth.type)
            await handler.authenticate(self._page, auth, name)

            # After auth, handle cookie consent (first visit)
            await CookieConsentHandler.handle(self._page, self._config.cookie_consent)
        else:
            logger.debug(f"[{name}] Skipping auth (session reused)")

        return self

    async def __aexit__(self, *_args: object) -> None:
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def scrape(self, target: TargetConfig) -> str:
        """Navigate to a target page and extract content.

        Args:
            target: Target configuration describing URL, wait conditions, and
                extraction method.

        Returns:
            Extracted content as a string.
        """
        assert self._page is not None, "ScraperSession not started — use 'async with ScraperSession(...) as s:'"

        logger.info(f"Navigating to {target.url!r}")
        await self._page.goto(target.url, wait_until=target.wait_for)

        # Handle cookie consent banner that may appear on the target page too
        await CookieConsentHandler.handle(self._page, self._config.cookie_consent)

        # Wait for a specific element if configured
        if target.wait_for_selector:
            logger.debug(f"Waiting for selector {target.wait_for_selector!r}")
            try:
                await self._page.wait_for_selector(
                    target.wait_for_selector,
                    timeout=target.wait_for_selector_timeout_ms,
                )
            except Exception as exc:
                logger.warning(f"wait_for_selector timed out: {exc} — extracting page as-is")

        return await _extract(self._page, target.extract)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


async def _extract(page: object, config: ExtractConfig) -> str:
    """Extract content from the current page using the configured method.

    Args:
        page: Playwright ``Page``.
        config: Extraction configuration.
    """
    selector = config.selector or "body"

    if config.type == "text":
        content: str = await page.inner_text(selector)  # type: ignore[attr-defined]
        return content.strip()

    if config.type == "dom":
        html: str = await page.inner_html(selector)  # type: ignore[attr-defined]
        return html

    if config.type == "screenshot":
        png_bytes: bytes = await page.screenshot(full_page=True)  # type: ignore[attr-defined]
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"

    if config.type == "custom":
        if not config.custom_extractor:
            raise ValueError("ExtractConfig.custom_extractor must be set for type=custom")
        from genai_tk.utils.config_mngr import import_from_qualified

        extractor_fn = import_from_qualified(config.custom_extractor)
        result = await extractor_fn(page, config)
        return str(result)

    raise ValueError(f"Unknown extraction type: {config.type!r}")


# ---------------------------------------------------------------------------
# Convenience async function
# ---------------------------------------------------------------------------


async def run_scraper(
    config: WebScraperConfig,
    target_name: str | None = None,
) -> str:
    """Authenticate and scrape a single target, returning extracted content.

    This is the primary entry point used by the LangChain tool.

    Args:
        config: Full web scraper configuration.
        target_name: Name of the target to scrape.  Defaults to the first target.

    Returns:
        Extracted page content as a string.
    """
    if not config.targets:
        raise ValueError(f"Scraper '{config.name}' has no targets defined")

    target = config.targets[0] if target_name is None else config.get_target(target_name)

    async with ScraperSession(config) as session:
        return await session.scrape(target)
