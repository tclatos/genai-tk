"""Session validity checking and storage-state lifecycle management.

Reads a Playwright ``storageState`` JSON file and checks whether the
cookies inside are still valid (none expired within the next 60 seconds).
This avoids an unnecessary login round-trip when a cached session is still
live.

Example:
    ```python
    from genai_tk.tools.browser.session_manager import SessionManager
    from genai_tk.tools.browser.models import AuthConfig

    auth = AuthConfig(...)
    if SessionManager.has_valid_session(auth, scraper_name="my_site"):
        print("Session still live, skipping login")
    ```
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger

from genai_tk.tools.browser.models import AuthConfig

# How many seconds of buffer to require before a cookie is considered valid.
# If the earliest-expiring cookie expires within this window, we re-auth.
_EXPIRY_BUFFER_SECONDS = 60


class SessionManager:
    """Utility methods for Playwright storage-state persistence."""

    @staticmethod
    def state_path(auth: AuthConfig, scraper_name: str) -> Path:
        """Resolve the storage-state path for the given scraper name.

        Args:
            auth: Authentication configuration holding session settings.
            scraper_name: Scraper config name used to fill ``{name}`` placeholder.
        """
        return auth.session.resolve_path(scraper_name)

    @staticmethod
    def has_valid_session(auth: AuthConfig, scraper_name: str) -> bool:
        """Return ``True`` if a valid, non-expired session exists on disk.

        Checks:
        1. ``session.check_validity`` is enabled.
        2. The storage-state file exists.
        3. All session cookies expire at least ``_EXPIRY_BUFFER_SECONDS`` in the future.

        If the file is unreadable or malformed, returns ``False`` (triggers re-auth).

        Args:
            auth: Authentication configuration.
            scraper_name: Used to resolve ``{name}`` in the storage-state path.
        """
        if not auth.session.check_validity:
            return False

        path = SessionManager.state_path(auth, scraper_name)
        if not path.exists():
            logger.debug(f"No session file at {path} — will authenticate")
            return False

        try:
            state = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Cannot read session file {path}: {exc} — will re-authenticate")
            return False

        cookies: list[dict] = state.get("cookies", [])
        if not cookies:
            logger.debug(f"Session file {path} has no cookies — will authenticate")
            return False

        now = time.time()
        threshold = now + _EXPIRY_BUFFER_SECONDS

        for cookie in cookies:
            expires = cookie.get("expires", -1)
            if expires == -1:
                # Session cookie (no explicit expiry) — treat as valid
                continue
            if expires < threshold:
                logger.info(
                    f"Session cookie '{cookie.get('name', '?')}' expires soon "
                    f"(in {int(expires - now)}s) — will re-authenticate"
                )
                return False

        logger.info(f"Reusing valid session from {path}")
        return True

    @staticmethod
    def delete_session(auth: AuthConfig, scraper_name: str) -> None:
        """Delete the stored session file to force re-authentication on next run.

        Args:
            auth: Authentication configuration.
            scraper_name: Used to resolve the storage-state path.
        """
        path = SessionManager.state_path(auth, scraper_name)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted session file {path}")

    @staticmethod
    def ensure_parent_dir(auth: AuthConfig, scraper_name: str) -> None:
        """Create parent directories for the storage-state file if missing.

        Args:
            auth: Authentication configuration.
            scraper_name: Used to resolve the storage-state path.
        """
        path = SessionManager.state_path(auth, scraper_name)
        path.parent.mkdir(parents=True, exist_ok=True)
