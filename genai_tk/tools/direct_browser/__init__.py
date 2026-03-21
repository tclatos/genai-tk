"""Host-local Playwright browser automation tools.

Provides the same 12 browser tools as ``genai_tk.tools.sandbox_browser`` but
backed by a host-local Chromium launched via Playwright.  Uses real GPU, real
platform UA, and real network stack — useful for sites with deep fingerprinting
that detect the AIO sandbox environment.
"""

from genai_tk.tools.direct_browser.factory import create_direct_browser_tools
from genai_tk.tools.direct_browser.models import DirectBrowserConfig
from genai_tk.tools.direct_browser.session import DirectBrowserSession

__all__ = [
    "DirectBrowserConfig",
    "DirectBrowserSession",
    "create_direct_browser_tools",
]
