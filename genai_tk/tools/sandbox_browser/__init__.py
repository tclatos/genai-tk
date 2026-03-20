"""Agent-driven browser automation tools for AIO sandbox.

Provides LangChain tools that control a real Chromium browser inside an
AIO sandbox container via CDP.  The agent uses these primitive tools —
guided by site-specific SKILL.md files — to navigate, authenticate, and
extract data from websites.
"""

from genai_tk.tools.sandbox_browser.factory import create_sandbox_browser_tools
from genai_tk.tools.sandbox_browser.models import CredentialRef, SandboxBrowserConfig
from genai_tk.tools.sandbox_browser.session import SandboxBrowserSession

__all__ = [
    "CredentialRef",
    "SandboxBrowserConfig",
    "SandboxBrowserSession",
    "create_sandbox_browser_tools",
]
