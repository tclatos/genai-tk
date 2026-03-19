"""Realistic rotating user-agent pool for Playwright browser contexts.

Maintains a curated pool of real Chrome 120–134 user agents covering
Windows, macOS, and Linux.  Three modes are supported:

- ``rotating`` — round-robin through the pool (per-process counter)
- ``random``   — uniformly random pick on each call
- any other string — returned verbatim (caller-supplied UA)

Example:
    ```python
    from genai_tk.tools.browser.user_agents import get_user_agent

    ua = get_user_agent("rotating")  # → "Mozilla/5.0 (Windows NT 10.0...)"
    ua = get_user_agent("random")
    ua = get_user_agent("Mozilla/5.0 (X11; Linux x86_64)...")
    ```
"""

from __future__ import annotations

import random
import threading

# ---------------------------------------------------------------------------
# UA pool — real Chrome 120-134 signatures, multiple OS
# ---------------------------------------------------------------------------

_UA_POOL: list[str] = [
    # Chrome 134 — Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.6998.89 Safari/537.36",
    # Chrome 134 — macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.6998.89 Safari/537.36",
    # Chrome 134 — Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    # Chrome 133 — Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.6943.141 Safari/537.36",
    # Chrome 133 — macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.6943.98 Safari/537.36",
    # Chrome 133 — Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.6943.141 Safari/537.36",
    # Chrome 132 — Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.6834.160 Safari/537.36",
    # Chrome 132 — macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.6834.83 Safari/537.36",
    # Chrome 132 — Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    # Chrome 131 — Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    # Chrome 131 — macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    # Chrome 130 — Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    # Chrome 130 — macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    # Chrome 129 — Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    # Chrome 128 — Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    # Chrome 127 — macOS (older machines)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    # Edge 134 — Windows (Chromium-based, helps diversity)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36 Edg/134.0.0.0",
    # Edge 133 — Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0",
]

# ---------------------------------------------------------------------------
# Round-robin counter (thread-safe)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_counter: int = 0


def _next_index() -> int:
    global _counter
    with _lock:
        idx = _counter % len(_UA_POOL)
        _counter += 1
        return idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_user_agent(mode: str = "rotating") -> str:
    """Return a user-agent string for the given mode.

    Args:
        mode: ``"rotating"`` for round-robin, ``"random"`` for uniform random,
              or any other string which is returned verbatim.
    """
    if mode == "rotating":
        return _UA_POOL[_next_index()]
    if mode == "random":
        return random.choice(_UA_POOL)
    return mode


def get_pool() -> list[str]:
    """Return a copy of the full UA pool (useful for testing)."""
    return list(_UA_POOL)
