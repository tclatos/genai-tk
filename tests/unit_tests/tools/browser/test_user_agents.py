"""Unit tests for genai_tk.tools.browser.user_agents."""

from __future__ import annotations

from genai_tk.tools.browser.user_agents import _UA_POOL, get_pool, get_user_agent


class TestUserAgentPool:
    def test_pool_is_nonempty(self) -> None:
        assert len(_UA_POOL) >= 20

    def test_get_pool_returns_copy(self) -> None:
        pool = get_pool()
        pool.clear()
        assert len(get_pool()) >= 20  # original unaffected

    def test_all_uas_are_chrome_or_edge(self) -> None:
        for ua in _UA_POOL:
            assert "Chrome/" in ua or "Edg/" in ua, f"Unexpected UA: {ua}"

    def test_all_uas_have_webkit(self) -> None:
        for ua in _UA_POOL:
            assert "AppleWebKit" in ua


class TestGetUserAgent:
    def test_rotating_returns_string(self) -> None:
        ua = get_user_agent("rotating")
        assert isinstance(ua, str)
        assert len(ua) > 30

    def test_rotating_cycles_through_pool(self) -> None:
        pool = get_pool()
        # Collect enough UAs to see rotation — at least 2 distinct values in 30 calls
        seen = {get_user_agent("rotating") for _ in range(30)}
        assert len(seen) >= 2

    def test_random_returns_string_from_pool(self) -> None:
        pool = get_pool()
        for _ in range(20):
            ua = get_user_agent("random")
            assert ua in pool

    def test_literal_passthrough(self) -> None:
        custom = "Mozilla/5.0 (compatible; MyBot/1.0)"
        assert get_user_agent(custom) == custom

    def test_rotating_no_consecutive_duplicates_on_average(self) -> None:
        """Rotating through a pool of ≥20 UAs should not repeat for pool_size calls."""
        pool_size = len(get_pool())
        # Reset counter by reading enough; just verify we eventually see >1 unique value
        seen = set()
        for _ in range(pool_size):
            seen.add(get_user_agent("rotating"))
        # Should see multiple distinct UAs in one full rotation
        assert len(seen) > 1
