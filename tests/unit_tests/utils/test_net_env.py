"""Unit tests for genai_tk.utils.net_env (no network — probes are monkeypatched)."""

from __future__ import annotations

import pytest

from genai_tk.utils.net_env import (
    classify_host,
    default_bypass_hosts,
    ensure_no_proxy_hosts,
    no_proxy_entries,
    recommended_bypass_hosts,
)


class TestEnsureNoProxyHosts:
    def test_adds_to_both_spellings(self) -> None:
        env: dict[str, str] = {}
        ensure_no_proxy_hosts(["localhost", "example.com"], env)
        assert env["NO_PROXY"] == "localhost,example.com"
        assert env["no_proxy"] == "localhost,example.com"

    def test_preserves_existing_entries(self) -> None:
        env = {"NO_PROXY": "pypi.org", "no_proxy": "pypi.org"}
        ensure_no_proxy_hosts(["pypi.org", "example.com"], env)
        assert env["NO_PROXY"] == "pypi.org,example.com"

    def test_idempotent(self) -> None:
        env: dict[str, str] = {}
        ensure_no_proxy_hosts(["example.com"], env)
        ensure_no_proxy_hosts(["example.com"], env)
        assert env["NO_PROXY"] == "example.com"


class TestNoProxyEntries:
    def test_union_of_both_spellings(self) -> None:
        env = {"NO_PROXY": "a.com,b.com", "no_proxy": "b.com,c.com"}
        assert no_proxy_entries(env) == ["a.com", "b.com", "c.com"]

    def test_ignores_blank_entries(self) -> None:
        env = {"NO_PROXY": "a.com,, b.com "}
        assert no_proxy_entries(env) == ["a.com", "b.com"]

    def test_empty_when_unset(self) -> None:
        assert no_proxy_entries({}) == []


class TestClassification:
    def test_bypass_needed_when_proxy_blocks_but_direct_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from genai_tk.utils import net_env

        def fake_probe(host: str, *, use_env_proxy: bool, timeout: float = 3.0) -> bool:
            return not use_env_proxy

        monkeypatch.setattr(net_env, "probe_host_reachable", fake_probe)
        assert classify_host("example.com") == "bypass_needed"

    def test_proxy_ok_when_reachable_through_proxy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from genai_tk.utils import net_env

        monkeypatch.setattr(net_env, "probe_host_reachable", lambda *args, **kwargs: True)
        assert classify_host("example.com") == "proxy_ok"

    def test_unreachable_both_ways(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from genai_tk.utils import net_env

        monkeypatch.setattr(net_env, "probe_host_reachable", lambda *args, **kwargs: False)
        assert classify_host("example.com") == "unreachable"

    def test_recommended_bypass_hosts_groups_by_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from genai_tk.utils import net_env

        def fake_classify(host: str, *, timeout: float = 3.0) -> str:
            return {"a.com": "proxy_ok", "b.com": "bypass_needed", "c.com": "unreachable"}[host]

        monkeypatch.setattr(net_env, "classify_host", fake_classify)
        result = recommended_bypass_hosts(["a.com", "b.com", "c.com"])
        assert result == {"proxy_ok": ["a.com"], "bypass_needed": ["b.com"], "unreachable": ["c.com"]}


class TestDefaultBypassHosts:
    def test_includes_loopback(self) -> None:
        assert "localhost" in default_bypass_hosts()
