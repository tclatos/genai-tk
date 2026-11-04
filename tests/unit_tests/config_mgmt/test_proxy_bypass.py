"""Unit tests for genai_tk.config_mgmt.config_mngr.proxy_bypass."""

from __future__ import annotations

from omegaconf import OmegaConf

from genai_tk.config_mgmt.config_mngr.proxy_bypass import (
    PROXY_ENV_VARS,
    apply_proxy_bypass,
    purge_proxy_env,
)


class TestPurgeProxyEnv:
    def test_removes_all_proxy_variables(self) -> None:
        env = {
            "HTTP_PROXY": "http://165.225.21.246:80",
            "HTTPS_PROXY": "http://165.225.21.246:80",
            "http_proxy": "http://165.225.21.246:80",
            "https_proxy": "http://165.225.21.246:80",
            "WSL_PAC_URL": "http://127.0.0.1:9000/proxy.pac",
            "OTHER_VAR": "keep_me",
        }
        removed = purge_proxy_env(env)
        assert set(removed) == set(PROXY_ENV_VARS)
        for var in PROXY_ENV_VARS:
            assert var not in env
        assert env["OTHER_VAR"] == "keep_me"

    def test_noop_when_no_proxy_vars(self) -> None:
        env = {"OTHER_VAR": "keep_me"}
        removed = purge_proxy_env(env)
        assert removed == []
        assert env == {"OTHER_VAR": "keep_me"}


class TestApplyProxyBypass:
    def test_merges_default_bypass_hosts(self) -> None:
        config = OmegaConf.create({})
        env: dict[str, str] = {}
        apply_proxy_bypass(config, env=env)
        no_proxy = env.get("NO_PROXY", "")
        assert "localhost" in no_proxy
        assert "127.0.0.1" in no_proxy
        assert "huggingface.co" in no_proxy
        assert "pypi.org" in no_proxy
        assert env.get("no_proxy") == no_proxy

    def test_merges_extra_bypass_hosts(self) -> None:
        config = OmegaConf.create({"net": {"proxy_bypass_hosts": ["custom.internal.domain", "another.host"]}})
        env: dict[str, str] = {}
        apply_proxy_bypass(config, env=env)
        no_proxy = env.get("NO_PROXY", "")
        assert "custom.internal.domain" in no_proxy
        assert "another.host" in no_proxy
        assert "huggingface.co" in no_proxy

    def test_disable_proxy_flag_purges_env(self) -> None:
        config = OmegaConf.create({"net": {"disable_proxy": True}})
        env = {
            "HTTP_PROXY": "http://proxy.corp:8080",
            "HTTPS_PROXY": "http://proxy.corp:8080",
            "http_proxy": "http://proxy.corp:8080",
            "https_proxy": "http://proxy.corp:8080",
            "WSL_PAC_URL": "http://127.0.0.1:9000/pac",
        }
        apply_proxy_bypass(config, env=env)
        for var in PROXY_ENV_VARS:
            assert var not in env
        assert "NO_PROXY" not in env

    def test_resilient_to_malformed_config(self) -> None:
        # Should not raise even with completely invalid config object
        apply_proxy_bypass(None, env={})  # type: ignore[arg-type]
