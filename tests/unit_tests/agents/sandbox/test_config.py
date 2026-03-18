"""Unit tests for sandbox config loader — all OmegaConf calls mocked."""

from unittest.mock import MagicMock, patch

from genai_tk.agents.sandbox.config import (
    get_docker_aio_settings,
    get_docker_smol_settings,
    get_e2b_settings,
    load_sandbox_config,
    resolve_sandbox_name,
)
from genai_tk.agents.sandbox.models import DockerAioSettings, SandboxConfig

_PATCH_TARGET = "genai_tk.utils.config_mngr.global_config"


def _mock_global_config(data: dict):
    """Return a mock global_config() whose .get() returns *data*."""
    mock_cfg = MagicMock()
    mock_cfg.get.return_value = data
    return mock_cfg


class TestLoadSandboxConfig:
    def test_returns_defaults_when_section_missing(self) -> None:
        with patch(_PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_global_config({})
            cfg = load_sandbox_config()
        assert cfg.default == "local"
        assert cfg.docker.aio.host_port == 18091

    def test_returns_defaults_on_import_error(self) -> None:
        with patch(_PATCH_TARGET, side_effect=Exception):
            cfg = load_sandbox_config()
        assert isinstance(cfg, SandboxConfig)

    def test_loads_custom_values(self) -> None:
        raw = {
            "default": "docker",
            "docker": {
                "aio": {"host_port": 9999, "image": "custom:v1"},
                "smolagents": {"mem_limit": "2g"},
            },
            "e2b": {"api_key": "mykey"},
            "wasm": {"enabled": False},
        }
        with patch(_PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_global_config(raw)
            cfg = load_sandbox_config()
        assert cfg.default == "docker"
        assert cfg.docker.aio.host_port == 9999
        assert cfg.docker.aio.image == "custom:v1"
        assert cfg.docker.smolagents.mem_limit == "2g"
        assert cfg.e2b.api_key == "mykey"

    def test_handles_omegaconf_container(self) -> None:
        """Verify OmegaConf DictConfig objects are converted via to_container."""
        from omegaconf import OmegaConf

        raw_omega = OmegaConf.create({"default": "e2b", "e2b": {"api_key": "omega-key", "timeout": 60}})
        with patch(_PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_global_config(raw_omega)
            cfg = load_sandbox_config()
        assert cfg.default == "e2b"
        assert cfg.e2b.api_key == "omega-key"
        assert cfg.e2b.timeout == 60


class TestGetDockerAioSettings:
    def test_returns_docker_aio(self) -> None:
        raw = {"default": "docker", "docker": {"aio": {"host_port": 8888}}}
        with patch(_PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_global_config(raw)
            aio = get_docker_aio_settings()
        assert isinstance(aio, DockerAioSettings)
        assert aio.host_port == 8888


class TestGetDockerSmolSettings:
    def test_returns_smol_defaults(self) -> None:
        with patch(_PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_global_config({})
            smol = get_docker_smol_settings()
        assert smol.image == "python:3.12-slim"
        assert smol.mem_limit == "512m"


class TestGetE2bSettings:
    def test_api_key_none_by_default(self) -> None:
        with patch(_PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_global_config({})
            e2b = get_e2b_settings()
        assert e2b.api_key is None

    def test_api_key_loaded(self) -> None:
        raw = {"e2b": {"api_key": "loaded-key"}}
        with patch(_PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_global_config(raw)
            e2b = get_e2b_settings()
        assert e2b.api_key == "loaded-key"


class TestResolveDefaultSandbox:
    def test_returns_local_by_default(self) -> None:
        with patch(_PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_global_config({})
            result = resolve_sandbox_name(None)
        assert result == "local"

    def test_returns_configured_default(self) -> None:
        raw = {"default": "docker"}
        with patch(_PATCH_TARGET) as mock_gc:
            mock_gc.return_value = _mock_global_config(raw)
            result = resolve_sandbox_name(None)
        assert result == "docker"
