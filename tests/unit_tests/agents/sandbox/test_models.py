"""Unit tests for sandbox Pydantic models."""

from genai_tk.agents.sandbox.models import (
    DockerAioSettings,
    DockerSandboxSettings,
    DockerSmolSettings,
    E2bSandboxSettings,
    SandboxConfig,
    WasmSandboxSettings,
)


class TestDockerAioSettings:
    def test_defaults(self) -> None:
        s = DockerAioSettings()
        assert s.host == "127.0.0.1"
        assert s.host_port == 18091
        assert s.startup_timeout == 60.0
        assert s.work_dir == "/home/user"
        assert s.env_vars == {}

    def test_custom_values(self) -> None:
        s = DockerAioSettings(host="0.0.0.0", host_port=9000, env_vars={"FOO": "bar"})
        assert s.host == "0.0.0.0"
        assert s.host_port == 9000
        assert s.env_vars == {"FOO": "bar"}

    def test_model_validate(self) -> None:
        s = DockerAioSettings.model_validate({"image": "my-image:1.0", "host_port": 8080})
        assert s.image == "my-image:1.0"
        assert s.host_port == 8080
        assert s.host == "127.0.0.1"  # default


class TestDockerSmolSettings:
    def test_defaults(self) -> None:
        s = DockerSmolSettings()
        assert s.image == "python:3.12-slim"
        assert s.mem_limit == "512m"
        assert s.cpu_quota == 50000
        assert s.pids_limit == 100

    def test_custom_values(self) -> None:
        s = DockerSmolSettings(image="python:3.11", mem_limit="256m")
        assert s.image == "python:3.11"
        assert s.mem_limit == "256m"


class TestDockerSandboxSettings:
    def test_defaults(self) -> None:
        s = DockerSandboxSettings()
        assert isinstance(s.aio, DockerAioSettings)
        assert isinstance(s.smolagents, DockerSmolSettings)

    def test_nested_model_validate(self) -> None:
        s = DockerSandboxSettings.model_validate({"aio": {"host_port": 9999}, "smolagents": {"mem_limit": "1g"}})
        assert s.aio.host_port == 9999
        assert s.smolagents.mem_limit == "1g"


class TestE2bSandboxSettings:
    def test_defaults(self) -> None:
        s = E2bSandboxSettings()
        assert s.api_key is None
        assert s.template is None
        assert s.timeout == 300

    def test_with_api_key(self) -> None:
        s = E2bSandboxSettings(api_key="test-key", timeout=60)
        assert s.api_key == "test-key"
        assert s.timeout == 60


class TestWasmSandboxSettings:
    def test_defaults(self) -> None:
        s = WasmSandboxSettings()
        assert s.enabled is False


class TestSandboxConfig:
    def test_defaults(self) -> None:
        cfg = SandboxConfig()
        assert cfg.default == "local"
        assert isinstance(cfg.docker, DockerSandboxSettings)
        assert isinstance(cfg.e2b, E2bSandboxSettings)
        assert isinstance(cfg.wasm, WasmSandboxSettings)

    def test_model_validate_full(self) -> None:
        data = {
            "default": "docker",
            "docker": {
                "aio": {"host_port": 9191, "image": "custom:latest"},
                "smolagents": {"mem_limit": "1g"},
            },
            "e2b": {"api_key": "abc123", "timeout": 120},
            "wasm": {"enabled": False},
        }
        cfg = SandboxConfig.model_validate(data)
        assert cfg.default == "docker"
        assert cfg.docker.aio.host_port == 9191
        assert cfg.docker.aio.image == "custom:latest"
        assert cfg.docker.smolagents.mem_limit == "1g"
        assert cfg.e2b.api_key == "abc123"
        assert cfg.e2b.timeout == 120

    def test_partial_config(self) -> None:
        cfg = SandboxConfig.model_validate({"default": "e2b"})
        assert cfg.default == "e2b"
        assert cfg.docker.aio.host_port == 18091  # default preserved
