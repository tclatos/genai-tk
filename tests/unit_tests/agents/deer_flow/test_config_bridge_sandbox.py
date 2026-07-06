"""Unit tests for DeerFlow config bridge sandbox forwarding.

Verifies that ``sandbox.docker.aio`` settings (``env_vars``, ``volumes``,
``image``) are forwarded into the generated DeerFlow ``config.yaml`` so the
LangChain/deepagents harness and the DeerFlow harness share one
``config/sandbox.yaml`` source of truth.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from genai_tk.agents.sandbox.models import DockerAioSettings, VolumeMountConfig

pytestmark = pytest.mark.unit


def _write_docker_config(aio: DockerAioSettings, tmp_path: Path) -> dict:
    """Generate a DeerFlow config with the given AIO settings and return its parsed sandbox section."""
    from genai_tk.agents.deer_flow.config_bridge import write_deer_flow_config

    with patch("genai_tk.agents.sandbox.config.get_docker_aio_settings", return_value=aio):
        config_path = write_deer_flow_config(
            models=[
                {"name": "test-model", "display_name": "Test", "use": "langchain_openai:ChatOpenAI", "model": "gpt-4"}
            ],
            sandbox="docker",
            config_dir=str(tmp_path),
        )

    parsed = yaml.safe_load(Path(config_path).read_text())
    return parsed["sandbox"]


def test_docker_sandbox_forwards_env_vars(tmp_path: Path) -> None:
    """env_vars from sandbox.yaml become DeerFlow sandbox.environment."""
    aio = DockerAioSettings(env_vars={"DEBUG": "1", "MY_API_KEY": "$MY_API_KEY"})
    sandbox = _write_docker_config(aio, tmp_path)

    assert sandbox["environment"] == {"DEBUG": "1", "MY_API_KEY": "$MY_API_KEY"}


def test_docker_sandbox_forwards_volumes(tmp_path: Path) -> None:
    """volumes from sandbox.yaml become DeerFlow sandbox.mounts with identical field names."""
    aio = DockerAioSettings(
        volumes=[
            VolumeMountConfig(host_path="/host/data", container_path="/mnt/data", read_only=True),
            VolumeMountConfig(host_path="/host/skills", container_path="/mnt/skills", read_only=False),
        ]
    )
    sandbox = _write_docker_config(aio, tmp_path)

    mounts = sandbox["mounts"]
    assert len(mounts) == 2
    assert mounts[0] == {"host_path": "/host/data", "container_path": "/mnt/data", "read_only": True}
    assert mounts[1] == {"host_path": "/host/skills", "container_path": "/mnt/skills", "read_only": False}


def test_docker_sandbox_forwards_image(tmp_path: Path) -> None:
    """The configured image is forwarded so both harnesses run the same container image."""
    aio = DockerAioSettings(image="my-registry/sandbox:v2")
    sandbox = _write_docker_config(aio, tmp_path)

    assert sandbox["image"] == "my-registry/sandbox:v2"


def test_docker_sandbox_uses_aio_sandbox_provider(tmp_path: Path) -> None:
    """The generated sandbox.use points at DeerFlow's AioSandboxProvider."""
    sandbox = _write_docker_config(DockerAioSettings(), tmp_path)

    assert "aio_sandbox_provider:AioSandboxProvider" in sandbox["use"]


def test_docker_sandbox_empty_env_and_volumes_default_to_empty(tmp_path: Path) -> None:
    """When env_vars/volumes are unset, the forwarded keys are empty (not absent)."""
    sandbox = _write_docker_config(DockerAioSettings(), tmp_path)

    assert sandbox["environment"] == {}
    assert sandbox["mounts"] == []
