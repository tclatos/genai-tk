"""Integration tests for LangchainAgent sandbox support.

Tests cover the sandbox override mechanics (profile promotion, backend wiring,
skill_directories clearing) using mocked Docker/AioSandboxBackend so no real
container is required.

Real-Docker tests (``--include-docker``) actually start the agent-infra/sandbox
container and require ``docker`` to be available on the host.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genai_tk.agents.langchain.langchain_agent import LangchainAgent, SandboxType

# ---------------------------------------------------------------------------
# Type annotation
# ---------------------------------------------------------------------------


def test_sandbox_type_alias() -> None:
    """SandboxType is exported from the module and is Literal['local', 'docker']."""
    import typing

    args = typing.get_args(SandboxType)
    assert set(args) == {"local", "docker"}


def test_sandbox_field_accepts_literal_values() -> None:
    """LangchainAgent accepts 'local' and 'docker' as sandbox values."""
    a = LangchainAgent(llm="parrot_local@fake", sandbox="local")
    assert a.sandbox == "local"
    b = LangchainAgent(llm="parrot_local@fake", sandbox="docker")
    assert b.sandbox == "docker"
    c = LangchainAgent(llm="parrot_local@fake", sandbox=None)
    assert c.sandbox is None


# ---------------------------------------------------------------------------
# Profile promotion mechanics (no Docker needed)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.fake_models
@pytest.mark.asyncio
async def test_docker_sandbox_promotes_react_to_deep() -> None:
    """When sandbox='docker', a react profile is promoted to type='deep'."""
    from genai_tk.agents.langchain.config import BackendConfig

    mock_agent = MagicMock()

    async def fake_create(profile: Any, **kwargs: Any) -> MagicMock:
        assert profile.type == "deep", f"Expected deep, got {profile.type}"
        assert isinstance(profile.backend, BackendConfig)
        assert profile.backend.type == "docker"
        return mock_agent

    with patch("genai_tk.agents.langchain.factory.create_langchain_agent", side_effect=fake_create):
        agent = LangchainAgent(llm="parrot_local@fake", agent_type="react", sandbox="docker")
        await agent._ensure_initialized()


@pytest.mark.integration
@pytest.mark.fake_models
@pytest.mark.asyncio
async def test_docker_sandbox_keeps_deep_type() -> None:
    """When sandbox='docker' on a deep profile, type stays 'deep'."""
    from genai_tk.agents.langchain.config import BackendConfig

    mock_agent = MagicMock()

    async def fake_create(profile: Any, **kwargs: Any) -> MagicMock:
        assert profile.type == "deep"
        assert isinstance(profile.backend, BackendConfig)
        assert profile.backend.type == "docker"
        return mock_agent

    with patch("genai_tk.agents.langchain.factory.create_langchain_agent", side_effect=fake_create):
        agent = LangchainAgent(llm="parrot_local@fake", agent_type="deep", sandbox="docker")
        await agent._ensure_initialized()


@pytest.mark.integration
@pytest.mark.fake_models
@pytest.mark.asyncio
async def test_docker_sandbox_clears_skill_directories() -> None:
    """skill_directories is always empty when sandbox='docker'.

    Local host paths are not accessible inside the container; passing them to
    SkillsMiddleware causes a 404 error from the sandbox API.
    """
    captured: dict[str, Any] = {}

    async def fake_create(profile: Any, **kwargs: Any) -> MagicMock:
        captured["skill_directories"] = profile.skill_directories
        return MagicMock()

    with patch("genai_tk.agents.langchain.factory.create_langchain_agent", side_effect=fake_create):
        agent = LangchainAgent(llm="parrot_local@fake", sandbox="docker")
        await agent._ensure_initialized()

    assert captured["skill_directories"] == [], (
        "skill_directories must be empty with docker sandbox to prevent SkillsMiddleware 404"
    )


@pytest.mark.integration
@pytest.mark.fake_models
@pytest.mark.asyncio
async def test_local_sandbox_preserves_profile() -> None:
    """sandbox='local' (or None) leaves the profile type and backend unchanged."""
    captured: dict[str, Any] = {}

    async def fake_create(profile: Any, **kwargs: Any) -> MagicMock:
        captured["type"] = profile.type
        captured["backend"] = profile.backend
        return MagicMock()

    with patch("genai_tk.agents.langchain.factory.create_langchain_agent", side_effect=fake_create):
        agent = LangchainAgent(llm="parrot_local@fake", agent_type="react", sandbox="local")
        await agent._ensure_initialized()

    assert captured["type"] == "react"
    # backend may be None or the profile-default; it must NOT be a docker BackendConfig
    from genai_tk.agents.langchain.config import BackendConfig

    backend = captured["backend"]
    assert not (isinstance(backend, BackendConfig) and backend.type == "docker")


@pytest.mark.integration
@pytest.mark.fake_models
@pytest.mark.asyncio
async def test_no_sandbox_preserves_profile() -> None:
    """sandbox=None leaves the profile entirely unchanged."""
    captured: dict[str, Any] = {}

    async def fake_create(profile: Any, **kwargs: Any) -> MagicMock:
        captured["type"] = profile.type
        return MagicMock()

    with patch("genai_tk.agents.langchain.factory.create_langchain_agent", side_effect=fake_create):
        agent = LangchainAgent(llm="parrot_local@fake", agent_type="react", sandbox=None)
        await agent._ensure_initialized()

    assert captured["type"] == "react"


# ---------------------------------------------------------------------------
# AioSandboxBackend wiring (mocked Docker)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.fake_models
@pytest.mark.asyncio
async def test_docker_sandbox_uses_shared_config_image() -> None:
    """The AioSandboxBackend started for 'docker' sandbox uses the image from sandbox.yaml."""
    from genai_tk.agents.sandbox.models import DockerAioSettings

    started_config: dict[str, Any] = {}

    class FakeBackend:
        def __init__(self, config: DockerAioSettings) -> None:
            started_config["image"] = config.image
            started_config["server_url"] = config.opensandbox_server_url

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

    fake_agent = MagicMock()
    fake_agent._backend = FakeBackend.__new__(FakeBackend)

    async def fake_create(profile: Any, **kwargs: Any) -> MagicMock:
        # Simulate the backend being created inside create_langchain_agent
        from genai_tk.agents.langchain.config import create_backend

        backend = await create_backend(profile.backend)
        return fake_agent

    with (
        patch("genai_tk.agents.sandbox.aio_backend.AioSandboxBackend", FakeBackend),
        patch("genai_tk.agents.langchain.factory.create_langchain_agent", side_effect=fake_create),
        patch("genai_tk.utils.config_mngr.global_config", return_value=MagicMock(**{"get.return_value": {}})),
    ):
        agent = LangchainAgent(llm="parrot_local@fake", sandbox="docker")
        await agent._ensure_initialized()

    assert started_config["image"] == "ghcr.io/agent-infra/sandbox:latest"
    assert started_config["server_url"] == "http://localhost:8080"


# ---------------------------------------------------------------------------
# Full fake-model run with mocked sandbox backend
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.fake_models
@pytest.mark.asyncio
async def test_docker_sandbox_end_to_end_mocked() -> None:
    """arun() completes successfully with sandbox='docker' when backend is mocked."""
    mock_backend = AsyncMock()
    mock_backend.start = AsyncMock()
    mock_backend.stop = AsyncMock()

    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(return_value={"messages": [MagicMock(content="42 is the answer")]})
    mock_compiled._backend = mock_backend

    async def fake_create(profile: Any, **kwargs: Any) -> MagicMock:
        assert profile.type == "deep"
        assert profile.skill_directories == []
        return mock_compiled

    with patch("genai_tk.agents.langchain.factory.create_langchain_agent", side_effect=fake_create):
        agent = LangchainAgent(llm="parrot_local@fake", sandbox="docker")
        result = await agent.arun("What is 6 × 7?")

    assert "42" in result


# ---------------------------------------------------------------------------
# Stale-container cleanup removed — lifecycle managed by OpenSandbox server
# ---------------------------------------------------------------------------
