"""Tests for Deer-flow profile validation functions."""

from unittest.mock import patch

import pytest

from genai_tk.agents.deer_flow.cli_commands import (
    _check_agent_sandbox_importable,
    _validate_and_normalize_sandbox,
    _validate_docker_sandbox,
)
from genai_tk.agents.deer_flow.profile import (
    DeerFlowProfile,
    DockerSandboxError,
    MCPServerNotFoundError,
    ProfileNotFoundError,
    get_available_modes,
    get_available_profile_names,
    validate_mcp_servers,
    validate_profile_name,
)


def test_get_available_profile_names():
    """Test getting profile names from list."""
    profiles = [
        DeerFlowProfile(name="Profile 1", description="Test 1"),
        DeerFlowProfile(name="Profile 2", description="Test 2"),
    ]
    names = get_available_profile_names(profiles)
    assert names == ["Profile 1", "Profile 2"]


def test_get_available_modes():
    """Test getting valid modes."""
    modes = get_available_modes()
    assert modes == ["flash", "thinking", "pro", "ultra"]


def test_validate_profile_name_success():
    """Test successful profile validation."""
    profiles = [
        DeerFlowProfile(name="Research Assistant", description="Researcher"),
        DeerFlowProfile(name="Coder", description="Coding helper"),
    ]

    # Exact match
    result = validate_profile_name("Research Assistant", profiles)
    assert result.name == "Research Assistant"

    # Case insensitive
    result = validate_profile_name("research assistant", profiles)
    assert result.name == "Research Assistant"


def test_validate_profile_name_not_found():
    """Test profile validation with invalid name."""
    profiles = [
        DeerFlowProfile(name="Research Assistant", description="Researcher"),
        DeerFlowProfile(name="Coder", description="Coding helper"),
    ]

    with pytest.raises(ProfileNotFoundError) as exc_info:
        validate_profile_name("NonExistent", profiles)

    error = exc_info.value
    assert error.profile_name == "NonExistent"
    assert "Research Assistant" in error.available_profiles
    assert "Coder" in error.available_profiles


def test_validate_mode_success():
    """Test that valid modes are accepted by Pydantic."""
    for mode in ("flash", "thinking", "pro", "ultra"):
        profile = DeerFlowProfile(name="test", mode=mode)  # type: ignore[arg-type]
        assert profile.mode == mode


def test_validate_mode_invalid():
    """Test that Pydantic rejects invalid modes."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DeerFlowProfile(name="test", mode="invalid_mode")  # type: ignore[arg-type]


def test_validate_mcp_servers_empty():
    """Test MCP validation with empty list."""
    result = validate_mcp_servers([])
    assert result == []


def test_validate_mcp_servers_success(monkeypatch):
    """Test successful MCP server validation."""

    # Mock the config to return some servers
    class MockConfig:
        def get(self, key, default=None):
            if key == "mcp":
                return {"servers": {"math": {}, "weather": {}, "tech_news": {}}}
            return default

    mock_global_config = MockConfig()

    # Patch at the right location
    import genai_tk.utils.config_mngr as config_mngr

    monkeypatch.setattr(config_mngr, "global_config", lambda: mock_global_config)

    # Should succeed with valid servers
    result = validate_mcp_servers(["math", "weather"])
    assert result == ["math", "weather"]


def test_validate_mcp_servers_invalid(monkeypatch):
    """Test MCP validation with invalid servers."""

    # Mock the config to return some servers
    class MockConfig:
        def get(self, key, default=None):
            if key == "mcp":
                return {"servers": {"math": {}, "weather": {}}}
            return default

    mock_global_config = MockConfig()

    # Patch at the right location
    import genai_tk.utils.config_mngr as config_mngr

    monkeypatch.setattr(config_mngr, "global_config", lambda: mock_global_config)

    # Should fail with invalid server
    with pytest.raises(MCPServerNotFoundError) as exc_info:
        validate_mcp_servers(["math", "invalid_server", "another_invalid"])

    error = exc_info.value
    assert "invalid_server" in error.invalid_servers
    assert "another_invalid" in error.invalid_servers
    assert "math" in error.available_servers
    assert "weather" in error.available_servers


# ---------------------------------------------------------------------------
# Sandbox validation tests
# ---------------------------------------------------------------------------


def test_validate_and_normalize_sandbox_local():
    """Test sandbox normalization for 'local'."""
    assert _validate_and_normalize_sandbox("local") == "local"
    assert _validate_and_normalize_sandbox("LOCAL") == "local"
    assert _validate_and_normalize_sandbox("  local  ") == "local"
    assert _validate_and_normalize_sandbox("") == "local"
    assert _validate_and_normalize_sandbox(None) == "local"


def test_validate_and_normalize_sandbox_docker():
    """Test sandbox normalization for 'docker'."""
    assert _validate_and_normalize_sandbox("docker") == "docker"
    assert _validate_and_normalize_sandbox("DOCKER") == "docker"
    assert _validate_and_normalize_sandbox("  Docker  ") == "docker"


def test_validate_and_normalize_sandbox_invalid():
    """Test sandbox normalization rejects invalid values."""
    from click.exceptions import Exit

    with pytest.raises(Exit):
        _validate_and_normalize_sandbox("podman")


class TestDockerSandboxValidation:
    """Tests for Docker sandbox prerequisite validation."""

    def test_validate_docker_sandbox_both_available(self):
        """No error when both Docker and agent-sandbox are available."""
        with (
            patch(
                "genai_tk.agents.deer_flow.cli_commands._check_docker_available",
                return_value=True,
            ),
            patch(
                "genai_tk.agents.deer_flow.cli_commands._check_agent_sandbox_importable",
                return_value=True,
            ),
        ):
            _validate_docker_sandbox()  # Should not raise

    def test_validate_docker_sandbox_no_docker(self):
        """Error when Docker is not available."""
        with (
            patch(
                "genai_tk.agents.deer_flow.cli_commands._check_docker_available",
                return_value=False,
            ),
            patch(
                "genai_tk.agents.deer_flow.cli_commands._check_agent_sandbox_importable",
                return_value=True,
            ),
            pytest.raises(DockerSandboxError) as exc_info,
        ):
            _validate_docker_sandbox()

        error = exc_info.value
        assert len(error.reasons) == 1
        assert "Docker is not available" in error.reasons[0]

    def test_validate_docker_sandbox_no_agent_sandbox(self):
        """Error when agent-sandbox package is missing."""
        with (
            patch(
                "genai_tk.agents.deer_flow.cli_commands._check_docker_available",
                return_value=True,
            ),
            patch(
                "genai_tk.agents.deer_flow.cli_commands._check_agent_sandbox_importable",
                return_value=False,
            ),
            pytest.raises(DockerSandboxError) as exc_info,
        ):
            _validate_docker_sandbox()

        error = exc_info.value
        assert len(error.reasons) == 1
        assert "agent-sandbox" in error.reasons[0]

    def test_validate_docker_sandbox_both_missing(self):
        """Error lists both reasons when Docker and agent-sandbox are missing."""
        with (
            patch(
                "genai_tk.agents.deer_flow.cli_commands._check_docker_available",
                return_value=False,
            ),
            patch(
                "genai_tk.agents.deer_flow.cli_commands._check_agent_sandbox_importable",
                return_value=False,
            ),
            pytest.raises(DockerSandboxError) as exc_info,
        ):
            _validate_docker_sandbox()

        error = exc_info.value
        assert len(error.reasons) == 2
        assert "Docker is not available" in error.reasons[0]
        assert "agent-sandbox" in error.reasons[1]

    def test_docker_sandbox_error_is_deer_flow_error(self):
        """DockerSandboxError is a DeerFlowError subclass (caught by CLI handler)."""
        from genai_tk.agents.deer_flow.profile import DeerFlowError

        error = DockerSandboxError(["test reason"])
        assert isinstance(error, DeerFlowError)
        assert "test reason" in str(error)
        assert "sandbox: local" in str(error)

    def test_check_agent_sandbox_importable_real(self):
        """agent-sandbox should be importable (we just installed it)."""
        assert _check_agent_sandbox_importable() is True
