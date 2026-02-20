"""Tests for Deer-flow profile validation functions."""

import pytest

from genai_tk.extra.agents.deer_flow.profile import (
    DeerFlowProfile,
    InvalidModeError,
    MCPServerNotFoundError,
    ProfileNotFoundError,
    get_available_modes,
    get_available_profile_names,
    validate_mcp_servers,
    validate_mode,
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
    """Test successful mode validation."""
    assert validate_mode("flash") == "flash"
    assert validate_mode("thinking") == "thinking"
    assert validate_mode("pro") == "pro"
    assert validate_mode("ultra") == "ultra"

    # Case insensitive
    assert validate_mode("FLASH") == "flash"
    assert validate_mode("Thinking") == "thinking"


def test_validate_mode_invalid():
    """Test mode validation with invalid mode."""
    with pytest.raises(InvalidModeError) as exc_info:
        validate_mode("invalid_mode")

    error = exc_info.value
    assert error.mode == "invalid_mode"
    assert "flash" in error.valid_modes
    assert "thinking" in error.valid_modes


def test_validate_mcp_servers_empty():
    """Test MCP validation with empty list."""
    result = validate_mcp_servers([])
    assert result == []


def test_validate_mcp_servers_success(monkeypatch):
    """Test successful MCP server validation."""

    # Mock the config to return some servers
    class MockConfig:
        def get_nested(self, key, default):
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
        def get_nested(self, key, default):
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
