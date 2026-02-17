"""Tests for LLM configuration in shared_config_loader.

This module tests the ability to specify and resolve LLM identifiers
in React agent configurations from YAML files.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml
from upath import UPath

from genai_tk.tools.langchain.shared_config_loader import create_langchain_agent_config


class TestSharedConfigLoaderLLM:
    """Test LLM configuration handling in shared config loader."""

    def create_test_config_file(self, agents_config: list[Dict[str, Any]]) -> UPath:
        """Create a temporary YAML configuration file for testing."""
        config_data = {"langchain_agents": agents_config}

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(config_data, temp_file, default_flow_style=False)
        temp_file.flush()

        return UPath(temp_file.name)

    def test_agent_with_explicit_llm_id(self):
        """Test loading an agent with explicit LLM ID in configuration."""
        agents_config = [
            {
                "name": "test_agent",
                "llm": "parrot_local@fake",  # Use a known fake model from tests
                "tools": [],
                "mcp_servers": [],
                "examples": ["Test example"],
            }
        ]

        config_file = self.create_test_config_file(agents_config)

        try:
            agent_config = create_langchain_agent_config(
                config_file=config_file, config_section="langchain_agents", config_name="test_agent", llm=None
            )

            assert agent_config is not None
            assert agent_config.name == "test_agent"
            assert agent_config.llm_id == "parrot_local@fake"
            assert agent_config.examples == ["Test example"]

        finally:
            Path(config_file).unlink()  # Clean up temp file

    def test_agent_with_llm_tag(self):
        """Test loading an agent with LLM tag (resolves to full ID)."""
        agents_config = [
            {
                "name": "test_agent_tag",
                "llm": "fake",  # This should resolve to parrot_local@fake
                "tools": [],
                "mcp_servers": [],
                "examples": [],
            }
        ]

        config_file = self.create_test_config_file(agents_config)

        try:
            agent_config = create_langchain_agent_config(
                config_file=config_file, config_section="langchain_agents", config_name="test_agent_tag", llm=None
            )

            assert agent_config is not None
            assert agent_config.name == "test_agent_tag"
            assert agent_config.llm_id == "parrot_local@fake"

        finally:
            Path(config_file).unlink()

    def test_agent_without_llm(self):
        """Test loading an agent without LLM specification (should use default)."""
        agents_config = [
            {
                "name": "default_agent",
                # No "llm" key specified
                "tools": [],
                "mcp_servers": [],
                "examples": [],
            }
        ]

        config_file = self.create_test_config_file(agents_config)

        try:
            agent_config = create_langchain_agent_config(
                config_file=config_file, config_section="langchain_agents", config_name="default_agent", llm=None
            )

            assert agent_config is not None
            assert agent_config.name == "default_agent"
            assert agent_config.llm_id is None  # No LLM specified in config

        finally:
            Path(config_file).unlink()

    def test_agent_with_invalid_llm_identifier(self):
        """Test loading an agent with invalid LLM identifier."""
        agents_config = [
            {
                "name": "invalid_agent",
                "llm": "nonexistent_model_invalid_provider",
                "tools": [],
                "mcp_servers": [],
                "examples": [],
            }
        ]

        config_file = self.create_test_config_file(agents_config)

        try:
            agent_config = create_langchain_agent_config(
                config_file=config_file, config_section="langchain_agents", config_name="invalid_agent", llm=None
            )

            # Should still create config but with None llm_id due to error
            assert agent_config is not None
            assert agent_config.name == "invalid_agent"
            assert agent_config.llm_id is None  # Resolution failed

        finally:
            Path(config_file).unlink()

    def test_nonexistent_agent_configuration(self):
        """Test loading a non-existent agent configuration."""
        agents_config = [
            {"name": "existing_agent", "llm": "parrot_local@fake", "tools": [], "mcp_servers": [], "examples": []}
        ]

        config_file = self.create_test_config_file(agents_config)

        try:
            agent_config = create_langchain_agent_config(
                config_file=config_file,
                config_section="langchain_agents",
                config_name="nonexistent_agent",  # This agent doesn't exist
                llm=None,
            )

            assert agent_config is None

        finally:
            Path(config_file).unlink()

    def test_agent_with_pre_prompt_and_llm(self):
        """Test loading agent with both pre_prompt and LLM configuration."""
        agents_config = [
            {
                "name": "full_config_agent",
                "llm": "parrot_local@fake",
                "pre_prompt": "You are a helpful assistant.",
                "tools": [],
                "mcp_servers": ["filesystem"],
                "examples": ["Example 1", "Example 2"],
            }
        ]

        config_file = self.create_test_config_file(agents_config)

        try:
            agent_config = create_langchain_agent_config(
                config_file=config_file, config_section="langchain_agents", config_name="full_config_agent", llm=None
            )

            assert agent_config is not None
            assert agent_config.name == "full_config_agent"
            assert agent_config.llm_id == "parrot_local@fake"
            assert agent_config.pre_prompt == "You are a helpful assistant."
            assert agent_config.mcp_servers == ["filesystem"]
            assert agent_config.examples == ["Example 1", "Example 2"]

        finally:
            Path(config_file).unlink()

    def test_load_all_agents_with_mixed_llm_configs(self):
        """Test loading all agents with mixed LLM configurations."""
        agents_config = [
            {"name": "agent_with_id", "llm": "parrot_local@fake", "tools": [], "examples": []},
            {"name": "agent_with_tag", "llm": "fake", "tools": [], "examples": []},
            {
                "name": "agent_default",
                # No LLM specified
                "tools": [],
                "examples": [],
            },
        ]

        config_file = self.create_test_config_file(agents_config)

        try:
            from genai_tk.tools.langchain.shared_config_loader import load_all_langchain_agent_configs

            all_configs = load_all_langchain_agent_configs(
                config_file=str(config_file), config_section="langchain_agents"
            )

            assert len(all_configs) == 3

            # Find each agent config
            configs_by_name = {config.name: config for config in all_configs}

            assert "agent_with_id" in configs_by_name
            assert configs_by_name["agent_with_id"].llm_id == "parrot_local@fake"

            assert "agent_with_tag" in configs_by_name
            assert configs_by_name["agent_with_tag"].llm_id == "parrot_local@fake"  # Resolved from tag

            assert "agent_default" in configs_by_name
            assert configs_by_name["agent_default"].llm_id is None  # No LLM specified

        finally:
            Path(config_file).unlink()
