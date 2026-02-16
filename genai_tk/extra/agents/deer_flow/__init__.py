"""Deer-flow integration for GenAI Toolkit.

This package provides a bridge between GenAI Toolkit and ByteDance's Deer-flow
agent system, enabling Deer-flow's advanced agent capabilities (subagents, sandboxing,
memory, skills) within the GenAI Toolkit ecosystem.

Architecture:
    - config_bridge: Generates Deer-flow config from GenAI Toolkit YAML configs
    - agent: Wrapper to create and run Deer-flow agents with GenAI Toolkit tools
    - cli_commands: CLI interface for running Deer-flow agents
    - The Streamlit UI lives in genai-blueprint/webapp/pages/demos/deer_flow_agent.py

Setup:
    1. make deer-flow-install   (clones deer-flow + installs deps)
    2. Or manually: git clone https://github.com/bytedance/deer-flow ext/deer-flow
       then: uv sync --group deerflow
"""

from genai_tk.extra.agents.deer_flow._path_setup import get_deer_flow_backend_path, setup_deer_flow_path
from genai_tk.extra.agents.deer_flow.agent import (
    DeerFlowAgentConfig,
    DeerFlowError,
    InvalidModeError,
    MCPServerNotFoundError,
    ProfileNotFoundError,
    create_deer_flow_agent_simple,
    load_deer_flow_profiles,
    validate_llm_identifier,
    validate_mcp_servers,
    validate_mode,
    validate_profile_name,
)

__all__ = [
    # Path setup
    "get_deer_flow_backend_path",
    "setup_deer_flow_path",
    # Agent creation
    "DeerFlowAgentConfig",
    "load_deer_flow_profiles",
    "create_deer_flow_agent_simple",
    # Validation functions
    "validate_profile_name",
    "validate_mode",
    "validate_mcp_servers",
    "validate_llm_identifier",
    # Exceptions
    "DeerFlowError",
    "ProfileNotFoundError",
    "InvalidModeError",
    "MCPServerNotFoundError",
]
