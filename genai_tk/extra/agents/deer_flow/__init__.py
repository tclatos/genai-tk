"""Deer-flow integration for GenAI Toolkit (HTTP client mode).

Connects to a running Deer-flow server via its HTTP API.
No in-process import of deer-flow is required.

Architecture:
    - profile:        DeerFlowProfile Pydantic model, profile loading
    - client:         Async HTTP client (LangGraph + Gateway APIs)
    - server_manager: Auto-start / stop of Deer-flow subprocesses
    - config_bridge:  Generate deer-flow config.yaml and extensions_config.json
    - cli_commands:   CLI interface (``cli deerflow``)
    - Streamlit UI:   genai-blueprint/webapp/pages/demos/deer_flow_agent.py

Quickstart:
    1. Set DEER_FLOW_PATH=/path/to/deer-flow
    2. ``cli deerflow --list``
    3. ``cli deerflow -p "Research Assistant" --chat``
"""

from genai_tk.extra.agents.deer_flow.client import DeerFlowClient, ErrorEvent, NodeEvent, TokenEvent
from genai_tk.extra.agents.deer_flow.config_bridge import setup_deer_flow_config
from genai_tk.extra.agents.deer_flow.profile import (
    DeerFlowError,
    DeerFlowProfile,
    InvalidModeError,
    MCPServerNotFoundError,
    ProfileNotFoundError,
    get_available_modes,
    get_available_profile_names,
    load_deer_flow_profiles,
    validate_mcp_servers,
    validate_mode,
    validate_profile_name,
)
from genai_tk.extra.agents.deer_flow.server_manager import DeerFlowServerManager

__all__ = [
    # Profile
    "DeerFlowProfile",
    "load_deer_flow_profiles",
    "get_available_modes",
    "get_available_profile_names",
    "validate_profile_name",
    "validate_mode",
    "validate_mcp_servers",
    # HTTP client
    "DeerFlowClient",
    "TokenEvent",
    "NodeEvent",
    "ErrorEvent",
    # Server lifecycle
    "DeerFlowServerManager",
    # Config generation
    "setup_deer_flow_config",
    # Exceptions
    "DeerFlowError",
    "ProfileNotFoundError",
    "InvalidModeError",
    "MCPServerNotFoundError",
]
