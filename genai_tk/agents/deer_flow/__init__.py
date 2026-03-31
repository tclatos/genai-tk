"""Deer-flow integration for GenAI Toolkit (embedded client mode).

Loads DeerFlow in-process via ``DEER_FLOW_PATH/backend`` — no LangGraph Server
or Gateway API processes are required for terminal usage (single-shot + chat).
The ``--web`` option still starts the backend servers for the Next.js frontend.

Architecture:
    - profile:         DeerFlowProfile Pydantic model, profile loading
    - embedded_client: In-process DeerFlow adapter + typed streaming events
    - server_manager:  Subprocess lifecycle (used only for ``--web``)
    - config_bridge:   Generate deer-flow config.yaml and extensions_config.json
    - cli_commands:    CLI interface (``cli deerflow``)
    - Streamlit UI:    genai-blueprint/webapp/pages/demos/deer_flow_agent.py

Quickstart:
    1. Set DEER_FLOW_PATH=/path/to/deer-flow
    2. ``cli deerflow --list``
    3. ``cli deerflow -p "Research Assistant" --chat``
"""

from genai_tk.agents.deer_flow.config_bridge import setup_deer_flow_config
from genai_tk.agents.deer_flow.embedded_client import (
    ClarificationEvent,
    EmbeddedDeerFlowClient,
    ErrorEvent,
    NodeEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from genai_tk.agents.deer_flow.profile import (
    DeerFlowError,
    DeerFlowMode,
    DeerFlowProfile,
    DeerFlowSandbox,
    DockerSandboxError,
    MCPServerNotFoundError,
    ProfileNotFoundError,
    get_available_modes,
    get_available_profile_names,
    load_deer_flow_profiles,
    validate_mcp_servers,
    validate_profile_name,
)
from genai_tk.agents.deer_flow.server_manager import DeerFlowServerManager

__all__ = [
    # Profile
    "DeerFlowProfile",
    "DeerFlowMode",
    "DeerFlowSandbox",
    "load_deer_flow_profiles",
    "get_available_modes",
    "get_available_profile_names",
    "validate_profile_name",
    "validate_mcp_servers",
    # Embedded client + events
    "EmbeddedDeerFlowClient",
    "TokenEvent",
    "NodeEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "ErrorEvent",
    "ClarificationEvent",
    # Server lifecycle (--web only)
    "DeerFlowServerManager",
    # Config generation
    "setup_deer_flow_config",
    # Exceptions
    "DeerFlowError",
    "DockerSandboxError",
    "ProfileNotFoundError",
    "MCPServerNotFoundError",
]
