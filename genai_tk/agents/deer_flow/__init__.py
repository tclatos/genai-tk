"""Deer-flow integration for GenAI Toolkit (embedded client mode).

Loads DeerFlow in-process via ``DEER_FLOW_PATH/backend`` — no LangGraph Server
or Gateway API processes are required.

Use ``--generate-config`` to export the DeerFlow-native config files and
launch instructions for the standard DeerFlow frontend/backend stack.

Architecture:
    - profile:         DeerFlowProfile Pydantic model, profile loading
    - embedded_client: In-process DeerFlow adapter + typed streaming events
    - config_bridge:   Generate deer-flow config.yaml and extensions_config.json
    - cli_commands:    CLI interface (``cli deerflow``)
    - Streamlit UI:    genai-blueprint/webapp/pages/demos/deer_flow_agent.py

Quickstart:
    1. Set DEER_FLOW_PATH=/path/to/deer-flow
    2. ``cli deerflow --list``
    3. ``cli deerflow -p "Research Assistant" --chat``
"""

# Lazy import: Defer expensive deer-flow modules (profile, config_bridge, embedded_client)
# until actually needed. These bring in llm_factory, spacy, embeddings, and other heavy deps.
# CLI startup only needs the cli_commands module which doesn't require these.

_loaded_modules = {}


def __getattr__(name: str):
    """Lazy-load deer-flow modules on first access."""
    global _loaded_modules

    # Map of attributes to their module origins
    _attr_to_module = {
        # Profile and related
        "DeerFlowProfile": "genai_tk.agents.deer_flow.profile",
        "DeerFlowMode": "genai_tk.agents.deer_flow.profile",
        "DeerFlowSandbox": "genai_tk.agents.deer_flow.profile",
        "load_deer_flow_profiles": "genai_tk.agents.deer_flow.profile",
        "get_available_modes": "genai_tk.agents.deer_flow.profile",
        "get_available_profile_names": "genai_tk.agents.deer_flow.profile",
        "validate_profile_name": "genai_tk.agents.deer_flow.profile",
        "validate_mcp_servers": "genai_tk.agents.deer_flow.profile",
        "DeerFlowError": "genai_tk.agents.deer_flow.profile",
        "DockerSandboxError": "genai_tk.agents.deer_flow.profile",
        "ProfileNotFoundError": "genai_tk.agents.deer_flow.profile",
        "MCPServerNotFoundError": "genai_tk.agents.deer_flow.profile",
        # Embedded client and events
        "EmbeddedDeerFlowClient": "genai_tk.agents.deer_flow.embedded_client",
        "TokenEvent": "genai_tk.agents.deer_flow.embedded_client",
        "NodeEvent": "genai_tk.agents.deer_flow.embedded_client",
        "ToolCallEvent": "genai_tk.agents.deer_flow.embedded_client",
        "ToolResultEvent": "genai_tk.agents.deer_flow.embedded_client",
        "ErrorEvent": "genai_tk.agents.deer_flow.embedded_client",
        "ClarificationEvent": "genai_tk.agents.deer_flow.embedded_client",
        # Config generation
        "setup_deer_flow_config": "genai_tk.agents.deer_flow.config_bridge",
    }

    if name not in _attr_to_module:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name = _attr_to_module[name]

    # Import module if not already cached
    if module_name not in _loaded_modules:
        import importlib

        _loaded_modules[module_name] = importlib.import_module(module_name)

    module = _loaded_modules[module_name]

    # Return the attribute from the module
    if hasattr(module, name):
        return getattr(module, name)

    raise AttributeError(f"module {module_name!r} has no attribute {name!r}")


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
    # Config generation
    "setup_deer_flow_config",
    # Exceptions
    "DeerFlowError",
    "DockerSandboxError",
    "ProfileNotFoundError",
    "MCPServerNotFoundError",
]
