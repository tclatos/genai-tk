"""Pydantic configuration models for genai-tk MCP server exposure.

Defines the schema for ``config/mcp/servers.yaml`` files that describe
which tools and agents should be exposed as MCP servers.

Example YAML:
    ```yaml
    mcp_expose_servers:
      - name: "search"
        description: "Web search tools exposed as MCP"
        tools:
          - factory: genai_tk.tools.langchain.search_tools_factory:create_search_function
            config:
              verbose: false
        agent:
          enabled: true
          name: run_search_agent
          description: "Run a full ReAct web-search agent"
          llm: gpt_41mini@openai
    ```
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf
from pydantic import BaseModel

from genai_tk.utils.config_mngr import global_config


class MCPToolConfig(BaseModel):
    """Configuration for a single tool factory entry in an MCP server.

    Mirrors the ``tools:`` list items in ``langchain.yaml``: every key except
    ``factory`` is passed as a keyword argument to the factory function.  A
    nested ``config`` key (as used by ``create_sql_tool_from_config``) is
    therefore just a regular kwarg named ``config``.

    Example:
        ```yaml
        # Flat kwargs (search tool)
        - factory: mod:create_search_function
          verbose: false
        # Nested config kwarg (SQL tool)
        - factory: mod:create_sql_tool_from_config
          config:
            database_uri: sqlite:///data.db
        ```
    """

    factory: str
    model_config = {"extra": "allow"}

    def factory_kwargs(self) -> dict:
        """Return all fields except ``factory`` as a flat kwargs dict."""
        return {k: v for k, v in self.model_dump().items() if k != "factory"}


class MCPAgentConfig(BaseModel):
    """Optional agent wrapper exposed as a single MCP tool.

    When ``enabled`` is true, the server also exposes a ``name`` tool
    that runs a full ReAct or DeepAgent and returns the final answer.
    """

    enabled: bool = True
    name: str = "run_agent"
    description: str = "Run the agent with a user query and return the final answer."
    llm: str | None = None
    profile: str | None = None


class MCPServerDefinition(BaseModel):
    """Full definition of a single MCP server to be exposed."""

    name: str
    description: str = ""
    tools: list[MCPToolConfig] = []
    agent: MCPAgentConfig | None = None


def load_mcp_server_definitions(config_path: Path | str | None = None) -> list[MCPServerDefinition]:
    """Load MCP server definitions from a YAML configuration file.

    Supports OmegaConf variable interpolation (e.g., ``${paths.project}``).

    Args:
        config_path: Path to the YAML file. Defaults to ``config/mcp/servers.yaml``
            relative to the project root.

    Returns:
        List of MCPServerDefinition instances.

    Example:
        ```python
        servers = load_mcp_server_definitions()
        for s in servers:
            print(s.name, [t.factory for t in s.tools])
        ```
    """
    if config_path is None:
        project_root = global_config().get_dir_path("paths.project")
        config_path = project_root / "config" / "mcp" / "servers.yaml"

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"MCP expose config not found at {path}. Create config/mcp/servers.yaml or pass an explicit path."
        )

    # Use OmegaConf for variable interpolation
    raw_cfg = OmegaConf.load(path)

    # Merge with global config so ${paths.project} etc. resolve
    try:
        merged = OmegaConf.merge(global_config().cfg, raw_cfg)
    except Exception as e:
        logger.warning(f"Could not merge with global config for interpolation: {e}")
        merged = raw_cfg

    raw_dict = OmegaConf.to_container(merged, resolve=True)

    entries = raw_dict.get("mcp_expose_servers", [])
    definitions = []
    for entry in entries:
        tools_raw = entry.get("tools", [])
        agent_raw = entry.get("agent")
        definitions.append(
            MCPServerDefinition(
                name=entry["name"],
                description=entry.get("description", ""),
                tools=[MCPToolConfig(**t) for t in tools_raw],
                agent=MCPAgentConfig(**agent_raw) if agent_raw else None,
            )
        )

    logger.info(f"Loaded {len(definitions)} MCP server definition(s) from {path}")
    return definitions


def get_mcp_server_definition(name: str, config_path: Path | str | None = None) -> MCPServerDefinition:
    """Return a single MCP server definition by name.

    Args:
        name: Server name as declared in the YAML.
        config_path: Optional path override.

    Returns:
        The matching MCPServerDefinition.
    """
    definitions = load_mcp_server_definitions(config_path)
    match = next((d for d in definitions if d.name == name), None)
    if match is None:
        available = [d.name for d in definitions]
        raise ValueError(f"MCP server '{name}' not found. Available: {available}")
    return match
