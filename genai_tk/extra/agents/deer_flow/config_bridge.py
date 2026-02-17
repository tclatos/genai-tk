"""Config bridge: Generate Deer-flow configuration from GenAI Toolkit configs.

Translates GenAI Toolkit's YAML-based configuration (llm.yaml, mcp_servers.yaml)
into Deer-flow's config.yaml and extensions_config.json formats.

This runs at agent creation time, writing temporary config files that Deer-flow reads.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from omegaconf import OmegaConf

from genai_tk.core.llm_factory import LlmFactory
from genai_tk.extra.agents.deer_flow._path_setup import get_deer_flow_backend_path


def load_skills_from_directories(skill_directories: list[str]) -> list[str]:
    """Load all skills from specified directories recursively.

    Discovers skills in 'public/' and 'custom/' subdirectories within each directory.
    Returns skills in format: 'category/skill-name' (e.g., 'public/deep-research').

    Args:
        skill_directories: List of directories to search for skills

    Returns:
        List of skill identifiers in format 'category/skill-name'
    """
    from genai_tk.utils.config_mngr import global_config

    skills = []
    config = global_config().root

    for skill_dir_template in skill_directories:
        # Resolve path variables using global config
        if "${paths.project}" in skill_dir_template:
            project_path = global_config().get_dir_path("paths.project")
            skill_dir_str = skill_dir_template.replace("${paths.project}", str(project_path))
        else:
            skill_dir_str = skill_dir_template

        skill_dir = Path(skill_dir_str)

        if not skill_dir.exists():
            logger.warning(f"Skill directory does not exist: {skill_dir}")
            continue

        # Look for 'public' and 'custom' subdirectories
        for category in ["public", "custom"]:
            category_dir = skill_dir / category
            if not category_dir.exists():
                continue

            # Get all skill directories (non-hidden directories)
            for skill_path in category_dir.iterdir():
                if skill_path.is_dir() and not skill_path.name.startswith("."):
                    skill_name = skill_path.name
                    skill_id = f"{category}/{skill_name}"
                    skills.append(skill_id)

    # Log discovered skills
    trace_loading = OmegaConf.select(config, "deerflow.skills.trace_loading", default=True)
    if trace_loading and skills:
        logger.info(f"Discovered {len(skills)} skills from directories: {', '.join(skills)}")

    return skills


def generate_deer_flow_models(llm_config_path: str = "config/providers/llm.yaml") -> list[dict[str, Any]]:
    """Generate Deer-flow model configs from GenAI Toolkit's llm.yaml.

    Uses LlmFactory to get model information and provider configuration.
    Only includes models whose provider API keys are available in the environment.

    Args:
        llm_config_path: Path to the LLM config YAML file (kept for backward compatibility)

    Returns:
        List of Deer-flow model config dicts
    """
    # Get all known models from LlmFactory
    all_models = LlmFactory.known_list()

    # Also load capabilities from YAML since LlmInfo doesn't expose them
    config_path = Path(llm_config_path)
    capabilities_map = {}
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f)
            llm_entries = raw.get("llm", [])
            for entry in llm_entries:
                if entry and entry.get("id"):
                    capabilities_map[entry["id"]] = entry.get("capabilities", [])

    models = []

    for model_info in all_models:
        model_id = model_info.id
        provider_name = model_info.provider
        model_name = model_info.model

        # Get provider info from the centralized PROVIDER_INFO
        try:
            provider_info = model_info.get_provider_info()
        except ValueError:
            # Skip models with unknown providers
            continue

        # Check if API key is available
        api_key_env_var = provider_info.api_key_env_var
        if api_key_env_var and not os.environ.get(api_key_env_var):
            continue

        # Extract capabilities from YAML
        capabilities = capabilities_map.get(model_id, [])
        supports_vision = "vision" in capabilities
        supports_thinking = "thinking" in capabilities

        # Build model config using provider info
        model_config: dict[str, Any] = {
            "name": model_id,
            "display_name": model_id.replace("_", " ").title(),
            "use": provider_info.get_use_string(),
            "model": model_name,
            "max_tokens": 4096,
            "supports_vision": supports_vision,
        }

        if supports_thinking:
            model_config["supports_thinking"] = True

        if api_key_env_var:
            model_config["api_key"] = f"${api_key_env_var}"

        # Add base URL from provider info
        if provider_info.api_base:
            model_config["api_base"] = provider_info.api_base

        # Azure-specific handling
        if provider_name == "azure":
            parts = model_name.split("/")
            if len(parts) == 2:
                model_config["model"] = parts[0]
                model_config["api_version"] = parts[1]

        models.append(model_config)

    logger.info(f"Generated {len(models)} Deer-flow model configs using LlmFactory")
    return models


def generate_extensions_config(
    mcp_server_names: list[str] | None = None,
    mcp_config_path: str = "config/mcp_servers.yaml",
) -> dict[str, Any]:
    """Generate Deer-flow extensions_config.json from GenAI Toolkit's mcp_servers.yaml.

    Args:
        mcp_server_names: Optional list of server names to include (None = all enabled)
        mcp_config_path: Path to the MCP servers YAML config

    Returns:
        Dict in Deer-flow extensions_config.json format
    """
    from genai_tk.core.mcp_client import get_mcp_servers_dict

    try:
        servers_dict = get_mcp_servers_dict(mcp_server_names)
    except Exception as e:
        logger.warning(f"Could not load MCP servers: {e}")
        return {"mcpServers": {}, "skills": {}}

    mcp_servers = {}
    for name, config in servers_dict.items():
        server_config: dict[str, Any] = {
            "enabled": True,
            "type": config.get("transport", "stdio"),
        }
        if "command" in config:
            server_config["command"] = config["command"]
        if "args" in config:
            server_config["args"] = config["args"]
        if "env" in config:
            # Filter out PATH from env (too long, not needed)
            env = {k: v for k, v in config["env"].items() if k != "PATH"}
            if env:
                server_config["env"] = env
        if "url" in config:
            server_config["url"] = config["url"]

        mcp_servers[name] = server_config

    result = {"mcpServers": mcp_servers, "skills": {}}
    logger.info(f"Generated extensions_config with {len(mcp_servers)} MCP servers")
    return result


def write_deer_flow_config(
    models: list[dict[str, Any]] | None = None,
    tool_groups: list[str] | None = None,
    config_dir: str | None = None,
) -> Path:
    """Write a complete Deer-flow config.yaml to a temporary or specified directory.

    Args:
        models: Model configs (from generate_deer_flow_models). If None, auto-generates.
        tool_groups: Tool groups to enable (default: ["web"])
        config_dir: Directory to write config files. If None, uses a temp directory.

    Returns:
        Path to the generated config.yaml
    """
    from genai_tk.utils.config_mngr import global_config

    if models is None:
        models = generate_deer_flow_models()

    if not models:
        logger.warning("No models available for Deer-flow config. Using a placeholder.")
        models = [
            {
                "name": "default",
                "display_name": "Default Model",
                "use": "langchain_openai:ChatOpenAI",
                "model": "gpt-4",
                "api_key": "$OPENAI_API_KEY",
                "max_tokens": 4096,
                "supports_vision": False,
            }
        ]

    if tool_groups is None:
        tool_groups = ["web"]

    # Get skills configuration from genai-tk config
    config = global_config().root
    skills_path = OmegaConf.select(config, "deerflow.skills.path", default=str(Path.cwd() / "skills"))
    skills_container_path = OmegaConf.select(config, "deerflow.skills.container_path", default="/mnt/skills")
    trace_loading = OmegaConf.select(config, "deerflow.skills.trace_loading", default=True)

    # Resolve skills path to absolute
    skills_path = Path(skills_path).expanduser().resolve()

    if trace_loading:
        logger.info(f"Deer-flow skills path: {skills_path}")
        if skills_path.exists():
            public_skills = skills_path / "public"
            custom_skills = skills_path / "custom"
            if public_skills.exists():
                public_skill_names = [d.name for d in public_skills.iterdir() if d.is_dir()]
                logger.info(
                    f"Available public skills: {', '.join(sorted(public_skill_names)[:5])}{'...' if len(public_skill_names) > 5 else ''}"
                )
            if custom_skills.exists():
                custom_skill_names = [d.name for d in custom_skills.iterdir() if d.is_dir()]
                if custom_skill_names:
                    logger.info(f"Available custom skills: {', '.join(custom_skill_names)}")
        else:
            logger.warning(f"Skills directory not found: {skills_path}")

    # Build the config dict
    config = {
        "models": models,
        "tool_groups": [{"name": g} for g in tool_groups],
        "tools": [
            {
                "name": "web_search",
                "group": "web",
                "use": "src.community.tavily.tools:web_search_tool",
                "max_results": 5,
            },
            {
                "name": "web_fetch",
                "group": "web",
                "use": "src.community.jina_ai.tools:web_fetch_tool",
                "timeout": 10,
            },
        ],
        "sandbox": {"use": "src.sandbox.local:LocalSandboxProvider"},
        "skills": {
            "path": str(skills_path),
            "container_path": skills_container_path,
        },
        "title": {"enabled": False},
        "summarization": {"enabled": False},
        "memory": {"enabled": False},
    }

    # Write to file
    if config_dir is None:
        config_dir = tempfile.mkdtemp(prefix="deer_flow_")

    config_path = Path(config_dir) / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Wrote Deer-flow config to {config_path}")
    return config_path


def write_extensions_config(
    extensions: dict[str, Any] | None = None,
    config_dir: str | None = None,
    mcp_server_names: list[str] | None = None,
) -> Path:
    """Write Deer-flow extensions_config.json to a directory.

    Args:
        extensions: Extensions config dict. If None, auto-generates from mcp_servers.yaml.
        config_dir: Directory to write file. If None, uses temp directory.
        mcp_server_names: MCP server names to include (used if extensions is None).

    Returns:
        Path to the generated extensions_config.json
    """
    if extensions is None:
        extensions = generate_extensions_config(mcp_server_names)

    if config_dir is None:
        config_dir = tempfile.mkdtemp(prefix="deer_flow_")

    ext_path = Path(config_dir) / "extensions_config.json"
    with open(ext_path, "w") as f:
        json.dump(extensions, f, indent=2)

    logger.info(f"Wrote Deer-flow extensions_config to {ext_path}")
    return ext_path


def setup_deer_flow_config(
    mcp_server_names: list[str] | None = None,
    enabled_skills: list[str] | None = None,
    skill_directories: list[str] | None = None,
    config_dir: str | None = None,
) -> tuple[Path, Path]:
    """One-call setup: generates both Deer-flow config files and sets env vars.

    Args:
        mcp_server_names: MCP servers to enable
        enabled_skills: Skills to enable (format: "category/skill-name" or "skill-name" for public)
                       Deprecated: use skill_directories instead
        skill_directories: Directories to load skills from recursively
        config_dir: Directory for config files (default: temp dir inside deer-flow backend)

    Returns:
        Tuple of (config.yaml path, extensions_config.json path)
    """
    from genai_tk.utils.config_mngr import global_config

    if config_dir is None:
        backend_path = get_deer_flow_backend_path()
        config_dir = str(backend_path.parent)

    config_path = write_deer_flow_config(config_dir=config_dir)

    # Generate extensions config with MCP servers
    extensions_config = generate_extensions_config(mcp_server_names)

    # Determine which skills to enable
    skills_to_enable = enabled_skills or []

    # If skill_directories is provided, load all skills from those directories
    if skill_directories:
        discovered_skills = load_skills_from_directories(skill_directories)
        skills_to_enable = discovered_skills
    # Fall back to global config if neither is specified
    elif not enabled_skills:
        config = global_config().root
        default_dirs = OmegaConf.select(config, "deerflow.skills.directories")
        if default_dirs:
            discovered_skills = load_skills_from_directories(default_dirs)
            skills_to_enable = discovered_skills

    # Add skills state configuration
    if skills_to_enable:
        config = global_config().root
        trace_loading = OmegaConf.select(config, "deerflow.skills.trace_loading", default=True)
        if trace_loading:
            logger.info(f"Enabling {len(skills_to_enable)} skills")

        skills_state = {}
        for skill_spec in skills_to_enable:
            # Parse "category/skill-name" or just "skill-name" (defaults to public)
            if "/" in skill_spec:
                category, skill_name = skill_spec.split("/", 1)
            else:
                category, skill_name = "public", skill_spec

            # Create nested dict structure
            if category not in skills_state:
                skills_state[category] = {}
            skills_state[category][skill_name] = True

        extensions_config["skills_state"] = skills_state

    ext_path = write_extensions_config(extensions=extensions_config, config_dir=config_dir)

    # Set env vars for Deer-flow to find the configs
    os.environ["DEER_FLOW_CONFIG_PATH"] = str(config_path)
    os.environ["DEER_FLOW_EXTENSIONS_CONFIG_PATH"] = str(ext_path)

    logger.info(f"Deer-flow config ready: {config_path}, {ext_path}")
    return config_path, ext_path
