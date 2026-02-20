"""Config bridge: Generate Deer-flow configuration from GenAI Toolkit configs.

Translates GenAI Toolkit's YAML-based configuration (llm.yaml, mcp_servers.yaml)
into Deer-flow's ``config.yaml`` and ``extensions_config.json`` formats.

Write both files into the deer-flow backend directory **before** starting the
server so it picks them up on launch.

Example:
```python
from genai_tk.extra.agents.deer_flow.config_bridge import setup_deer_flow_config
config_path, ext_path = setup_deer_flow_config(
    mcp_server_names=["tavily-mcp"],
    config_dir="/path/to/deer-flow/backend",
)
```
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


def load_skills_from_directories(skill_directories: list[str]) -> list[str]:
    """Discover all skills under the given directories.

    Looks for ``public/`` and ``custom/`` sub-directories inside each directory
    and returns skill identifiers in the form ``category/skill-name``.

    Args:
        skill_directories: Paths to search (may contain ``${paths.project}``).

    Returns:
        List of skill identifiers, e.g. ``["public/deep-research", "custom/my-skill"]``.
    """
    from genai_tk.utils.config_mngr import global_config

    config = global_config().root
    skills = []

    for skill_dir_template in skill_directories:
        if "${paths.project}" in skill_dir_template:
            project_path = global_config().get_dir_path("paths.project")
            skill_dir_str = skill_dir_template.replace("${paths.project}", str(project_path))
        else:
            skill_dir_str = skill_dir_template

        skill_dir = Path(skill_dir_str)
        if not skill_dir.exists():
            logger.warning(f"Skill directory does not exist: {skill_dir}")
            continue

        for category in ["public", "custom"]:
            category_dir = skill_dir / category
            if not category_dir.exists():
                continue
            for skill_path in category_dir.iterdir():
                if skill_path.is_dir() and not skill_path.name.startswith("."):
                    skills.append(f"{category}/{skill_path.name}")

    trace_loading = OmegaConf.select(config, "deerflow.skills.trace_loading", default=True)
    if trace_loading and skills:
        logger.debug(f"Discovered {len(skills)} skills from directories")
    return skills


def generate_deer_flow_models(llm_config_path: str = "config/providers/llm.yaml") -> list[dict[str, Any]]:
    """Build Deer-flow model config list from GenAI Toolkit's LlmFactory.

    Only includes models whose provider API keys are available in the environment.

    Args:
        llm_config_path: Kept for backward compatibility; no longer used directly.

    Returns:
        List of Deer-flow model config dicts.
    """
    all_models = LlmFactory.known_list()
    models = []

    for model_info in all_models:
        model_id = model_info.id
        provider_name = model_info.provider
        model_name = model_info.model

        try:
            provider_info = model_info.get_provider_info()
        except ValueError:
            continue

        api_key_env_var = provider_info.api_key_env_var
        if api_key_env_var and not os.environ.get(api_key_env_var):
            continue

        model_config: dict[str, Any] = {
            "name": model_id,
            "display_name": model_id.replace("_", " ").title(),
            "use": provider_info.get_use_string(),
            "model": model_name,
            "max_tokens": model_info.max_tokens or 4096,
            "supports_vision": model_info.supports_vision,
        }

        if model_info.supports_thinking:
            model_config["supports_thinking"] = True
            model_config["when_thinking_enabled"] = {"extra_body": {"thinking": {"type": "enabled"}}}

        if api_key_env_var:
            model_config["api_key"] = f"${api_key_env_var}"

        if provider_info.api_base:
            # langchain-openai 1.x uses openai_api_base (not api_base)
            model_config["openai_api_base"] = provider_info.api_base

        if provider_name == "azure":
            parts = model_name.split("/")
            if len(parts) == 2:
                model_config["model"] = parts[0]
                model_config["api_version"] = parts[1]

        models.append(model_config)

    logger.debug(f"Generated {len(models)} Deer-flow model configs using LlmFactory")
    return models


def generate_extensions_config(
    mcp_server_names: list[str] | None = None,
    mcp_config_path: str = "config/mcp_servers.yaml",
) -> dict[str, Any]:
    """Build Deer-flow extensions_config from GenAI Toolkit's mcp_servers.yaml.

    Args:
        mcp_server_names: Server names to include (``None`` → all enabled servers).
        mcp_config_path: Path to the MCP servers YAML (kept for compatibility).

    Returns:
        Dict in Deer-flow ``extensions_config.json`` format.
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
            env = {k: v for k, v in config["env"].items() if k != "PATH"}
            if env:
                server_config["env"] = env
        if "url" in config:
            server_config["url"] = config["url"]
        mcp_servers[name] = server_config

    result = {"mcpServers": mcp_servers, "skills": {}}
    logger.debug(f"Generated extensions_config with {len(mcp_servers)} MCP servers")
    return result


def write_deer_flow_config(
    models: list[dict[str, Any]] | None = None,
    tool_groups: list[str] | None = None,
    config_dir: str | None = None,
    sandbox: str = "local",
) -> Path:
    """Write a complete Deer-flow config.yaml.

    Args:
        models: Model configs (auto-generated from LlmFactory if None).
        tool_groups: Tool groups to enable (default: ``["web"]``).
        config_dir: Directory for output. Defaults to a temp directory if None.
        sandbox: Sandbox provider: ``"local"`` (no Docker) or ``"docker"``
            (requires AioSandboxProvider image).

    Returns:
        Path to the written config.yaml.
    """
    from genai_tk.utils.config_mngr import global_config

    if models is None:
        models = generate_deer_flow_models()

    if not models:
        logger.warning("No models available for Deer-flow config — using placeholder.")
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

    config = global_config().root
    # Read skills directories from config (deerflow.skills.directories list)
    skills_dirs_cfg = OmegaConf.select(config, "deerflow.skills.directories")
    if skills_dirs_cfg:
        # Use the first configured directory as the deer-flow skills path
        skills_path_str = list(skills_dirs_cfg)[0]
        if "${paths.project}" in skills_path_str:
            project_path = global_config().get_dir_path("paths.project")
            skills_path_str = skills_path_str.replace("${paths.project}", str(project_path))
    else:
        skills_path_str = str(Path.cwd() / "skills")
    skills_container_path = OmegaConf.select(config, "deerflow.skills.container_path", default="/mnt/skills")
    trace_loading = OmegaConf.select(config, "deerflow.skills.trace_loading", default=True)

    skills_path = Path(skills_path_str).expanduser().resolve()

    if trace_loading:
        logger.debug(f"Deer-flow skills path: {skills_path}")
        if skills_path.exists():
            for category in ("public", "custom"):
                cat_dir = skills_path / category
                if cat_dir.exists():
                    names = [d.name for d in cat_dir.iterdir() if d.is_dir()]
                    label = "public" if category == "public" else "custom"
                    preview = ", ".join(sorted(names)[:5])
                    suffix = "..." if len(names) > 5 else ""
                    logger.debug(f"Available {label} skills: {preview}{suffix}")
        else:
            logger.warning(f"Skills directory not found: {skills_path}")

    # Build sandbox config based on requested provider
    if sandbox == "docker":
        sandbox_cfg: dict[str, Any] = {
            "use": "src.community.aio_sandbox.aio_sandbox_provider:AioSandboxProvider",
        }
    else:
        sandbox_cfg = {"use": "src.sandbox.local:LocalSandboxProvider"}

    cfg: dict[str, Any] = {
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
        "sandbox": sandbox_cfg,
        "skills": {
            "path": str(skills_path),
            "container_path": skills_container_path,
        },
        "title": {"enabled": False},
        "summarization": {"enabled": False},
        "memory": {"enabled": False},
    }

    if config_dir is None:
        config_dir = tempfile.mkdtemp(prefix="deer_flow_")

    config_path = Path(config_dir) / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    logger.debug(f"Wrote Deer-flow config to {config_path}")
    return config_path


def write_extensions_config(
    extensions: dict[str, Any] | None = None,
    config_dir: str | None = None,
    mcp_server_names: list[str] | None = None,
) -> Path:
    """Write Deer-flow extensions_config.json.

    Args:
        extensions: Extensions dict (auto-generated if None).
        config_dir: Output directory (defaults to temp dir).
        mcp_server_names: MCP servers to include when auto-generating.

    Returns:
        Path to the written extensions_config.json.
    """
    if extensions is None:
        extensions = generate_extensions_config(mcp_server_names)

    if config_dir is None:
        config_dir = tempfile.mkdtemp(prefix="deer_flow_")

    ext_path = Path(config_dir) / "extensions_config.json"
    with open(ext_path, "w") as f:
        json.dump(extensions, f, indent=2)

    logger.debug(f"Wrote Deer-flow extensions_config to {ext_path}")
    return ext_path


def setup_deer_flow_config(
    mcp_server_names: list[str] | None = None,
    enabled_skills: list[str] | None = None,
    skill_directories: list[str] | None = None,
    config_dir: str | None = None,
    sandbox: str = "local",
) -> tuple[Path, Path]:
    """Generate both Deer-flow config files in one call.

    Call this **before** starting the server so it reads the generated files on
    launch.  The files are written to ``<deer_flow_path>/backend`` by default
    (resolved from the ``DEER_FLOW_PATH`` environment variable).

    Args:
        mcp_server_names: MCP servers to enable (None → all enabled).
        enabled_skills: Explicit list of skills in ``category/name`` format.
        skill_directories: Directories to auto-discover skills from recursively.
        config_dir: Override output directory. Defaults to ``$DEER_FLOW_PATH/backend``.
        sandbox: Sandbox provider: ``"local"`` or ``"docker"``.

    Returns:
        Tuple of (config.yaml path, extensions_config.json path).
    """
    from genai_tk.utils.config_mngr import global_config

    if config_dir is None:
        deer_flow_path = os.environ.get("DEER_FLOW_PATH", "")
        if deer_flow_path:
            config_dir = str(Path(deer_flow_path).expanduser().resolve() / "backend")
        else:
            config_dir = tempfile.mkdtemp(prefix="deer_flow_")
            logger.warning(
                "DEER_FLOW_PATH not set — writing Deer-flow config to temp dir: %s. "
                "Set DEER_FLOW_PATH so configs persist across restarts.",
                config_dir,
            )

    config_path = write_deer_flow_config(config_dir=config_dir, sandbox=sandbox)

    extensions_config = generate_extensions_config(mcp_server_names)

    # Determine skills to enable
    skills_to_enable = enabled_skills or []
    if skill_directories:
        skills_to_enable = load_skills_from_directories(skill_directories)
    elif not enabled_skills:
        config = global_config().root
        default_dirs = OmegaConf.select(config, "deerflow.skills.directories")
        if default_dirs:
            skills_to_enable = load_skills_from_directories(list(default_dirs))

    if skills_to_enable:
        config = global_config().root
        trace_loading = OmegaConf.select(config, "deerflow.skills.trace_loading", default=True)
        if trace_loading:
            logger.debug(f"Enabling {len(skills_to_enable)} skills")

        skills_state: dict[str, Any] = {}
        for skill_spec in skills_to_enable:
            if "/" in skill_spec:
                category, skill_name = skill_spec.split("/", 1)
            else:
                category, skill_name = "public", skill_spec
            skills_state.setdefault(category, {})[skill_name] = True

        extensions_config["skills_state"] = skills_state

    ext_path = write_extensions_config(extensions=extensions_config, config_dir=config_dir)

    logger.debug("Deer-flow config ready: %s, %s", config_path, ext_path)
    return config_path, ext_path
