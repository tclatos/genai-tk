"""Config bridge: Generate Deer-flow configuration from GenAI Toolkit configs.

Translates GenAI Toolkit's YAML-based configuration (llm.yaml, mcp_servers.yaml)
into Deer-flow's ``config.yaml`` and ``extensions_config.json`` formats.

Write both files into the deer-flow backend directory **before** starting the
server so it picks them up on launch.

Example:
```python
from genai_tk.agents.deer_flow.config_bridge import setup_deer_flow_config

config_path, ext_path, warnings = setup_deer_flow_config(
    mcp_server_names=["tavily-mcp"],
    config_dir="/path/to/deer-flow/backend",
)
```
"""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from omegaconf import OmegaConf

from genai_tk.core.llm_factory import LlmFactory
from genai_tk.core.providers import PROVIDER_INFO
from genai_tk.utils.import_utils import get_module_from_qualified, get_object_name_from_qualified


@dataclass
class ConfigSetupWarnings:
    """Warnings collected during Deer-flow config setup."""

    missing_skill_directories: list[str] = field(default_factory=list)
    external_symlinks: list[str] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        """Return True if any warnings were collected."""
        return bool(self.missing_skill_directories or self.external_symlinks)


def _to_colon_notation(qualified_name: str) -> str:
    """Convert dot-notation qualified name to colon-notation for deer-flow.

    Deer-flow's resolver expects ``module.path:ClassName`` but GenAI Toolkit
    uses ``module.path.ClassName`` internally.
    """
    module = get_module_from_qualified(qualified_name)
    obj = get_object_name_from_qualified(qualified_name)
    return f"{module}:{obj}"


def _deer_flow_module_prefix() -> str:
    """Return the Python module prefix for deer-flow tool/sandbox imports.

    Modern deer-flow (post-harness refactor) uses ``deerflow.`` while the
    legacy layout uses ``src.``.
    """
    df_path = os.environ.get("DEER_FLOW_PATH", "")
    if df_path and (Path(df_path) / "backend" / "packages" / "harness").exists():
        return "deerflow"
    return "src"


def _check_external_symlinks(skills_path: Path, sandbox: str, warnings: ConfigSetupWarnings | None = None) -> None:
    """Warn when skills path contains symlinks to targets outside the directory.

    Docker bind-mounts cannot follow symlinks whose targets live outside the
    mounted tree, so any ``public/`` or ``custom/`` entry that is a symlink to
    an external path will appear as a broken link inside the container.

    Args:
        skills_path: Resolved skills root directory.
        sandbox: Sandbox type (warning only relevant for ``"docker"``).
        warnings: Optional ConfigSetupWarnings object to collect warnings into.
    """
    if sandbox != "docker":
        return
    external: list[str] = []
    for category in ("public", "custom"):
        cat_dir = skills_path / category
        if not cat_dir.exists():
            continue
        for item in cat_dir.iterdir():
            if item.is_symlink():
                try:
                    item.resolve().relative_to(skills_path)
                except ValueError:
                    external.append(f"{category}/{item.name}")
    if external:
        shown = ", ".join(external[:5])
        suffix = f" … ({len(external)} total)" if len(external) > 5 else f" ({len(external)} total)"
        msg = (
            f"Docker sandbox: skills dir '{skills_path}' contains symlinks to external paths: "
            f"{shown}{suffix}. "
            "Docker cannot follow these — those skill files will return 404 in the container. "
            "Fix: set 'deerflow.skills.directories[0]' to a directory with real files "
            "(e.g. '${paths.project}/ext/deer-flow/skills')."
        )
        logger.warning(msg)
        if warnings is not None:
            warnings.external_symlinks.append(msg)


def load_skills_from_directories(
    skill_directories: list[str], warnings: ConfigSetupWarnings | None = None
) -> list[str]:
    """Discover all skills under the given directories.

    Looks for ``public/`` and ``custom/`` sub-directories inside each directory
    and returns skill identifiers in the form ``category/skill-name``.

    Args:
        skill_directories: Paths to search (may contain ``${paths.project}``).
        warnings: Optional ConfigSetupWarnings object to collect warnings into.

    Returns:
        List of skill identifiers, e.g. ``["public/deep-research", "custom/my-skill"]``.
    """
    from genai_tk.utils.config_mngr import get_raw_config, paths_config

    config = get_raw_config()
    skills = []

    for skill_dir_template in skill_directories:
        if "${paths.project}" in skill_dir_template:
            project_path = paths_config().project
            skill_dir_str = skill_dir_template.replace("${paths.project}", str(project_path))
        else:
            skill_dir_str = skill_dir_template

        skill_dir = Path(skill_dir_str)
        if not skill_dir.exists():
            msg = f"Skill directory does not exist: {skill_dir}"
            logger.warning(msg)
            if warnings is not None:
                warnings.missing_skill_directories.append(msg)
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


def _build_dynamic_model_entry(canonical_llm_id: str) -> dict[str, Any] | None:
    """Build a minimal DeerFlow model config dict for a canonically-resolved LLM ID.

    Used when the ID is not in llm.yaml but has been resolved via models.dev
    (e.g. ``openai/gpt-oss-120b@openrouter``).

    Args:
        canonical_llm_id: Canonical LLM ID in ``model@provider`` format.

    Returns:
        DeerFlow model config dict, or ``None`` if the provider is unknown or
        its API key is missing from the environment.
    """
    model_part, _, provider_name = canonical_llm_id.rpartition("@")
    if not provider_name:
        return None
    provider_info = PROVIDER_INFO.get(provider_name)
    if not provider_info:
        return None
    api_key_env_var = provider_info.api_key_env_var
    if api_key_env_var and not os.environ.get(api_key_env_var):
        logger.debug(f"Skipping dynamic model '{canonical_llm_id}': {api_key_env_var} not set")
        return None

    entry: dict[str, Any] = {
        "name": canonical_llm_id,
        "display_name": model_part,
        "use": _to_colon_notation(provider_info.get_use_string()),
        "model": model_part,
        "max_tokens": 4096,
        "supports_vision": False,
    }
    if api_key_env_var:
        entry["api_key"] = f"${api_key_env_var}"
    if provider_info.api_base:
        entry["openai_api_base"] = provider_info.api_base
    return entry


def generate_deer_flow_models(
    llm_config_path: str = "config/providers/llm.yaml",
    selected_llm_id: str | None = None,
) -> list[dict[str, Any]]:
    """Build Deer-flow model config list from GenAI Toolkit's LlmFactory.

    Uses ``known_items_dict()`` which merges llm.yaml exceptions with models.dev
    registry entries.  For gateway providers (openrouter, github, etc.) whose
    models aren't enumerated in the registry, a dynamic entry is built from
    ``PROVIDER_INFO`` and the resolved canonical ID.

    When ``selected_llm_id`` is provided only that one model is returned.

    Args:
        llm_config_path: Kept for backward compatibility; no longer used directly.
        selected_llm_id: Resolved GenAI-tk model ID to include (None → all available).

    Returns:
        List of Deer-flow model config dicts.
    """
    all_items = LlmFactory.known_items_dict()
    if selected_llm_id:
        if selected_llm_id in all_items:
            all_items = {selected_llm_id: all_items[selected_llm_id]}
        else:
            # Gateway models (openrouter etc.) aren't enumerated in known_items_dict.
            # Build a dynamic entry from PROVIDER_INFO + the resolved canonical ID.
            dynamic = _build_dynamic_model_entry(selected_llm_id)
            if dynamic:
                return [dynamic]
            logger.warning(f"Model '{selected_llm_id}' not found in known models and cannot be built dynamically")
            return []
    models = []

    for model_info in all_items.values():
        model_id = model_info.id
        provider_name = model_info.provider
        model_name = model_info.model

        # Fake/test-only providers don't support bind_tools and are useless in DeerFlow
        if provider_name == "fake":
            continue

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
            "use": _to_colon_notation(provider_info.get_use_string()),
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
    selected_llm: str | None = None,
    skills_path: str | None = None,
    warnings: ConfigSetupWarnings | None = None,
) -> Path:
    """Write a complete Deer-flow config.yaml.

    Args:
        models: Model configs (auto-generated from LlmFactory if None).
        tool_groups: Tool groups to enable (default: all standard groups).
        config_dir: Directory for output. Defaults to a temp directory if None.
        sandbox: Sandbox provider: ``"local"`` (no Docker) or ``"docker"``
            (requires AioSandboxProvider image).
        selected_llm: Resolved GenAI-tk model ID; when set only that model is written.
        skills_path: Explicit skills root directory to mount in the sandbox.  When
            provided this takes precedence over ``deerflow.skills.directories`` in
            the global config (which is not merged at startup).
        warnings: Optional ConfigSetupWarnings object to collect warnings into.

    Returns:
        Path to the written config.yaml.
    """
    from genai_tk.utils.config_mngr import get_raw_config, paths_config

    if warnings is None:
        warnings = ConfigSetupWarnings()

    if models is None:
        models = generate_deer_flow_models(selected_llm_id=selected_llm)

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
        tool_groups = ["web", "file:read", "file:write", "bash"]

    config = get_raw_config()
    # Resolve skills path — prefer the explicitly passed value, then fall back to
    # deerflow.skills.directories in the global config (NOTE: agents/deerflow.yaml
    # is loaded on-demand and is NOT merged into get_raw_config(), so the config
    # lookup returns None unless the caller passes the resolved value explicitly).
    if skills_path is not None:
        skills_path_str = skills_path
    else:
        skills_dirs_cfg = OmegaConf.select(config, "deerflow.skills.directories")
        if skills_dirs_cfg:
            skills_path_str = list(skills_dirs_cfg)[0]
            if "${paths.project}" in skills_path_str:
                project_path = paths_config().project
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
            msg = f"Skills directory not found: {skills_path}"
            logger.warning(msg)
            warnings.missing_skill_directories.append(msg)

    _check_external_symlinks(skills_path, sandbox, warnings=warnings)

    pfx = _deer_flow_module_prefix()

    # Build sandbox config based on requested provider
    if sandbox == "docker":
        from genai_tk.agents.sandbox.config import get_docker_aio_settings

        aio = get_docker_aio_settings()
        sandbox_cfg: dict[str, Any] = {
            "use": f"{pfx}.community.aio_sandbox.aio_sandbox_provider:AioSandboxProvider",
            "image": aio.image,
            "opensandbox_server_url": aio.opensandbox_server_url,
        }
    else:
        # Local sandbox: map /mnt/user-data → CWD so the agent's virtual
        # paths resolve to the actual working directory instead of the
        # non-existent Docker mount point.
        sandbox_cfg = {
            "use": f"{pfx}.sandbox.local:LocalSandboxProvider",
            "mounts": [
                {
                    "host_path": str(Path.cwd()),
                    "container_path": "/mnt/user-data",
                    "read_only": False,
                },
            ],
        }

    cfg: dict[str, Any] = {
        "models": models,
        "tool_groups": [{"name": g} for g in tool_groups],
        "tools": [
            {
                "name": "web_search",
                "group": "web",
                "use": f"{pfx}.community.tavily.tools:web_search_tool",
                "max_results": 5,
            },
            {
                "name": "web_fetch",
                "group": "web",
                "use": f"{pfx}.community.jina_ai.tools:web_fetch_tool",
                "timeout": 10,
            },
            {
                "name": "image_search",
                "group": "web",
                "use": f"{pfx}.community.image_search.tools:image_search_tool",
                "max_results": 5,
            },
            {
                "name": "ls",
                "group": "file:read",
                "use": f"{pfx}.sandbox.tools:ls_tool",
            },
            {
                "name": "read_file",
                "group": "file:read",
                "use": f"{pfx}.sandbox.tools:read_file_tool",
            },
            {
                "name": "write_file",
                "group": "file:write",
                "use": f"{pfx}.sandbox.tools:write_file_tool",
            },
            {
                "name": "str_replace",
                "group": "file:write",
                "use": f"{pfx}.sandbox.tools:str_replace_tool",
            },
            {
                "name": "bash",
                "group": "bash",
                "use": f"{pfx}.sandbox.tools:bash_tool",
            },
        ],
        "sandbox": sandbox_cfg,
        "skills": {
            "path": str(skills_path),
            "container_path": skills_container_path,
        },
    }

    # Merge title / summarization / memory from deerflow.general in config
    general = OmegaConf.select(config, "deerflow.general")
    if general:
        general_dict = OmegaConf.to_container(general, resolve=True)
        for key in ("title", "summarization", "memory"):
            if key in general_dict:
                cfg[key] = general_dict[key]
    else:
        # Sensible fallbacks when general section is absent
        cfg["title"] = {"enabled": True, "max_words": 6, "max_chars": 60, "model_name": None}
        cfg["summarization"] = {"enabled": False}
        cfg["memory"] = {"enabled": False}

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
    selected_llm: str | None = None,
    warnings: ConfigSetupWarnings | None = None,
) -> tuple[Path, Path, ConfigSetupWarnings]:
    """Generate both Deer-flow config files in one call.

    Call this **before** starting the server so it reads the generated files on
    launch.  ``config.yaml`` is written to both ``<deer_flow_path>`` (root) and
    ``<deer_flow_path>/backend``; ``extensions_config.json`` is written to
    ``<deer_flow_path>/backend`` only.

    Args:
        mcp_server_names: MCP servers to enable (None → all enabled).
        enabled_skills: Explicit list of skills in ``category/name`` format.
        skill_directories: Directories to auto-discover skills from recursively.
        config_dir: Override output directory. Defaults to ``$DEER_FLOW_PATH/backend``.
        sandbox: Sandbox provider: ``"local"`` or ``"docker"``.
        selected_llm: Resolved GenAI-tk model ID; when set only that model is written.
        warnings: Optional ConfigSetupWarnings object to collect warnings into.

    Returns:
        Tuple of (config.yaml path in backend, extensions_config.json path, warnings).
    """
    from genai_tk.utils.config_mngr import get_raw_config

    if warnings is None:
        warnings = ConfigSetupWarnings()

    # Resolve DEER_FLOW_PATH unconditionally — used both for output dir and skills fallback.
    deer_flow_root: Path | None = None
    _deer_flow_env = os.environ.get("DEER_FLOW_PATH", "").strip()
    if _deer_flow_env:
        deer_flow_root = Path(_deer_flow_env).expanduser().resolve()

    if config_dir is None:
        if deer_flow_root is not None:
            config_dir = str(deer_flow_root / "backend")
        else:
            config_dir = tempfile.mkdtemp(prefix="deer_flow_")
            logger.warning(
                "DEER_FLOW_PATH not set — writing Deer-flow config to temp dir: %s. "
                "Set DEER_FLOW_PATH so configs persist across restarts.",
                config_dir,
            )

    # Resolve the first skills directory so write_deer_flow_config can use it as the
    # Docker volume mount path.  agents/deerflow.yaml is NOT merged into get_raw_config(),
    # so the path must be threaded through explicitly.
    resolved_skills_path: str | None = None
    effective_skill_dirs = skill_directories or []
    if not effective_skill_dirs:
        from genai_tk.utils.config_mngr import get_raw_config as _get_raw

        _cfg = _get_raw()
        _default_dirs = OmegaConf.select(_cfg, "deerflow.skills.directories")
        if _default_dirs:
            effective_skill_dirs = list(_default_dirs)
    if effective_skill_dirs:
        _first = effective_skill_dirs[0]
        if "${paths.project}" in _first:
            from genai_tk.utils.config_mngr import paths_config as _paths

            _first = _first.replace("${paths.project}", str(_paths().project))
        candidate = Path(_first).expanduser().resolve()
        if not candidate.exists() and deer_flow_root is not None:
            # Config default points to ${paths.project}/ext/deer-flow/skills which only
            # exists inside the genai-tk repo itself.  When running from any other
            # directory fall back to $DEER_FLOW_PATH/skills (the skills bundled with
            # the user's deer-flow clone).
            deer_flow_skills = deer_flow_root / "skills"
            if deer_flow_skills.exists():
                logger.debug(
                    f"Configured skills dir '{candidate}' not found, "
                    f"falling back to DEER_FLOW_PATH/skills: {deer_flow_skills}"
                )
                # Replace the first entry with the fallback so skill discovery also uses it.
                effective_skill_dirs = [str(deer_flow_skills)] + list(effective_skill_dirs[1:])
                candidate = deer_flow_skills
        resolved_skills_path = str(candidate)

    config_path = write_deer_flow_config(
        config_dir=config_dir,
        sandbox=sandbox,
        selected_llm=selected_llm,
        skills_path=resolved_skills_path,
        warnings=warnings,
    )

    # Also copy config.yaml to the deer-flow root directory (required by deer-flow docs)
    if deer_flow_root is not None:
        root_config_path = deer_flow_root / "config.yaml"
        shutil.copy2(config_path, root_config_path)
        logger.debug(f"Copied config.yaml to deer-flow root: {root_config_path}")

    extensions_config = generate_extensions_config(mcp_server_names)

    # Determine skills to enable — always use the resolved effective_skill_dirs
    # (which may have been redirected to $DEER_FLOW_PATH/skills above).
    skills_to_enable = enabled_skills or []
    if not enabled_skills:
        if effective_skill_dirs:
            skills_to_enable = load_skills_from_directories(effective_skill_dirs, warnings=warnings)

    if skills_to_enable:
        config = get_raw_config()
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

    logger.debug(f"Deer-flow config ready: {config_path}, {ext_path}")
    return config_path, ext_path, warnings
