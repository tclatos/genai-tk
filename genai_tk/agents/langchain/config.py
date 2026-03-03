"""Unified configuration models for LangChain-based agents.

This module defines Pydantic models for the unified agent configuration,
covering all agent types (react, deep, custom). It also provides factory
functions for checkpointers, backends, and dynamic middleware instantiation.

Unified config structure in ``langchain.yaml``:
```yaml
langchain_agents:
  defaults:
    type: react
    llm: null
    middlewares:
      - class: genai_tk.agents.langchain.rich_middleware:RichToolCallMiddleware
    checkpointer:
      type: none
    backend:
      type: none          # none | aio_sandbox | class
    enable_planning: true
    enable_file_system: true
    skills:
      directories:
        - ${paths.project}/skills

  default_profile: "Research"

  profiles:
    - name: "Research"
      type: deep
      llm: "gpt_41@openai"
      # Use the built-in Docker sandbox backend:
      backend:
        type: aio_sandbox
        host_port: 18091
        startup_timeout: 90.0
      # Or a custom deepagents-compatible backend:
      # backend:
      #   type: class
      #   class: my_pkg.backends:MyBackend
      #   kwargs:
      #     some_option: value
      middlewares:
        - class: deepagents.middleware.summarization:SummarizationMiddleware
          model: "gpt-4.1@openrouter"
          trigger: ["tokens", 4000]
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain.agents.middleware import AgentMiddleware
from langgraph.checkpoint.base import BaseCheckpointSaver
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from genai_tk.utils.config_mngr import import_from_qualified

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

AgentType = Literal["react", "deep", "custom"]

# Deep-only profile fields - used to warn when set on non-deep agents
_DEEP_ONLY_FIELDS = ("skill_directories", "subagents")


# ============================================================================
# Middleware
# ============================================================================


class MiddlewareConfig(BaseModel):
    """Configuration for a single agent middleware.

    Uses ``class`` key (aliased to ``class_path``) for the qualified import path,
    plus any additional kwargs passed to the constructor.
    ```yaml
    middlewares:
      - class: genai_tk.agents.langchain.rich_middleware:RichToolCallMiddleware
      - class: genai_tk.agents.langchain.rich_middleware:ToolCallLimitMiddleware
        thread_limit: 20
    ```
    """

    class_path: str = Field(..., alias="class")
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    @property
    def extra_kwargs(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


# ============================================================================
# Backend
# ============================================================================


class BackendConfig(BaseModel):
    """Configuration for a deepagents ``BackendProtocol`` implementation.

    Four backend types are supported:

    ``none`` (default)
        No backend — deepagents uses its built-in state backend.

    ``filesystem``
        A ``FilesystemBackend`` scoped to a ``root_dir``.  Uses
        ``virtual_mode=True`` to sandbox file operations:
        ```yaml
        backend:
          type: filesystem
          root_dir: ${paths.project}
        ```

    ``aio_sandbox``
        The built-in Docker-based ``AioSandboxBackend``.  Any field from
        ``AioSandboxBackendConfig`` can be set directly as a sibling key:
        ```yaml
        backend:
          type: aio_sandbox
          host_port: 18091
          startup_timeout: 90.0
          work_dir: /workspace
          env_vars:
            MY_VAR: "1"
        ```

    ``class``
        Any deepagents-compatible ``BackendProtocol`` loaded by qualified
        import path.  Constructor kwargs go in the ``kwargs`` mapping:
        ```yaml
        backend:
          type: class
          class: my_package.backends:MyBackend
          kwargs:
            some_option: value
        ```
    """

    type: Literal["none", "filesystem", "aio_sandbox", "class"] = "none"
    root_dir: str | None = None
    class_path: str | None = Field(None, alias="class")
    kwargs: dict[str, Any] = {}
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    @property
    def extra_kwargs(self) -> dict[str, Any]:
        """Return extra fields (used as constructor kwargs for ``aio_sandbox``)."""
        return dict(self.model_extra or {})


# ============================================================================
# Checkpointer
# ============================================================================


class CheckpointerConfig(BaseModel):
    """Configuration for a LangGraph checkpointer.

    ```yaml
    checkpointer:
      type: memory          # in-memory MemorySaver (default for chat mode)

    checkpointer:
      type: class           # any LangGraph-compatible saver
      class: langgraph.checkpoint.sqlite:SqliteSaver
      kwargs:
        conn_string: "data/checkpoints.db"
    ```
    """

    type: Literal["none", "memory", "class"] = "none"
    class_path: str | None = Field(None, alias="class")
    kwargs: dict[str, Any] = {}
    model_config = ConfigDict(populate_by_name=True)


# ============================================================================
# Skills (deep-agent-specific)
# ============================================================================


class SkillsConfig(BaseModel):
    """Skills configuration for deep agents."""

    directories: list[str] = []
    trace_loading: bool = True


# ============================================================================
# Agent profile
# ============================================================================


class AgentProfileConfig(BaseModel):
    """Unified configuration for a single agent profile.

    Covers all agent types (react, deep, custom).
    Deep-only fields (``skill_directories``, ``subagents``, ``backend``) trigger
    a console warning when used with ``type: react`` or ``type: custom``.
    """

    name: str
    type: AgentType = "react"
    description: str = ""
    llm: str | None = None
    system_prompt: str | None = None
    pre_prompt: str | None = None
    tools: list[dict[str, Any]] = []
    mcp_servers: list[str] = []
    middlewares: list[MiddlewareConfig] | None = None  # None = use inherited defaults
    checkpointer: CheckpointerConfig | None = None  # None = use inherited defaults
    backend: BackendConfig | None = None  # None = use inherited defaults
    # Deep-agent-specific
    skill_directories: list[str] = []
    enable_planning: bool = True
    enable_file_system: bool = True
    subagents: list[dict[str, Any]] = []
    # UI / documentation
    features: list[str] = []
    examples: list[str] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============================================================================
# Defaults & top-level config
# ============================================================================


class AgentDefaults(BaseModel):
    """Inheritable defaults applied to every profile that does not override them."""

    type: AgentType = "react"
    llm: str | None = None
    middlewares: list[MiddlewareConfig] = []
    checkpointer: CheckpointerConfig = CheckpointerConfig(type="none")
    backend: BackendConfig = BackendConfig(type="none")
    enable_planning: bool = True
    enable_file_system: bool = True
    skills: SkillsConfig = SkillsConfig()
    model_config = ConfigDict(arbitrary_types_allowed=True)


class LangchainAgentsConfig(BaseModel):
    """Top-level config model for the unified ``langchain.yaml``."""

    defaults: AgentDefaults = AgentDefaults()
    default_profile: str = ""
    profiles: list[AgentProfileConfig] = []
    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============================================================================
# Config loading
# ============================================================================


def load_unified_config(config_path: str | None = None) -> LangchainAgentsConfig:
    """Load and parse the unified ``langchain.yaml`` configuration file.

    Args:
        config_path: Path to ``langchain.yaml``. Defaults to
            ``{paths.config}/agents/langchain.yaml``.
    """
    from pathlib import Path

    import yaml

    from genai_tk.utils.config_mngr import global_config

    if config_path is None:
        config_dir = global_config().get_dir_path("paths.config")
        config_path = str(config_dir / "agents" / "langchain.yaml")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Agent config not found at {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw or "langchain_agents" not in raw:
        raise ValueError(f"Config file {path} is missing 'langchain_agents' section")

    section = raw["langchain_agents"]
    defaults_raw = section.get("defaults", {})
    default_profile = section.get("default_profile", "")
    profiles_raw = section.get("profiles", [])

    defaults = AgentDefaults.model_validate(defaults_raw) if defaults_raw else AgentDefaults()
    profiles = [AgentProfileConfig.model_validate(p) for p in profiles_raw]

    return LangchainAgentsConfig(defaults=defaults, default_profile=default_profile, profiles=profiles)


def resolve_profile(
    config: LangchainAgentsConfig,
    name: str,
    type_override: AgentType | None = None,
) -> AgentProfileConfig:
    """Find a profile by name, merge with defaults, warn about incompatibilities.

    Profile fields override defaults (shallow merge per field).  If a profile
    does not declare ``middlewares`` or ``checkpointer`` the default values are
    used.

    Args:
        config: Loaded top-level config.
        name: Profile name (case-insensitive).
        type_override: If set, overrides the profile's ``type`` field.

    Returns:
        Resolved ``AgentProfileConfig`` with defaults applied.
    """
    from rich.console import Console

    console = Console()
    name_lower = name.lower()
    match = next((p for p in config.profiles if p.name.lower() == name_lower), None)
    if match is None:
        available = [p.name for p in config.profiles]
        raise ValueError(f"Profile '{name}' not found. Available: {available}")

    d = config.defaults

    # Merge: profile wins, fall back to defaults
    resolved = AgentProfileConfig(
        name=match.name,
        type=type_override or match.type or d.type,
        description=match.description,
        llm=match.llm or d.llm,
        system_prompt=match.system_prompt,
        pre_prompt=match.pre_prompt,
        tools=match.tools,
        mcp_servers=match.mcp_servers,
        middlewares=match.middlewares if match.middlewares is not None else d.middlewares,
        checkpointer=match.checkpointer if match.checkpointer is not None else d.checkpointer,
        backend=match.backend if match.backend is not None else d.backend,
        skill_directories=match.skill_directories or d.skills.directories,
        enable_planning=match.enable_planning if "enable_planning" in match.model_fields_set else d.enable_planning,
        enable_file_system=match.enable_file_system
        if "enable_file_system" in match.model_fields_set
        else d.enable_file_system,
        subagents=match.subagents,
        features=match.features,
        examples=match.examples,
    )

    # Compatibility warnings — only for fields explicitly set on *this* profile,
    # not for values inherited from defaults.
    if resolved.type != "deep":
        warnings: list[str] = []
        if match.skill_directories:  # explicitly set, not inherited
            warnings.append(f"skill_directories ({resolved.skill_directories})")
        if match.subagents:  # explicitly set, not inherited
            warnings.append("subagents")
        if match.backend is not None and match.backend.type != "none":  # explicitly set
            warnings.append(f"backend (type={match.backend.type})")
        if warnings:
            console.print(
                f"[bold yellow]⚠  Profile '{resolved.name}' (type={resolved.type}) has deep-agent-only "
                f"features that will be ignored: {', '.join(warnings)}.[/bold yellow]"
            )

    return resolved


# ============================================================================
# Checkpointer factory
# ============================================================================


def create_checkpointer(config: CheckpointerConfig | None, force_memory: bool = False) -> BaseCheckpointSaver | None:
    """Instantiate a LangGraph checkpointer from config.

    Args:
        config: Checkpointer configuration, or None for no checkpointer.
        force_memory: If True, always return a ``MemorySaver`` regardless of config.
            Used when ``--chat`` is requested but profile has ``type: none``.
    """
    if force_memory:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    if config is None or config.type == "none":
        return None

    if config.type == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    if config.type == "class":
        if not config.class_path:
            raise ValueError("checkpointer.class is required when type is 'class'")
        cls = import_from_qualified(config.class_path)
        return cls(**config.kwargs)

    raise ValueError(f"Unknown checkpointer type: {config.type!r}")


# ============================================================================
# Middleware factory
# ============================================================================


def instantiate_middlewares(
    configs: list[MiddlewareConfig],
    agent_type: AgentType,
) -> list[AgentMiddleware]:
    """Dynamically import and instantiate middleware from config.

    Issues a console warning when ``deepagents.*`` middleware is used with a
    non-deep agent type.

    Args:
        configs: List of middleware configurations.
        agent_type: The resolved agent type (for compatibility warnings).
    """
    from rich.console import Console

    console = Console()
    middlewares: list[AgentMiddleware] = []

    for cfg in configs:
        module_path, _, class_name = cfg.class_path.rpartition(":")
        if not module_path:
            logger.warning(f"Invalid middleware class path: {cfg.class_path!r}. Expected 'module:ClassName'.")
            continue

        # Compatibility warning for deepagents middleware used with non-deep agents
        if agent_type != "deep" and module_path.startswith("deepagents"):
            console.print(
                f"[bold yellow]⚠  Middleware '{class_name}' from deepagents is designed for deep agents "
                f"and may not work correctly with agent type '{agent_type}'.[/bold yellow]"
            )

        try:
            cls = import_from_qualified(cfg.class_path)
        except Exception as e:
            logger.warning(f"Failed to import middleware '{cfg.class_path}': {e}")
            continue

        kwargs = cfg.extra_kwargs
        # Resolve any LLM name in 'model' kwarg using LlmFactory
        if "model" in kwargs and isinstance(kwargs["model"], str):
            try:
                from genai_tk.core.llm_factory import LlmFactory

                kwargs["model"] = LlmFactory.resolve_llm_identifier(kwargs["model"])
            except Exception:
                pass  # Leave as-is; the middleware constructor will handle it

        try:
            instance = cls(**kwargs)
        except Exception as e:
            logger.warning(f"Failed to instantiate middleware '{cfg.class_path}' with kwargs {kwargs}: {e}")
            continue

        middlewares.append(instance)

    return middlewares


# ============================================================================
# Backend factory
# ============================================================================


def _resolve_interpolation(value: str) -> str:
    """Resolve OmegaConf ``${…}`` interpolations in a single string value."""
    if "${" not in value:
        return value
    from omegaconf import OmegaConf

    from genai_tk.utils.config_mngr import global_config

    cfg = OmegaConf.create({"_v": value})
    merged = OmegaConf.merge(global_config().root, cfg)
    return str(OmegaConf.to_container(merged, resolve=True)["_v"])  # type: ignore[index]


async def create_backend(config: BackendConfig | None) -> BackendProtocol | None:
    """Instantiate and start a deepagents backend from config.

    Backends with an async ``start()`` method (e.g. ``AioSandboxBackend``) are
    started automatically.  Callers are responsible for calling ``stop()`` (or
    using the backend as an async context manager) when done.

    Args:
        config: Backend configuration, or ``None`` / ``type: none`` for no backend.

    Returns:
        A started ``BackendProtocol`` instance, or ``None`` when ``type`` is ``none``.
    """
    if config is None or config.type == "none":
        return None

    if config.type == "filesystem":
        from deepagents.backends.filesystem import FilesystemBackend

        root = _resolve_interpolation(config.root_dir) if config.root_dir else "."
        return FilesystemBackend(root_dir=root, virtual_mode=True)

    if config.type == "aio_sandbox":
        from genai_tk.agents.langchain.sandbox_backend import AioSandboxBackend, AioSandboxBackendConfig

        # Extra YAML keys (host_port, startup_timeout, etc.) map directly onto AioSandboxBackendConfig
        cfg_kwargs = {**config.kwargs, **config.extra_kwargs}
        sandbox_cfg = AioSandboxBackendConfig.model_validate(cfg_kwargs) if cfg_kwargs else AioSandboxBackendConfig()
        backend = AioSandboxBackend(config=sandbox_cfg)
        await backend.start()
        return backend

    if config.type == "class":
        if not config.class_path:
            raise ValueError("backend.class is required when type is 'class'")
        cls = import_from_qualified(config.class_path)
        backend = cls(**config.kwargs)
        if hasattr(backend, "start"):
            await backend.start()
        return backend

    raise ValueError(f"Unknown backend type: {config.type!r}")
