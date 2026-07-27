"""Pydantic models and factories for LangChain-based agent profiles.

Defines the ``harness: langchain`` profile schema (``AgentProfileConfig``) and
the middleware / backend / checkpointer config models + factories used by
:func:`genai_tk.agents.langchain.factory.create_langchain_agent`.

Profiles themselves are loaded from the unified ``agents:`` dict by
:func:`genai_tk.agents.harness.profiles.load_langchain_profiles` (which applies
the ``agent_defaults`` inheritable block); this module only holds the schema
and the runtime factories.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from genai_tk.agents.tools.tool_specs import ToolSpec
from genai_tk.config_mgmt.config_mngr import QualifiedClassName
from genai_tk.config_mgmt.import_utils import ImportResolver

import_from_qualified = ImportResolver.import_from_qualified

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from langchain.agents.middleware import AgentMiddleware
    from langgraph.checkpoint.base import BaseCheckpointSaver

AgentType = Literal["react", "deep", "custom"]


# ============================================================================
# Middleware
# ============================================================================


class MiddlewareConfig(BaseModel):
    """Configuration for a single agent middleware.

    Uses ``class`` key (aliased to ``class_path``) for the qualified import path,
    plus any additional kwargs passed to the constructor.
    ```yaml
    middlewares:
      - class: genai_tk.agents.langchain.middleware.rich_middleware.RichToolCallMiddleware
      - class: genai_tk.agents.langchain.middleware.rich_middleware.ToolCallLimitMiddleware
        thread_limit: 20
    ```
    """

    class_path: QualifiedClassName = Field(
        ..., alias="class", description="Qualified class name (module.path:ClassName)"
    )
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    @property
    def extra_kwargs(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


# ----------------------------------------------------------------------------
# Built-in middleware shorthand registry
# ----------------------------------------------------------------------------

# Maps a friendly name to a MiddlewareConfig dict (same shape as YAML). Used by
# ``LangchainAgent(extra_middlewares=[...])`` to apply well-known middlewares by
# name without spelling out the qualified class path.
_MIDDLEWARE_REGISTRY: dict[str, dict[str, Any]] = {
    "AnonymizationMiddleware": {
        "class": "genai_tk.agents.langchain.middleware.anonymization_middleware.AnonymizationMiddleware",
        "analyzed_fields": ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "LOCATION"],
        "fuzzy_deanonymize": True,
    },
    "RichToolCallMiddleware": {
        "class": "genai_tk.agents.langchain.middleware.rich_middleware.RichToolCallMiddleware",
    },
    "EmptyResponseRetryMiddleware": {
        "class": "genai_tk.agents.langchain.middleware.empty_response_retry.EmptyResponseRetryMiddleware",
        "max_retries": 2,
    },
}


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
          opensandbox_server_url: http://localhost:8080
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
          class_path: my_package.backends:MyBackend
          kwargs:
            some_option: value
        ```
    """

    type: Literal["none", "filesystem", "aio_sandbox", "docker", "class"] = Field(
        "none", description="Backend type: none (default), filesystem, aio_sandbox/docker, or class"
    )
    root_dir: str | None = Field(None, description="Root directory for the filesystem backend")
    class_path: QualifiedClassName | None = Field(
        None, alias="class", description="Qualified class name for custom backends (module.path:ClassName)"
    )
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Extra constructor kwargs for the class backend")
    model_config = ConfigDict(extra="allow", populate_by_name=True)

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
      class_path: langgraph.checkpoint.sqlite:SqliteSaver
      kwargs:
        conn_string: "data/checkpoints.db"
    ```
    """

    type: Literal["none", "memory", "class"] = Field(
        "none", description="Checkpointer type: none (no persistence), memory (in-process), or class (custom)"
    )
    class_path: QualifiedClassName | None = Field(
        None, alias="class", description="Qualified class name for custom checkpointers (module.path:ClassName)"
    )
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Constructor kwargs for the class checkpointer")
    model_config = ConfigDict(populate_by_name=True)


# ============================================================================
# Agent profile
# ============================================================================


class AgentProfileConfig(BaseModel):
    """Configuration for a single ``harness: langchain`` agent profile.

    Covers all agent types (react, deep, custom). Deep-only fields
    (``skill_directories``, ``subagents``, ``backend``) are only used by
    ``type: deep`` agents and ignored by react/custom profiles.
    """

    name: str = Field(..., description="Unique profile name used to select this configuration")
    type: AgentType = Field("react", description="Agent type: react, deep, or custom")
    harness: Literal["langchain"] = Field(
        "langchain", description="Harness discriminator; always 'langchain' for this profile type"
    )
    description: str = Field("", description="Human-readable description shown in UI and help text")
    llm: str | None = Field(None, description="LLM identifier (e.g. 'gpt-4o@openai'); falls back to defaults.llm")
    system_prompt: str | None = Field(None, description="System prompt injected at the start of the conversation")
    pre_prompt: str | None = Field(None, description="Prefix prepended to every user message")
    tools: list[ToolSpec] = Field(default_factory=list, description="Tool specifications loaded from YAML")
    mcp_servers: list[str] = Field(default_factory=list, description="MCP server names to attach to this profile")
    middlewares: list[MiddlewareConfig] | None = Field(
        None, description="Middleware stack; None inherits from defaults"
    )
    checkpointer: CheckpointerConfig | None = Field(
        None, description="Checkpointer config; None inherits from defaults"
    )
    backend: BackendConfig | None = Field(None, description="Execution backend config; None inherits from defaults")
    skill_directories: list[str] = Field(default_factory=list, description="Directories to scan for deep-agent skills")
    enable_planning: bool = Field(True, description="Enable multi-step planning (deep agents only)")
    enable_file_system: bool = Field(True, description="Allow file-system access inside the sandbox")
    subagents: list[dict[str, Any]] = Field(default_factory=list, description="Subagent definitions (deep agents only)")
    features: list[str] = Field(default_factory=list, description="Feature flags shown in the UI")
    examples: list[str] = Field(default_factory=list, description="Example prompts shown in the UI")
    recursion_limit: int = Field(
        100,
        description=(
            "Max LangGraph steps per turn, passed through as the `recursion_limit` "
            "run config. LangGraph's own built-in default is 25, which multi-step "
            "deep agents (planning + several tool calls) can exceed on a single "
            "non-trivial task, raising GraphRecursionError."
        ),
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
        try:
            cls = import_from_qualified(cfg.class_path)
        except Exception as e:
            logger.warning(f"Failed to import middleware '{cfg.class_path}': {e}")
            continue

        module_path = getattr(cls, "__module__", "")
        class_name = getattr(cls, "__name__", repr(cls))
        if not module_path:
            logger.warning(f"Invalid middleware class reference: {cfg.class_path!r}.")
            continue

        # Compatibility warning for deepagents middleware used with non-deep agents
        if agent_type != "deep" and module_path.startswith("deepagents"):
            console.print(
                f"[bold yellow]⚠  Middleware '{class_name}' from deepagents is designed for deep agents "
                f"and may not work correctly with agent type '{agent_type}'.[/bold yellow]"
            )

        kwargs = cfg.extra_kwargs
        # Resolve any LLM name in 'model' kwarg using get_llm
        if "model" in kwargs and isinstance(kwargs["model"], str):
            try:
                from genai_tk.core.factories.llm_factory import LlmFactory

                kwargs["model"] = LlmFactory.resolve_llm_identifier(kwargs["model"])
            except Exception:
                pass  # Leave as-is; the middleware constructor will handle it
        # YAML deserializes tuples as lists; SummarizationMiddleware expects tuples for
        # 'trigger' and 'keep' (ContextSize = tuple[str, int]).
        for _tuple_key in ("trigger", "keep"):
            if _tuple_key in kwargs and isinstance(kwargs[_tuple_key], list):
                kwargs[_tuple_key] = tuple(kwargs[_tuple_key])

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

    from genai_tk.config_mgmt.config_mngr import get_raw_config

    cfg = OmegaConf.create({"_v": value})
    merged = OmegaConf.merge(get_raw_config(), cfg)
    return str(OmegaConf.to_container(merged, resolve=True)["_v"])  # type: ignore[index]


async def instantiate_backend(config: BackendConfig | None) -> BackendProtocol | None:
    """Instantiate a backend from config without calling ``start()``.

    Use this when you need to configure the backend (e.g. add volume mounts)
    before the container starts.  The caller is responsible for calling
    ``await backend.start()`` afterwards.

    Args:
        config: Backend configuration, or ``None`` / ``type: none`` for no backend.

    Returns:
        An unstarted ``BackendProtocol`` instance, or ``None`` when ``type`` is ``none``.
    """
    if config is None or config.type == "none":
        return None

    if config.type == "filesystem":
        from deepagents.backends.filesystem import FilesystemBackend

        root = _resolve_interpolation(config.root_dir) if config.root_dir else "."
        return FilesystemBackend(root_dir=root, virtual_mode=True)

    if config.type in ("aio_sandbox", "docker"):
        from genai_tk.agents.sandbox.aio_backend import AioSandboxBackend
        from genai_tk.agents.sandbox.config import get_docker_aio_settings

        base_settings = get_docker_aio_settings()
        overrides = {**config.kwargs, **config.extra_kwargs}
        if overrides:
            sandbox_cfg = base_settings.model_copy(update=overrides)
        else:
            sandbox_cfg = base_settings
        return AioSandboxBackend(config=sandbox_cfg)

    if config.type == "class":
        if not config.class_path:
            raise ValueError("backend.class is required when type is 'class'")
        cls = import_from_qualified(config.class_path)
        return cls(**config.kwargs)

    raise ValueError(f"Unknown backend type: {config.type!r}")


async def create_backend(config: BackendConfig | None) -> BackendProtocol | None:
    """Instantiate and start a deepagents backend from config.

    Backends with an async ``start()`` method (e.g. ``AioSandboxBackend``) are
    started automatically.  Callers are responsible for calling ``stop()`` (or
    using the backend as an async context manager) when done.

    When you need to configure the backend before starting (e.g. add volume mounts),
    use ``instantiate_backend`` instead, configure, then call ``await backend.start()``.

    Args:
        config: Backend configuration, or ``None`` / ``type: none`` for no backend.

    Returns:
        A started ``BackendProtocol`` instance, or ``None`` when ``type`` is ``none``.
    """
    from typing import cast

    backend = await instantiate_backend(config)
    if backend is not None:
        start_method = getattr(backend, "start", None)
        if callable(start_method):
            await cast(Any, start_method)()
    return backend
