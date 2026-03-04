"""Unit tests for genai_tk.agents.langchain.config."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from genai_tk.agents.langchain.config import (
    AgentDefaults,
    AgentProfileConfig,
    BackendConfig,
    CheckpointerConfig,
    LangchainAgentsConfig,
    MiddlewareConfig,
    create_backend,
    create_checkpointer,
    instantiate_middlewares,
    load_unified_config,
    resolve_profile,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

MINIMAL_YAML: dict[str, Any] = {
    "langchain_agents": {
        "defaults": {"type": "react"},
        "default_profile": "simple",
        "profiles": [
            {"name": "simple", "type": "react", "llm": "parrot_local@fake", "tools": [], "mcp_servers": []},
            {
                "name": "deep_one",
                "type": "deep",
                "llm": "gpt41@openai",
                "tools": [],
                "mcp_servers": [],
                "skill_directories": ["/skills"],
            },
        ],
    }
}


def _write_yaml(tmp_path: Path, data: dict[str, Any]) -> Path:
    cfg_file = tmp_path / "langchain.yaml"
    cfg_file.write_text(yaml.dump(data))
    return cfg_file


# ---------------------------------------------------------------------------
# MiddlewareConfig
# ---------------------------------------------------------------------------


class TestMiddlewareConfig:
    def test_alias_class_field(self) -> None:
        cfg = MiddlewareConfig(**{"class": "some.module:SomeClass"})
        assert cfg.class_path == "some.module:SomeClass"

    def test_extra_kwargs_captured(self) -> None:
        cfg = MiddlewareConfig(**{"class": "mod:Cls", "limit": 10, "model": "gpt4"})
        assert cfg.extra_kwargs == {"limit": 10, "model": "gpt4"}

    def test_no_extras(self) -> None:
        cfg = MiddlewareConfig(**{"class": "mod:Cls"})
        assert cfg.extra_kwargs == {}


# ---------------------------------------------------------------------------
# CheckpointerConfig
# ---------------------------------------------------------------------------


class TestCheckpointerConfig:
    def test_defaults(self) -> None:
        cfg = CheckpointerConfig()
        assert cfg.type == "none"
        assert cfg.class_path is None
        assert cfg.kwargs == {}

    def test_memory_type(self) -> None:
        cfg = CheckpointerConfig(type="memory")
        assert cfg.type == "memory"

    def test_class_type_with_alias(self) -> None:
        cfg = CheckpointerConfig(
            **{"type": "class", "class": "langgraph.checkpoint.sqlite:SqliteSaver", "kwargs": {"conn": "db"}}
        )
        assert cfg.type == "class"
        assert cfg.class_path == "langgraph.checkpoint.sqlite:SqliteSaver"
        assert cfg.kwargs == {"conn": "db"}


# ---------------------------------------------------------------------------
# AgentDefaults / LangchainAgentsConfig
# ---------------------------------------------------------------------------


class TestLangchainAgentsConfig:
    def test_empty_defaults(self) -> None:
        defaults = AgentDefaults()
        assert defaults.type == "react"
        assert defaults.middlewares == []
        assert defaults.checkpointer.type == "none"

    def test_profile_validation(self) -> None:
        profile = AgentProfileConfig(name="test", type="react")
        assert profile.name == "test"
        assert profile.type == "react"

    def test_config_from_dict(self) -> None:
        cfg = LangchainAgentsConfig.model_validate(MINIMAL_YAML["langchain_agents"])
        assert cfg.default_profile == "simple"
        assert len(cfg.profiles) == 2
        assert cfg.profiles[0].name == "simple"
        assert cfg.profiles[1].type == "deep"


# ---------------------------------------------------------------------------
# load_unified_config
# ---------------------------------------------------------------------------


class TestLoadUnifiedConfig:
    def test_loads_minimal_yaml(self, tmp_path: Path) -> None:
        cfg_file = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg = load_unified_config(str(cfg_file))
        assert isinstance(cfg, LangchainAgentsConfig)
        assert len(cfg.profiles) == 2

    def test_default_profile_name(self, tmp_path: Path) -> None:
        cfg_file = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg = load_unified_config(str(cfg_file))
        assert cfg.default_profile == "simple"

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_unified_config(str(tmp_path / "nonexistent.yaml"))

    def test_missing_section_raises(self, tmp_path: Path) -> None:
        bad_yaml = {"other_section": {}}
        cfg_file = _write_yaml(tmp_path, bad_yaml)
        with pytest.raises(ValueError, match="missing 'langchain_agents'"):
            load_unified_config(str(cfg_file))

    def test_defaults_parsed(self, tmp_path: Path) -> None:
        data = {
            "langchain_agents": {
                "defaults": {"type": "deep", "llm": "gpt41@openai"},
                "default_profile": "d",
                "profiles": [{"name": "d"}],
            }
        }
        cfg_file = _write_yaml(tmp_path, data)
        cfg = load_unified_config(str(cfg_file))
        assert cfg.defaults.type == "deep"
        assert cfg.defaults.llm == "gpt41@openai"

    def test_middleware_in_defaults(self, tmp_path: Path) -> None:
        data = {
            "langchain_agents": {
                "defaults": {"middlewares": [{"class": "mod:RichMiddleware"}]},
                "default_profile": "",
                "profiles": [],
            }
        }
        cfg_file = _write_yaml(tmp_path, data)
        cfg = load_unified_config(str(cfg_file))
        assert len(cfg.defaults.middlewares) == 1
        assert cfg.defaults.middlewares[0].class_path == "mod:RichMiddleware"


# ---------------------------------------------------------------------------
# resolve_profile
# ---------------------------------------------------------------------------


class TestResolveProfile:
    def _make_config(self, tmp_path: Path) -> LangchainAgentsConfig:
        cfg_file = _write_yaml(tmp_path, MINIMAL_YAML)
        return load_unified_config(str(cfg_file))

    def test_finds_profile_by_name(self, tmp_path: Path) -> None:
        cfg = self._make_config(tmp_path)
        profile = resolve_profile(cfg, "simple")
        assert profile.name == "simple"

    def test_case_insensitive_match(self, tmp_path: Path) -> None:
        cfg = self._make_config(tmp_path)
        profile = resolve_profile(cfg, "SIMPLE")
        assert profile.name == "simple"

    def test_profile_not_found_raises(self, tmp_path: Path) -> None:
        cfg = self._make_config(tmp_path)
        with pytest.raises(ValueError, match="'unknown' not found"):
            resolve_profile(cfg, "unknown")

    def test_default_llm_inherited(self, tmp_path: Path) -> None:
        data = {
            "langchain_agents": {
                "defaults": {"llm": "default_llm@fake"},
                "default_profile": "p",
                "profiles": [{"name": "p"}],  # no llm set
            }
        }
        cfg_file = _write_yaml(tmp_path, data)
        cfg = load_unified_config(str(cfg_file))
        profile = resolve_profile(cfg, "p")
        assert profile.llm == "default_llm@fake"

    def test_profile_llm_overrides_default(self, tmp_path: Path) -> None:
        data = {
            "langchain_agents": {
                "defaults": {"llm": "default_llm@fake"},
                "default_profile": "p",
                "profiles": [{"name": "p", "llm": "profile_llm@fake"}],
            }
        }
        cfg_file = _write_yaml(tmp_path, data)
        cfg = load_unified_config(str(cfg_file))
        profile = resolve_profile(cfg, "p")
        assert profile.llm == "profile_llm@fake"

    def test_type_override(self, tmp_path: Path) -> None:
        cfg = self._make_config(tmp_path)
        profile = resolve_profile(cfg, "simple", type_override="custom")
        assert profile.type == "custom"

    def test_middleware_inherited(self, tmp_path: Path) -> None:
        data = {
            "langchain_agents": {
                "defaults": {"middlewares": [{"class": "mod:DefaultMiddleware"}]},
                "default_profile": "p",
                "profiles": [{"name": "p"}],  # no middlewares
            }
        }
        cfg_file = _write_yaml(tmp_path, data)
        cfg = load_unified_config(str(cfg_file))
        profile = resolve_profile(cfg, "p")
        assert len(profile.middlewares) == 1
        assert profile.middlewares[0].class_path == "mod:DefaultMiddleware"

    def test_profile_middleware_overrides_default(self, tmp_path: Path) -> None:
        data = {
            "langchain_agents": {
                "defaults": {"middlewares": [{"class": "mod:DefaultMiddleware"}]},
                "default_profile": "p",
                "profiles": [{"name": "p", "middlewares": [{"class": "mod:ProfileMiddleware"}]}],
            }
        }
        cfg_file = _write_yaml(tmp_path, data)
        cfg = load_unified_config(str(cfg_file))
        profile = resolve_profile(cfg, "p")
        assert len(profile.middlewares) == 1
        assert profile.middlewares[0].class_path == "mod:ProfileMiddleware"

    def test_deep_field_on_react_emits_warning(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        data = {
            "langchain_agents": {
                "defaults": {},
                "default_profile": "p",
                "profiles": [{"name": "p", "type": "react", "skill_directories": ["/skills"]}],
            }
        }
        cfg_file = _write_yaml(tmp_path, data)
        cfg = load_unified_config(str(cfg_file))
        resolve_profile(cfg, "p")
        # Rich prints to its own console, but at minimum no exception should be raised

    def test_deep_fields_allowed_on_deep_type(self, tmp_path: Path) -> None:
        cfg_file = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg = load_unified_config(str(cfg_file))
        profile = resolve_profile(cfg, "deep_one")
        assert "/skills" in profile.skill_directories


# ---------------------------------------------------------------------------
# create_checkpointer
# ---------------------------------------------------------------------------


class TestCreateCheckpointer:
    def test_none_config_returns_none(self) -> None:
        assert create_checkpointer(None) is None

    def test_type_none_returns_none(self) -> None:
        cfg = CheckpointerConfig(type="none")
        assert create_checkpointer(cfg) is None

    def test_type_memory_returns_memory_saver(self) -> None:
        from langgraph.checkpoint.memory import MemorySaver

        cfg = CheckpointerConfig(type="memory")
        result = create_checkpointer(cfg)
        assert isinstance(result, MemorySaver)

    def test_force_memory_overrides_none(self) -> None:
        from langgraph.checkpoint.memory import MemorySaver

        cfg = CheckpointerConfig(type="none")
        result = create_checkpointer(cfg, force_memory=True)
        assert isinstance(result, MemorySaver)

    def test_force_memory_with_no_config(self) -> None:
        from langgraph.checkpoint.memory import MemorySaver

        result = create_checkpointer(None, force_memory=True)
        assert isinstance(result, MemorySaver)

    def test_class_type_dynamic_import(self) -> None:
        # Use a real class from langgraph to avoid complex mocking
        from langgraph.checkpoint.memory import MemorySaver

        cfg = CheckpointerConfig(**{"type": "class", "class": "langgraph.checkpoint.memory:MemorySaver"})
        result = create_checkpointer(cfg)
        assert isinstance(result, MemorySaver)

    def test_class_type_missing_class_path_raises(self) -> None:
        cfg = CheckpointerConfig(type="class")  # no class_path
        with pytest.raises(ValueError, match="checkpointer.class is required"):
            create_checkpointer(cfg)


# ---------------------------------------------------------------------------
# instantiate_middlewares
# ---------------------------------------------------------------------------


class TestInstantiateMiddlewares:
    def test_empty_list(self) -> None:
        result = instantiate_middlewares([], "react")
        assert result == []

    def test_invalid_class_path_rejected(self) -> None:
        """Invalid class paths are rejected at model validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MiddlewareConfig(**{"class": "no_colon_here"})

    def test_import_failure_skipped(self) -> None:
        cfg = MiddlewareConfig(**{"class": "nonexistent.module:SomeClass"})
        result = instantiate_middlewares([cfg], "react")
        assert result == []  # import error → warning, no crash

    def test_deepagents_middleware_with_non_deep_emits_warning(self, capsys: pytest.CaptureFixture) -> None:
        # Just ensure no exception; warning goes to Rich console (not capsys)
        cfg = MiddlewareConfig(**{"class": "deepagents.middleware.summarization:SummarizationMiddleware"})
        # This will fail on import since deepagents may not be installed, but
        # the compatibility warning path should be exercised first
        instantiate_middlewares([cfg], "react")


# ---------------------------------------------------------------------------
# BackendConfig
# ---------------------------------------------------------------------------


class TestBackendConfig:
    def test_defaults(self) -> None:
        cfg = BackendConfig()
        assert cfg.type == "none"
        assert cfg.class_path is None
        assert cfg.kwargs == {}

    def test_aio_sandbox_type(self) -> None:
        cfg = BackendConfig(type="aio_sandbox")
        assert cfg.type == "aio_sandbox"

    def test_class_type_with_alias(self) -> None:
        cfg = BackendConfig(**{"type": "class", "class": "my_pkg.backends:MyBackend", "kwargs": {"opt": 1}})
        assert cfg.type == "class"
        assert cfg.class_path == "my_pkg.backends:MyBackend"
        assert cfg.kwargs == {"opt": 1}

    def test_extra_kwargs_for_aio_sandbox(self) -> None:
        """Extra YAML keys (e.g. host_port) are surfaced via extra_kwargs."""
        cfg = BackendConfig(type="aio_sandbox", host_port=19091, startup_timeout=120.0)  # type: ignore[call-arg]
        assert cfg.extra_kwargs["host_port"] == 19091
        assert cfg.extra_kwargs["startup_timeout"] == 120.0

    def test_agent_defaults_has_none_backend(self) -> None:
        defaults = AgentDefaults()
        assert defaults.backend.type == "none"

    def test_profile_backend_defaults_to_none(self) -> None:
        profile = AgentProfileConfig(name="test")
        assert profile.backend is None  # None means "inherit from defaults"


# ---------------------------------------------------------------------------
# resolve_profile — backend inheritance
# ---------------------------------------------------------------------------


class TestResolveProfileBackend:
    def _make_config_with_backend(
        self, tmp_path: Path, default_backend: dict, profile_backend: dict | None
    ) -> LangchainAgentsConfig:
        profile_data: dict = {"name": "p", "type": "deep"}
        if profile_backend is not None:
            profile_data["backend"] = profile_backend
        data = {
            "langchain_agents": {
                "defaults": {"backend": default_backend},
                "default_profile": "p",
                "profiles": [profile_data],
            }
        }
        cfg_file = tmp_path / "langchain.yaml"
        import yaml

        cfg_file.write_text(yaml.dump(data))
        return load_unified_config(str(cfg_file))

    def test_backend_inherited_from_defaults(self, tmp_path: Path) -> None:
        cfg = self._make_config_with_backend(tmp_path, {"type": "aio_sandbox", "host_port": 19091}, None)
        profile = resolve_profile(cfg, "p")
        assert profile.backend is not None
        assert profile.backend.type == "aio_sandbox"
        assert profile.backend.extra_kwargs.get("host_port") == 19091

    def test_profile_backend_overrides_default(self, tmp_path: Path) -> None:
        cfg = self._make_config_with_backend(
            tmp_path,
            {"type": "aio_sandbox"},
            {"type": "class", "class": "my_pkg:MyBackend"},
        )
        profile = resolve_profile(cfg, "p")
        assert profile.backend is not None
        assert profile.backend.type == "class"
        assert profile.backend.class_path == "my_pkg:MyBackend"

    def test_backend_none_in_default_and_profile(self, tmp_path: Path) -> None:
        cfg = self._make_config_with_backend(tmp_path, {"type": "none"}, None)
        profile = resolve_profile(cfg, "p")
        assert profile.backend is not None
        assert profile.backend.type == "none"

    def test_non_deep_profile_with_backend_triggers_warning(self, tmp_path: Path) -> None:
        """No exception; a Rich warning is printed but the profile is resolved."""
        data = {
            "langchain_agents": {
                "defaults": {},
                "default_profile": "p",
                "profiles": [{"name": "p", "type": "react", "backend": {"type": "aio_sandbox"}}],
            }
        }
        import yaml

        cfg_file = tmp_path / "langchain.yaml"
        cfg_file.write_text(yaml.dump(data))
        cfg = load_unified_config(str(cfg_file))
        profile = resolve_profile(cfg, "p")  # must not raise
        assert profile.backend is not None
        assert profile.backend.type == "aio_sandbox"


# ---------------------------------------------------------------------------
# create_backend
# ---------------------------------------------------------------------------


class TestCreateBackend:
    @pytest.mark.asyncio
    async def test_none_config_returns_none(self) -> None:
        assert await create_backend(None) is None

    @pytest.mark.asyncio
    async def test_type_none_returns_none(self) -> None:
        cfg = BackendConfig(type="none")
        assert await create_backend(cfg) is None

    @pytest.mark.asyncio
    async def test_aio_sandbox_instantiates_and_starts(self) -> None:
        from unittest.mock import AsyncMock, patch

        cfg = BackendConfig(type="aio_sandbox", host_port=19091)  # type: ignore[call-arg]

        with patch(
            "genai_tk.agents.langchain.sandbox_backend.AioSandboxBackend.start",
            new_callable=AsyncMock,
        ) as mock_start:
            from genai_tk.agents.langchain.sandbox_backend import AioSandboxBackend

            backend = await create_backend(cfg)

        assert isinstance(backend, AioSandboxBackend)
        mock_start.assert_awaited_once()
        # Extra kwargs forwarded to config
        assert backend.config.host_port == 19091

    @pytest.mark.asyncio
    async def test_aio_sandbox_default_config(self) -> None:
        from unittest.mock import AsyncMock, patch

        cfg = BackendConfig(type="aio_sandbox")

        with patch(
            "genai_tk.agents.langchain.sandbox_backend.AioSandboxBackend.start",
            new_callable=AsyncMock,
        ):
            from genai_tk.agents.langchain.sandbox_backend import AioSandboxBackend

            backend = await create_backend(cfg)

        assert isinstance(backend, AioSandboxBackend)
        # Default config values
        assert backend.config.host_port == 18091
        assert backend.config.work_dir == "/home/user"

    @pytest.mark.asyncio
    async def test_class_type_missing_class_path_raises(self) -> None:
        cfg = BackendConfig(type="class")  # no class_path
        with pytest.raises(ValueError, match="backend.class is required"):
            await create_backend(cfg)

    @pytest.mark.asyncio
    async def test_class_type_dynamic_import(self) -> None:
        """'class' type imports the class and calls start() if present."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_backend = MagicMock()
        mock_backend.start = AsyncMock()
        mock_cls = MagicMock(return_value=mock_backend)

        with patch("genai_tk.agents.langchain.config.import_from_qualified", return_value=mock_cls):
            cfg = BackendConfig(**{"type": "class", "class": "my_pkg:MyBackend", "kwargs": {"opt": 1}})
            backend = await create_backend(cfg)

        mock_cls.assert_called_once_with(opt=1)
        mock_backend.start.assert_awaited_once()
        assert backend is mock_backend

    @pytest.mark.asyncio
    async def test_class_type_no_start_method(self) -> None:
        """'class' type with no start() on the backend — should not raise."""
        from unittest.mock import MagicMock, patch

        mock_backend = MagicMock(spec=[])  # spec=[] → no attributes at all
        mock_cls = MagicMock(return_value=mock_backend)

        with patch("genai_tk.agents.langchain.config.import_from_qualified", return_value=mock_cls):
            cfg = BackendConfig(**{"type": "class", "class": "my_pkg:MyBackend"})
            backend = await create_backend(cfg)

        assert backend is mock_backend

    @pytest.mark.asyncio
    async def test_unknown_type_raises(self) -> None:
        cfg = BackendConfig.model_construct(type="unknown")  # bypass validation
        with pytest.raises(ValueError, match="Unknown backend type"):
            await create_backend(cfg)  # type: ignore[arg-type]
