"""Unit tests for genai_tk.agents.langchain.config."""

import pytest

from genai_tk.agents.langchain.config import (
    AgentProfileConfig,
    BackendConfig,
    CheckpointerConfig,
    MiddlewareConfig,
    create_backend,
    create_checkpointer,
    instantiate_middlewares,
)

# ---------------------------------------------------------------------------
# MiddlewareConfig
# ---------------------------------------------------------------------------


class TestMiddlewareConfig:
    def test_alias_class_field(self) -> None:
        cfg = MiddlewareConfig(**{"class": "some.module.SomeClass"})
        assert cfg.class_path == "some.module.SomeClass"

    def test_extra_kwargs_captured(self) -> None:
        cfg = MiddlewareConfig(**{"class": "mod.Cls", "limit": 10, "model": "gpt4"})
        assert cfg.extra_kwargs == {"limit": 10, "model": "gpt4"}

    def test_no_extras(self) -> None:
        cfg = MiddlewareConfig(**{"class": "mod.Cls"})
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
            **{"type": "class", "class": "langgraph.checkpoint.sqlite.SqliteSaver", "kwargs": {"conn": "db"}}
        )
        assert cfg.type == "class"
        assert cfg.class_path == "langgraph.checkpoint.sqlite.SqliteSaver"
        assert cfg.kwargs == {"conn": "db"}


# ---------------------------------------------------------------------------
# AgentProfileConfig
# ---------------------------------------------------------------------------


class TestAgentProfileConfig:
    def test_profile_validation(self) -> None:
        profile = AgentProfileConfig(name="test", type="react")
        assert profile.name == "test"
        assert profile.type == "react"

    def test_harness_discriminator(self) -> None:
        profile = AgentProfileConfig(name="test")
        assert profile.harness == "langchain"


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

        cfg = CheckpointerConfig(**{"type": "class", "class": "langgraph.checkpoint.memory.MemorySaver"})
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
        cfg = MiddlewareConfig(**{"class": "nonexistent.module.SomeClass"})
        result = instantiate_middlewares([cfg], "react")
        assert result == []  # import error → warning, no crash

    def test_deepagents_middleware_with_non_deep_emits_warning(self, capsys: pytest.CaptureFixture) -> None:
        # Just ensure no exception; warning goes to Rich console (not capsys)
        cfg = MiddlewareConfig(**{"class": "deepagents.middleware.summarization.SummarizationMiddleware"})
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
        cfg = BackendConfig(**{"type": "class", "class": "my_pkg.backends.MyBackend", "kwargs": {"opt": 1}})
        assert cfg.type == "class"
        assert cfg.class_path == "my_pkg.backends.MyBackend"
        assert cfg.kwargs == {"opt": 1}

    def test_extra_kwargs_for_aio_sandbox(self) -> None:
        """Extra YAML keys (e.g. opensandbox_server_url) are surfaced via extra_kwargs."""
        cfg = BackendConfig(type="aio_sandbox", opensandbox_server_url="http://myserver:8080", startup_timeout=120.0)  # type: ignore[call-arg]
        assert cfg.extra_kwargs["opensandbox_server_url"] == "http://myserver:8080"
        assert cfg.extra_kwargs["startup_timeout"] == 120.0

    def test_profile_backend_defaults_to_none(self) -> None:
        profile = AgentProfileConfig(name="test")
        assert profile.backend is None  # None means "inherit from defaults"


# ---------------------------------------------------------------------------
# create_backend
# ---------------------------------------------------------------------------


class TestCreateBackend:
    async def test_none_config_returns_none(self) -> None:
        assert await create_backend(None) is None

    async def test_type_none_returns_none(self) -> None:
        cfg = BackendConfig(type="none")
        assert await create_backend(cfg) is None

    async def test_aio_sandbox_instantiates_and_starts(self) -> None:
        from genai_tk.config_mgmt.features import is_available

        if not is_available("harnessing"):
            pytest.skip("Optional feature 'harnessing' not installed — run: uv sync --extra harnessing")

        from unittest.mock import AsyncMock, patch

        cfg = BackendConfig(type="aio_sandbox", opensandbox_server_url="http://myserver:8080")  # type: ignore[call-arg]

        with patch(
            "genai_tk.agents.langchain.sandbox_backend.AioSandboxBackend.start",
            new_callable=AsyncMock,
        ) as mock_start:
            from genai_tk.agents.langchain.sandbox_backend import AioSandboxBackend

            backend = await create_backend(cfg)

        assert isinstance(backend, AioSandboxBackend)
        mock_start.assert_awaited_once()
        # Extra kwargs forwarded to config
        assert backend.config.opensandbox_server_url == "http://myserver:8080"

    async def test_aio_sandbox_default_config(self) -> None:
        from genai_tk.config_mgmt.features import is_available

        if not is_available("harnessing"):
            pytest.skip("Optional feature 'harnessing' not installed — run: uv sync --extra harnessing")

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
        assert backend.config.opensandbox_server_url == "http://localhost:8080"
        assert backend.config.work_dir == "/home/user"

    async def test_class_type_missing_class_path_raises(self) -> None:
        cfg = BackendConfig(type="class")  # no class_path
        with pytest.raises(ValueError, match="backend.class is required"):
            await create_backend(cfg)

    async def test_class_type_dynamic_import(self) -> None:
        """'class' type imports the class and calls start() if present."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_backend = MagicMock()
        mock_backend.start = AsyncMock()
        mock_cls = MagicMock(return_value=mock_backend)

        with patch("genai_tk.agents.langchain.config.import_from_qualified", return_value=mock_cls):
            cfg = BackendConfig(**{"type": "class", "class": "my_pkg.MyBackend", "kwargs": {"opt": 1}})
            backend = await create_backend(cfg)

        mock_cls.assert_called_once_with(opt=1)
        mock_backend.start.assert_awaited_once()
        assert backend is mock_backend

    async def test_class_type_no_start_method(self) -> None:
        """'class' type with no start() on the backend — should not raise."""
        from unittest.mock import MagicMock, patch

        mock_backend = MagicMock(spec=[])  # spec=[] → no attributes at all
        mock_cls = MagicMock(return_value=mock_backend)

        with patch("genai_tk.agents.langchain.config.import_from_qualified", return_value=mock_cls):
            cfg = BackendConfig(**{"type": "class", "class": "my_pkg.MyBackend"})
            backend = await create_backend(cfg)

        assert backend is mock_backend

    async def test_unknown_type_raises(self) -> None:
        cfg = BackendConfig.model_construct(type="unknown")  # bypass validation
        with pytest.raises(ValueError, match="Unknown backend type"):
            await create_backend(cfg)  # type: ignore[arg-type]
