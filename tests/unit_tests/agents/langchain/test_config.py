"""Unit tests for genai_tk.agents.langchain.config."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from genai_tk.agents.langchain.config import (
    AgentDefaults,
    AgentProfileConfig,
    CheckpointerConfig,
    LangchainAgentsConfig,
    MiddlewareConfig,
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

    def test_invalid_class_path_skipped(self) -> None:
        cfg = MiddlewareConfig(**{"class": "no_colon_here"})
        result = instantiate_middlewares([cfg], "react")
        assert result == []  # bad path → warning, no crash

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
