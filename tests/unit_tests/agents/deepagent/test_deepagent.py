"""Tests for deepagent CLI config models, profile resolution, and LLM bridge."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genai_tk.agents.deepagent.models import DeepagentConfig, DeepagentProfile, load_deepagent_config

# ---------------------------------------------------------------------------
# DeepagentProfile tests
# ---------------------------------------------------------------------------


def test_profile_defaults():
    """DeepagentProfile has sensible defaults."""
    profile = DeepagentProfile(name="test")
    assert profile.name == "test"
    assert profile.llm is None
    assert profile.auto_approve is False
    assert profile.enable_memory is True
    assert profile.enable_skills is True
    assert profile.enable_shell is True
    assert profile.shell_allow_list == []
    assert profile.sandbox == "none"
    assert profile.system_prompt is None
    assert profile.tools == []


def test_profile_full_construction():
    """DeepagentProfile accepts all fields."""
    profile = DeepagentProfile(
        name="coder",
        description="Coding agent",
        llm="fast_model",
        auto_approve=True,
        enable_memory=False,
        enable_skills=False,
        enable_shell=True,
        shell_allow_list=["pytest", "git"],
        sandbox="modal",
        system_prompt="You are a coder.",
        tools=["web_search", "fetch_url"],
    )
    assert profile.llm == "fast_model"
    assert profile.auto_approve is True
    assert profile.shell_allow_list == ["pytest", "git"]
    assert profile.tools == ["web_search", "fetch_url"]


# ---------------------------------------------------------------------------
# DeepagentConfig tests
# ---------------------------------------------------------------------------


def test_config_defaults():
    """DeepagentConfig has sensible defaults."""
    config = DeepagentConfig()
    assert config.default_model is None
    assert config.default_profile is None
    assert config.auto_approve is False
    assert config.enable_memory is True
    assert config.enable_shell is True
    assert config.profiles == []


def test_config_get_profile_found():
    """DeepagentConfig.get_profile returns profile by name (case-insensitive)."""
    config = DeepagentConfig(
        profiles=[
            DeepagentProfile(name="Coder"),
            DeepagentProfile(name="Researcher"),
        ]
    )
    assert config.get_profile("Coder") is not None
    assert config.get_profile("coder") is not None
    assert config.get_profile("CODER") is not None
    assert config.get_profile("coder").name == "Coder"


def test_config_get_profile_not_found():
    """DeepagentConfig.get_profile returns None for unknown names."""
    config = DeepagentConfig(profiles=[DeepagentProfile(name="Coder")])
    assert config.get_profile("Researcher") is None
    assert config.get_profile("") is None


def test_config_model_validate_from_dict():
    """DeepagentConfig can be built from a plain dict (as OmegaConf would provide)."""
    raw = {
        "default_model": "fast_model",
        "default_profile": "coder",
        "auto_approve": True,
        "profiles": [
            {"name": "coder", "llm": "fast_model"},
            {"name": "researcher", "llm": "default", "tools": ["web_search"]},
        ],
    }
    config = DeepagentConfig.model_validate(raw)
    assert config.default_model == "fast_model"
    assert len(config.profiles) == 2
    assert config.profiles[1].tools == ["web_search"]


# ---------------------------------------------------------------------------
# load_deepagent_config tests
# ---------------------------------------------------------------------------


def test_load_deepagent_config_returns_defaults_when_missing():
    """load_deepagent_config returns defaults when 'deepagent' key is absent."""
    with patch("genai_tk.utils.config_mngr.global_config") as mock_gc:
        mock_gc.return_value.get.return_value = {}
        config = load_deepagent_config()
    assert isinstance(config, DeepagentConfig)
    assert config.profiles == []


def test_load_deepagent_config_parses_profiles():
    """load_deepagent_config correctly parses profiles from config dict."""
    raw = {
        "default_model": "gpt41mini@openai",
        "profiles": [
            {"name": "coder", "llm": "fast_model", "auto_approve": False},
        ],
    }
    with patch("genai_tk.utils.config_mngr.global_config") as mock_gc:
        mock_gc.return_value.get.return_value = raw
        config = load_deepagent_config()
    assert config.default_model == "gpt41mini@openai"
    assert len(config.profiles) == 1
    assert config.profiles[0].name == "coder"


def test_load_deepagent_config_handles_exception_gracefully():
    """load_deepagent_config returns empty config on unexpected error."""
    with patch("genai_tk.utils.config_mngr.global_config", side_effect=RuntimeError("broken")):
        config = load_deepagent_config()
    assert isinstance(config, DeepagentConfig)


# ---------------------------------------------------------------------------
# LLM bridge tests
# ---------------------------------------------------------------------------


def test_resolve_model_from_profile_uses_override():
    """resolve_model_from_profile prefers llm_override over profile.llm."""
    from genai_tk.agents.deepagent.llm_bridge import resolve_model_from_profile

    profile = DeepagentProfile(name="test", llm="profile_llm")
    config = DeepagentConfig(default_model="config_llm")
    fake_model = MagicMock()

    with patch("genai_tk.agents.deepagent.llm_bridge._resolve_identifier", return_value=fake_model) as mock_resolve:
        result = resolve_model_from_profile(profile, "override_llm", config)
        mock_resolve.assert_called_once_with("override_llm")
    assert result is fake_model


def test_resolve_model_from_profile_falls_back_to_profile():
    """resolve_model_from_profile uses profile.llm when no override given."""
    from genai_tk.agents.deepagent.llm_bridge import resolve_model_from_profile

    profile = DeepagentProfile(name="test", llm="profile_llm")
    config = DeepagentConfig(default_model="config_llm")
    fake_model = MagicMock()

    with patch("genai_tk.agents.deepagent.llm_bridge._resolve_identifier", return_value=fake_model) as mock_resolve:
        resolve_model_from_profile(profile, None, config)
        mock_resolve.assert_called_once_with("profile_llm")


def test_resolve_model_from_profile_falls_back_to_config_default():
    """resolve_model_from_profile uses config.default_model when profile has no llm."""
    from genai_tk.agents.deepagent.llm_bridge import resolve_model_from_profile

    profile = DeepagentProfile(name="test", llm=None)
    config = DeepagentConfig(default_model="config_default")
    fake_model = MagicMock()

    with patch("genai_tk.agents.deepagent.llm_bridge._resolve_identifier", return_value=fake_model) as mock_resolve:
        resolve_model_from_profile(profile, None, config)
        mock_resolve.assert_called_once_with("config_default")


def test_resolve_identifier_raises_when_nothing_resolved():
    """_resolve_identifier raises ValueError when no identifier is available."""
    from genai_tk.agents.deepagent.llm_bridge import _resolve_identifier

    with patch("genai_tk.agents.deepagent.llm_bridge.global_config") as mock_gc:
        mock_gc.return_value.get.return_value = None
        with pytest.raises(ValueError, match="No LLM specified"):
            _resolve_identifier(None)


def test_resolve_identifier_raises_on_bad_id():
    """_resolve_identifier raises ValueError when LlmFactory cannot resolve the ID."""
    from genai_tk.agents.deepagent.llm_bridge import _resolve_identifier

    with patch.object(
        __import__("genai_tk.core.llm_factory", fromlist=["LlmFactory"]).LlmFactory,
        "resolve_llm_identifier_safe",
        return_value=(None, "Unknown model: bad_id"),
    ):
        with pytest.raises(ValueError, match="Unknown model"):
            _resolve_identifier("bad_id")


def test_resolve_identifier_creates_model():
    """_resolve_identifier calls get_llm with the resolved identifier."""
    from genai_tk.agents.deepagent.llm_bridge import _resolve_identifier

    fake_model = MagicMock()
    with (
        patch.object(
            __import__("genai_tk.core.llm_factory", fromlist=["LlmFactory"]).LlmFactory,
            "resolve_llm_identifier_safe",
            return_value=("gpt41mini@openai", None),
        ),
        patch("genai_tk.agents.deepagent.llm_bridge.get_llm", return_value=fake_model) as mock_get_llm,
    ):
        result = _resolve_identifier("fast_model")
        mock_get_llm.assert_called_once_with(llm="gpt41mini@openai")
    assert result is fake_model


# ---------------------------------------------------------------------------
# CLI registration smoke test
# ---------------------------------------------------------------------------


def test_deepagent_commands_registers():
    """DeepagentCommands.register() produces a valid Typer sub-app without errors."""
    import typer

    from genai_tk.agents.deepagent.cli_commands import DeepagentCommands

    app = typer.Typer()
    DeepagentCommands().register(app)
    # Verify the 'deepagent' group was added
    registered_names = [info.name for info in app.registered_groups]
    assert "deepagent" in registered_names


# ---------------------------------------------------------------------------
# GenaiTkModelAdapter tests
# ---------------------------------------------------------------------------


def test_model_adapter_construction():
    """GenaiTkModelAdapter stores the model identifier and resolves via bridge."""
    from genai_tk.agents.deepagent.model_adapter import GenaiTkModelAdapter

    fake_delegate = MagicMock()
    with patch("genai_tk.agents.deepagent.llm_bridge._resolve_identifier", return_value=fake_delegate):
        adapter = GenaiTkModelAdapter(model="fast_model")

    assert adapter.model == "fast_model"
    assert adapter._delegate is fake_delegate


def test_model_adapter_llm_type():
    """GenaiTkModelAdapter._llm_type returns the expected string."""
    from genai_tk.agents.deepagent.model_adapter import GenaiTkModelAdapter

    fake_delegate = MagicMock()
    with patch("genai_tk.agents.deepagent.llm_bridge._resolve_identifier", return_value=fake_delegate):
        adapter = GenaiTkModelAdapter(model="default")

    assert adapter._llm_type == "genai_tk_adapter"


def test_model_adapter_generate_delegates():
    """GenaiTkModelAdapter._generate forwards to the delegate."""
    from langchain_core.messages import HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from genai_tk.agents.deepagent.model_adapter import GenaiTkModelAdapter

    fake_result = ChatResult(generations=[ChatGeneration(message=HumanMessage(content="hi"))])
    fake_delegate = MagicMock()
    fake_delegate._generate.return_value = fake_result

    with patch("genai_tk.agents.deepagent.llm_bridge._resolve_identifier", return_value=fake_delegate):
        adapter = GenaiTkModelAdapter(model="default")

    messages = [HumanMessage(content="hello")]
    result = adapter._generate(messages)

    fake_delegate._generate.assert_called_once_with(messages, stop=None, run_manager=None)
    assert result is fake_result


def test_model_adapter_model_name_property():
    """GenaiTkModelAdapter.model_name falls back to the model identifier."""
    from genai_tk.agents.deepagent.model_adapter import GenaiTkModelAdapter

    fake_delegate = MagicMock(spec=[])  # no model_name attribute

    with patch("genai_tk.agents.deepagent.llm_bridge._resolve_identifier", return_value=fake_delegate):
        adapter = GenaiTkModelAdapter(model="gpt41mini@openai")

    assert adapter.model_name == "gpt41mini@openai"


# ---------------------------------------------------------------------------
# DeepagentConfig.switcher_models tests
# ---------------------------------------------------------------------------


def test_config_switcher_models_default():
    """DeepagentConfig.switcher_models defaults to an empty list."""
    config = DeepagentConfig()
    assert config.switcher_models == []


def test_config_switcher_models_parsed():
    """DeepagentConfig.switcher_models is correctly parsed from a dict."""
    raw = {"switcher_models": ["default", "fast_model", "gpt41mini@openai"]}
    config = DeepagentConfig.model_validate(raw)
    assert config.switcher_models == ["default", "fast_model", "gpt41mini@openai"]


# ---------------------------------------------------------------------------
# toml_bridge tests
# ---------------------------------------------------------------------------


def test_write_genai_tk_provider_noop_on_empty(tmp_path):
    """write_genai_tk_provider is a no-op when models list is empty."""
    from genai_tk.agents.deepagent.toml_bridge import write_genai_tk_provider

    cfg_file = tmp_path / "config.toml"
    write_genai_tk_provider([], config_path=cfg_file)
    assert not cfg_file.exists()


def test_write_genai_tk_provider_creates_toml(tmp_path):
    """write_genai_tk_provider creates the config file with the genai_tk provider."""
    import tomllib

    from genai_tk.agents.deepagent.toml_bridge import (
        _ADAPTER_CLASS_PATH,
        write_genai_tk_provider,
    )

    cfg_file = tmp_path / "config.toml"
    models = ["default", "fast_model"]

    with patch("genai_tk.agents.deepagent.toml_bridge._clear_deepagents_cache"):
        write_genai_tk_provider(models, config_path=cfg_file)

    assert cfg_file.exists()
    data = tomllib.loads(cfg_file.read_text())
    provider = data["models"]["providers"]["genai_tk"]
    assert provider["class_path"] == _ADAPTER_CLASS_PATH
    assert provider["models"] == ["default", "fast_model"]
    # api_key_env must be present so deepagents-cli's credential check passes
    assert provider["api_key_env"] == "HOME"


def test_write_genai_tk_provider_preserves_existing(tmp_path):
    """write_genai_tk_provider keeps existing TOML keys when updating."""
    import tomllib

    from genai_tk.agents.deepagent.toml_bridge import write_genai_tk_provider

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[models]\ndefault = "anthropic:claude-3-5-haiku-latest"\n', encoding="utf-8")

    with patch("genai_tk.agents.deepagent.toml_bridge._clear_deepagents_cache"):
        write_genai_tk_provider(["default"], config_path=cfg_file)

    data = tomllib.loads(cfg_file.read_text())
    assert data["models"]["default"] == "anthropic:claude-3-5-haiku-latest"
    assert "genai_tk" in data["models"]["providers"]


def test_write_genai_tk_provider_clears_cache(tmp_path):
    """write_genai_tk_provider resets the deepagents-cli model cache."""
    from genai_tk.agents.deepagent.toml_bridge import write_genai_tk_provider

    cfg_file = tmp_path / "config.toml"
    with patch("genai_tk.agents.deepagent.toml_bridge._clear_deepagents_cache") as mock_clear:
        write_genai_tk_provider(["default"], config_path=cfg_file)
    mock_clear.assert_called_once()


# ---------------------------------------------------------------------------
# AioSandboxConfig tests
# ---------------------------------------------------------------------------


def test_aio_sandbox_config_defaults():
    """AioSandboxConfig has all-None defaults except env_vars."""
    from genai_tk.agents.deepagent.models import AioSandboxConfig

    cfg = AioSandboxConfig()
    assert cfg.image is None
    assert cfg.host is None
    assert cfg.host_port is None
    assert cfg.startup_timeout is None
    assert cfg.work_dir is None
    assert cfg.env_vars == {}


def test_aio_sandbox_config_in_profile():
    """DeepagentProfile can hold an AioSandboxConfig."""
    from genai_tk.agents.deepagent.models import AioSandboxConfig, DeepagentProfile

    profile = DeepagentProfile(
        name="aio_test",
        sandbox="aio",
        sandbox_config=AioSandboxConfig(image="my-image:latest", host_port=19000),
    )
    assert profile.sandbox == "aio"
    assert profile.sandbox_config is not None
    assert profile.sandbox_config.image == "my-image:latest"
    assert profile.sandbox_config.host_port == 19000


def test_config_sandbox_config_parsed():
    """DeepagentConfig sandbox_config field round-trips correctly."""
    from genai_tk.agents.deepagent.models import AioSandboxConfig, DeepagentConfig

    cfg = DeepagentConfig(sandbox_config=AioSandboxConfig(host="0.0.0.0", startup_timeout=90.0))
    assert cfg.sandbox_config is not None
    assert cfg.sandbox_config.host == "0.0.0.0"
    assert cfg.sandbox_config.startup_timeout == 90.0


# ---------------------------------------------------------------------------
# sandbox_bridge tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sandbox_context_yields_none_when_not_aio():
    """sandbox_context yields None for sandbox values other than 'aio'."""
    from genai_tk.agents.deepagent.models import DeepagentConfig, DeepagentProfile
    from genai_tk.agents.deepagent.sandbox_bridge import sandbox_context

    for sandbox_val in ("none", "modal", "daytona", "runloop", "langsmith"):
        profile = DeepagentProfile(name="p", sandbox=sandbox_val)
        config = DeepagentConfig()
        async with sandbox_context(profile, config) as backend:
            assert backend is None, f"Expected None for sandbox={sandbox_val!r}"


@pytest.mark.asyncio
async def test_sandbox_context_aio_starts_backend():
    """sandbox_context yields the AioSandboxBackend instance for sandbox='aio'."""
    from genai_tk.agents.deepagent.models import AioSandboxConfig, DeepagentConfig, DeepagentProfile
    from genai_tk.agents.deepagent.sandbox_bridge import sandbox_context

    profile = DeepagentProfile(
        name="aio",
        sandbox="aio",
        sandbox_config=AioSandboxConfig(image="my-image:latest"),
    )
    config = DeepagentConfig()

    mock_backend = MagicMock()

    with (
        patch("genai_tk.agents.langchain.sandbox_backend.AioSandboxBackendConfig") as MockConfig,
        patch("genai_tk.agents.langchain.sandbox_backend.AioSandboxBackend") as MockBackend,
    ):
        MockBackend.return_value.__aenter__ = AsyncMock(return_value=mock_backend)
        MockBackend.return_value.__aexit__ = AsyncMock(return_value=None)

        async with sandbox_context(profile, config) as backend:
            MockConfig.assert_called_once()
            MockBackend.assert_called_once()
            assert backend is mock_backend


def test_effective_sandbox_type_returns_none_for_aio_backend():
    """effective_sandbox_type returns None when a custom aio backend is active."""
    from genai_tk.agents.deepagent.models import DeepagentProfile
    from genai_tk.agents.deepagent.sandbox_bridge import effective_sandbox_type

    profile = DeepagentProfile(name="p", sandbox="aio")
    fake_backend = MagicMock()
    assert effective_sandbox_type(profile, fake_backend) is None


def test_effective_sandbox_type_forwards_builtin_types():
    """effective_sandbox_type passes through deepagents-cli native sandbox types."""
    from genai_tk.agents.deepagent.models import DeepagentProfile
    from genai_tk.agents.deepagent.sandbox_bridge import effective_sandbox_type

    for sandbox_val in ("modal", "daytona", "runloop", "langsmith"):
        profile = DeepagentProfile(name="p", sandbox=sandbox_val)
        assert effective_sandbox_type(profile, None) == sandbox_val


def test_effective_sandbox_type_returns_none_for_none():
    """effective_sandbox_type returns None when sandbox is 'none'."""
    from genai_tk.agents.deepagent.models import DeepagentProfile
    from genai_tk.agents.deepagent.sandbox_bridge import effective_sandbox_type

    profile = DeepagentProfile(name="p", sandbox="none")
    assert effective_sandbox_type(profile, None) is None


def test_sandbox_bridge_env_vars_merge():
    """_build_backend_config merges env_vars from global and profile, profile wins."""
    from genai_tk.agents.deepagent.models import AioSandboxConfig, DeepagentConfig, DeepagentProfile
    from genai_tk.agents.deepagent.sandbox_bridge import _build_backend_config

    config = DeepagentConfig(
        sandbox_config=AioSandboxConfig(
            image="base-image",
            env_vars={"GLOBAL_VAR": "global", "SHARED": "from_global"},
        )
    )
    profile = DeepagentProfile(
        name="p",
        sandbox="aio",
        sandbox_config=AioSandboxConfig(
            env_vars={"PROFILE_VAR": "profile", "SHARED": "from_profile"},
        ),
    )

    with patch(
        "genai_tk.agents.langchain.sandbox_backend.AioSandboxBackendConfig",
        side_effect=lambda **kw: kw,
    ):
        result = _build_backend_config(profile, config)

    assert result["env_vars"]["GLOBAL_VAR"] == "global"
    assert result["env_vars"]["PROFILE_VAR"] == "profile"
    assert result["env_vars"]["SHARED"] == "from_profile"  # profile wins
    assert result["image"] == "base-image"  # base image preserved when profile doesn't override

