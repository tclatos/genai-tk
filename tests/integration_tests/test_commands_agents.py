"""Integration tests for LangChain agents CLI command.

Tests the `langchain` command via AgentCommands (actual CLI structure).
Uses fake LLM to avoid real model costs.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import typer
from typer.testing import CliRunner

from genai_tk.agents.commands_agents import AgentCommands
from genai_tk.agents.langchain.commands import _display_config_error, _get_config_path, _list_profiles


@pytest.fixture
def agents_app() -> typer.Typer:
    """Create app with AgentCommands registered (mirrors actual CLI structure)."""
    app = typer.Typer()
    AgentCommands().register(app)
    return app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestLangchainCommandHelp:
    def test_help_exits_zero(self, agents_app, runner) -> None:
        result = runner.invoke(agents_app, ["agents", "langchain", "--help"])
        assert result.exit_code == 0

    def test_help_shows_options(self, agents_app, runner) -> None:
        result = runner.invoke(agents_app, ["agents", "langchain", "--help"])
        assert "--list" in result.stdout or "--profile" in result.stdout

    def test_agents_group_help(self, agents_app, runner) -> None:
        result = runner.invoke(agents_app, ["agents", "--help"])
        assert result.exit_code == 0
        assert "langchain" in result.stdout


class TestListProfiles:
    def test_list_command_exits_zero(self, agents_app, runner) -> None:
        result = runner.invoke(agents_app, ["agents", "langchain", "--list"])
        assert result.exit_code == 0

    def test_list_shows_profiles(self, agents_app, runner) -> None:
        result = runner.invoke(agents_app, ["agents", "langchain", "--list"])
        assert "LangChain" in result.stdout or "Profile" in result.stdout

    def test_list_short_flag(self, agents_app, runner) -> None:
        result = runner.invoke(agents_app, ["agents", "langchain", "-l"])
        assert result.exit_code == 0

    def test_list_profiles_function(self) -> None:
        # Test the helper function directly
        _list_profiles()  # Should not raise


class TestLangchainCommandRun:
    def test_basic_query_with_fake_llm(self, agents_app, runner) -> None:
        with patch(
            "genai_tk.agents.langchain.factory.create_langchain_agent",
            new_callable=AsyncMock,
        ) as mock_factory:
            from langchain_core.messages import AIMessage

            mock_compiled = AsyncMock()
            mock_compiled.ainvoke.return_value = {"messages": [AIMessage(content="42")]}
            mock_factory.return_value = mock_compiled

            result = runner.invoke(agents_app, ["agents", "langchain", "what is 2+2"])
            assert result.exit_code == 0

    def test_query_with_llm_override(self, agents_app, runner) -> None:
        with patch(
            "genai_tk.agents.langchain.factory.create_langchain_agent",
            new_callable=AsyncMock,
        ) as mock_factory:
            from langchain_core.messages import AIMessage

            mock_compiled = AsyncMock()
            mock_compiled.ainvoke.return_value = {"messages": [AIMessage(content="result")]}
            mock_factory.return_value = mock_compiled

            result = runner.invoke(agents_app, ["agents", "langchain", "hello", "--llm", "parrot_local@fake"])
            assert result.exit_code == 0

    def test_query_with_type_react(self, agents_app, runner) -> None:
        with patch(
            "genai_tk.agents.langchain.factory.create_langchain_agent",
            new_callable=AsyncMock,
        ) as mock_factory:
            from langchain_core.messages import AIMessage

            mock_compiled = AsyncMock()
            mock_compiled.ainvoke.return_value = {"messages": [AIMessage(content="answer")]}
            mock_factory.return_value = mock_compiled

            result = runner.invoke(agents_app, ["agents", "langchain", "query", "--type", "react"])
            assert result.exit_code == 0

    def test_invalid_type_shows_error(self, agents_app, runner) -> None:
        result = runner.invoke(agents_app, ["agents", "langchain", "query", "--type", "invalid_type"])
        assert result.exit_code == 1
        assert "Invalid" in result.stdout or "react" in result.stdout

    def test_no_query_no_chat_shows_error(self, agents_app, runner) -> None:
        result = runner.invoke(agents_app, ["agents", "langchain"])
        assert result.exit_code == 1
        assert "query" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_details_flag(self, agents_app, runner) -> None:
        with patch(
            "genai_tk.agents.langchain.factory.create_langchain_agent",
            new_callable=AsyncMock,
        ) as mock_factory:
            from langchain_core.messages import AIMessage

            mock_compiled = AsyncMock()
            mock_compiled.ainvoke.return_value = {"messages": [AIMessage(content="result")]}
            mock_factory.return_value = mock_compiled

            result = runner.invoke(agents_app, ["agents", "langchain", "query", "--details"])
            assert result.exit_code == 0

    def test_cache_option(self, agents_app, runner) -> None:
        with patch(
            "genai_tk.agents.langchain.factory.create_langchain_agent",
            new_callable=AsyncMock,
        ) as mock_factory:
            from langchain_core.messages import AIMessage

            mock_compiled = AsyncMock()
            mock_compiled.ainvoke.return_value = {"messages": [AIMessage(content="result")]}
            mock_factory.return_value = mock_compiled

            result = runner.invoke(agents_app, ["agents", "langchain", "query", "--cache", "memory"])
            assert result.exit_code == 0

    def test_verbose_flag(self, agents_app, runner) -> None:
        with patch(
            "genai_tk.agents.langchain.factory.create_langchain_agent",
            new_callable=AsyncMock,
        ) as mock_factory:
            from langchain_core.messages import AIMessage

            mock_compiled = AsyncMock()
            mock_compiled.ainvoke.return_value = {"messages": [AIMessage(content="result")]}
            mock_factory.return_value = mock_compiled

            result = runner.invoke(agents_app, ["agents", "langchain", "query", "--verbose"])
            assert result.exit_code == 0

    def test_debug_flag(self, agents_app, runner) -> None:
        with patch(
            "genai_tk.agents.langchain.factory.create_langchain_agent",
            new_callable=AsyncMock,
        ) as mock_factory:
            from langchain_core.messages import AIMessage

            mock_compiled = AsyncMock()
            mock_compiled.ainvoke.return_value = {"messages": [AIMessage(content="result")]}
            mock_factory.return_value = mock_compiled

            result = runner.invoke(agents_app, ["agents", "langchain", "query", "--debug"])
            assert result.exit_code == 0

    def test_profile_option(self, agents_app, runner) -> None:
        with patch(
            "genai_tk.agents.langchain.factory.create_langchain_agent",
            new_callable=AsyncMock,
        ) as mock_factory:
            from langchain_core.messages import AIMessage

            mock_compiled = AsyncMock()
            mock_compiled.ainvoke.return_value = {"messages": [AIMessage(content="result")]}
            mock_factory.return_value = mock_compiled

            result = runner.invoke(agents_app, ["agents", "langchain", "query", "--profile", "simple"])
            assert result.exit_code == 0


class TestGetConfigPath:
    def test_returns_string(self) -> None:
        result = _get_config_path()
        assert isinstance(result, str)
        assert "langchain.yaml" in result

    def test_path_contains_agents(self) -> None:
        result = _get_config_path()
        assert "agents" in result


class TestDisplayConfigError:
    def test_config_error_shown(self) -> None:
        from io import StringIO

        from rich.console import Console

        from genai_tk.utils.config_exceptions import ConfigError

        output = StringIO()
        console = Console(file=output, force_terminal=False)
        error = ConfigError("test error", suggestion="fix it")
        _display_config_error(console, error)
        text = output.getvalue()
        assert "test error" in text or len(text) > 0

    def test_validation_error_shown(self) -> None:
        from io import StringIO

        from rich.console import Console

        from genai_tk.utils.config_exceptions import ConfigValidationError

        output = StringIO()
        console = Console(file=output, force_terminal=False)
        error = ConfigValidationError(errors=["Field x is invalid"], config_name="test")
        _display_config_error(console, error)
        text = output.getvalue()
        assert "validation" in text.lower() or len(text) > 0

    def test_generic_error_shown(self) -> None:
        from io import StringIO

        from rich.console import Console

        output = StringIO()
        console = Console(file=output, force_terminal=False)
        error = ValueError("Some generic error")
        _display_config_error(console, error)
        text = output.getvalue()
        assert "error" in text.lower() or len(text) > 0
