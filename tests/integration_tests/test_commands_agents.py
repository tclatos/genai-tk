"""Integration tests for the unified ``cli agents`` commands (``run`` | ``list``).

The ``agents`` group exposes a single ``run`` (any LangChain or DeerFlow profile
via the harness layer) and ``list``. These tests drive the Typer app the same way
the real CLI is wired (``AgentCommands().register``) and stub ``create_harness``
with a fake harness so no real model is invoked.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from genai_tk.agents.commands_agents import AgentCommands
from genai_tk.agents.harness import EndEvent, TokenEvent


@pytest.fixture
def agents_app() -> typer.Typer:
    """Create app with AgentCommands registered (mirrors actual CLI structure)."""
    app = typer.Typer()
    AgentCommands().register(app)
    return app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class _FakeHarness:
    """Minimal stand-in harness yielding one token then ending."""

    name = "fake"

    def __init__(self, text: str = "42") -> None:
        self._text = text

    async def astream(self, message: str, *, thread_id: str | None = None) -> AsyncIterator:
        yield TokenEvent(text=self._text)
        yield EndEvent()

    async def aclose(self) -> None:
        return None


class TestAgentsGroupHelp:
    def test_help_exits_zero(self, agents_app, runner) -> None:
        result = runner.invoke(agents_app, ["agents", "--help"])
        assert result.exit_code == 0

    def test_help_lists_run_and_list_only(self, agents_app, runner) -> None:
        result = runner.invoke(agents_app, ["agents", "--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "list" in result.stdout
        # The framework-specific subcommands are gone.
        assert "langchain" not in result.stdout
        assert "deerflow" not in result.stdout


class TestListProfiles:
    def test_list_command_exits_zero(self, agents_app, runner) -> None:
        result = runner.invoke(agents_app, ["agents", "list"])
        assert result.exit_code == 0

    def test_list_shows_profiles_and_kind(self, agents_app, runner) -> None:
        result = runner.invoke(agents_app, ["agents", "list"])
        assert result.exit_code == 0
        # The table title or a header row should mention profiles.
        assert "Profile" in result.stdout or "Agent" in result.stdout


class TestRunCommand:
    def test_run_with_fake_harness(self, agents_app, runner) -> None:
        with patch("genai_tk.agents.harness.create_harness", return_value=_FakeHarness("hello")):
            result = runner.invoke(agents_app, ["agents", "run", "simple", "hi"])
        assert result.exit_code == 0
        assert "hello" in result.stdout

    def test_run_default_profile_when_omitted(self, agents_app, runner) -> None:
        # No positional profile → default profile is resolved; the query comes via stdin.
        with patch("genai_tk.agents.harness.create_harness", return_value=_FakeHarness("ok")):
            result = runner.invoke(agents_app, ["agents", "run"], input="hi")
        assert result.exit_code == 0
        assert "Using default profile" in result.stdout
        assert "ok" in result.stdout

    def test_run_passes_overrides_to_create_harness(self, agents_app, runner) -> None:
        with patch("genai_tk.agents.harness.create_harness", return_value=_FakeHarness("x")) as mock_ch:
            result = runner.invoke(
                agents_app,
                ["agents", "run", "research", "hi", "--mode", "ultra", "--sandbox", "docker", "--mcp", "tavily-mcp"],
            )
        assert result.exit_code == 0
        _key, kwargs = mock_ch.call_args
        assert kwargs.get("mode_override") == "ultra"
        assert kwargs.get("sandbox_override") == "docker"
        assert "tavily-mcp" in (kwargs.get("extra_mcp") or [])

    def test_run_unknown_profile_exits_one(self, agents_app, runner) -> None:
        with patch("genai_tk.agents.harness.create_harness", side_effect=ValueError("not found")):
            result = runner.invoke(agents_app, ["agents", "run", "ghost", "hi"])
        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_run_no_query_no_chat_exits_one(self, agents_app, runner) -> None:
        # CliRunner stdin is empty and not a TTY → stdin read yields "" → error.
        with patch("genai_tk.agents.harness.create_harness", return_value=_FakeHarness()):
            result = runner.invoke(agents_app, ["agents", "run", "simple"])
        assert result.exit_code == 1
        assert "query" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_run_json_output(self, agents_app, runner) -> None:
        with patch("genai_tk.agents.harness.create_harness", return_value=_FakeHarness("jsontext")):
            result = runner.invoke(agents_app, ["agents", "run", "simple", "hi", "--json"])
        assert result.exit_code == 0
        # NDJSON events are printed one per line; the token event carries the text.
        assert "jsontext" in result.stdout


class TestRunOptions:
    def test_run_exposes_expected_flags(self, agents_app) -> None:
        """The run command must expose the unified cross-harness flags."""
        click_group = typer.main.get_command(agents_app)
        run_cmd = click_group.commands["agents"].commands["run"]
        option_names = {opt for param in run_cmd.params for opt in param.opts}
        for flag in ("--chat", "--mode", "--sandbox", "--mcp", "--llm", "--json", "--trace"):
            assert flag in option_names, f"run is missing flag {flag}"
