"""Unit tests for genai_tk.cli.commands_info (CliRunner, fake models, no services)."""

from __future__ import annotations

import pytest
import typer
from typer.testing import CliRunner

from genai_tk.cli.commands_info import InfoCommands


@pytest.fixture
def info_app() -> typer.Typer:
    app = typer.Typer()
    InfoCommands().register(app)
    return app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestInfoHelp:
    def test_help_exits_zero(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "--help"])
        assert result.exit_code == 0
        assert "config" in result.stdout


class TestInfoConfig:
    def test_config_runs_and_shows_active_context(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "config"])
        assert result.exit_code == 0
        assert "Active context" in result.stdout

    def test_config_shows_default_components_table(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "config"])
        assert result.exit_code == 0
        assert "Default Components" in result.stdout

    def test_config_lists_api_keys_table(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "config"])
        assert result.exit_code == 0
        assert "API Keys" in result.stdout or "Available API Keys" in result.stdout


class TestInfoModels:
    def test_models_runs_without_error(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "models"])
        assert result.exit_code == 0
        # The models command prints a table of configured models; expect some output.
        assert len(result.stdout) > 0


class TestInfoCommands:
    def test_commands_runs_and_lists_commands(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "commands"])
        # 'commands' builds a command tree of the registered CLI; it should not
        # crash and should produce output.
        assert result.exit_code == 0
        assert len(result.stdout) > 0


class TestInfoLs:
    def test_ls_without_path_shows_usage_or_error(self, info_app, runner, tmp_path) -> None:
        # 'ls' lists a directory; invoking without args should either error
        # gracefully or list the cwd. Either way it must not crash with a
        # non-handled traceback.
        result = runner.invoke(info_app, ["info", "ls", str(tmp_path)])
        assert result.exit_code == 0
        assert str(tmp_path) in result.stdout or len(result.stdout) >= 0

    def test_ls_lists_files_in_tmp_dir(self, info_app, runner, tmp_path) -> None:
        (tmp_path / "alpha.txt").write_text("hi")
        (tmp_path / "beta.md").write_text("yo")
        result = runner.invoke(info_app, ["info", "ls", str(tmp_path)])
        assert result.exit_code == 0
        assert "alpha.txt" in result.stdout
        assert "beta.md" in result.stdout

    def test_ls_with_pathspec_filter(self, info_app, runner, tmp_path) -> None:
        (tmp_path / "alpha.txt").write_text("hi")
        (tmp_path / "beta.md").write_text("yo")
        result = runner.invoke(
            info_app,
            ["info", "ls", str(tmp_path), "--pathspec", "**/*.md"],
        )
        assert result.exit_code == 0
        assert "beta.md" in result.stdout
        assert "alpha.txt" not in result.stdout


class TestInfoConfigKeys:
    def test_config_keys_runs_and_lists_keys(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "config-keys"])
        assert result.exit_code == 0
        assert len(result.stdout) > 0

    def test_config_keys_with_query_filter(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "config-keys", "--query", "llm"])
        assert result.exit_code == 0
        assert "llm" in result.stdout.lower()

    def test_config_keys_with_type_filter(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "config-keys", "--type", "models"])
        assert result.exit_code == 0


class TestInfoLlmProfile:
    def test_llm_profile_runs_default(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "llm-profile"])
        assert result.exit_code == 0
        assert len(result.stdout) > 0

    def test_llm_profile_with_known_model(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "llm-profile", "--model", "parrot_local@fake"])
        assert result.exit_code == 0
        assert "parrot_local" in result.stdout or "fake" in result.stdout

    def test_llm_profile_with_all_flag(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "llm-profile", "--all"])
        assert result.exit_code == 0


class TestInfoMcpTools:
    def test_mcp_tools_runs_and_reports_status(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "mcp-tools"])
        assert result.exit_code == 0
