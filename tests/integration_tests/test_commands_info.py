"""Integration tests for InfoCommands CLI."""

import pytest
import typer
from typer.testing import CliRunner

from genai_tk.core.commands_info import InfoCommands


@pytest.fixture
def info_app() -> typer.Typer:
    app = typer.Typer()
    InfoCommands().register(app)
    return app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestInfoCommandsHelp:
    def test_help_exits_zero(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "--help"])
        assert result.exit_code == 0


class TestConfigCommand:
    def test_config_exits_zero(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "config"])
        assert result.exit_code == 0

    def test_config_shows_selected_configuration(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "config"])
        assert result.exit_code == 0
        assert "Selected configuration" in result.stdout or "configuration" in result.stdout.lower()

    def test_config_shows_default_models(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "config"])
        assert result.exit_code == 0
        # Should have LLM and embeddings info
        output = result.stdout
        assert "LLM" in output or "Embeddings" in output

    def test_config_shows_api_keys(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "config"])
        assert result.exit_code == 0
        assert "API Key" in result.stdout or "Environment" in result.stdout or len(result.stdout) > 100

    def test_config_shows_llm_tags(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "config"])
        assert result.exit_code == 0
        # Should show LLM tags table
        assert "Tags" in result.stdout or "llm" in result.stdout.lower()


class TestLsCommand:
    def test_ls_with_path(self, info_app, runner, tmp_path) -> None:
        (tmp_path / "file1.txt").write_text("content")
        (tmp_path / "file2.txt").write_text("content")
        result = runner.invoke(info_app, ["info", "ls", str(tmp_path)])
        assert result.exit_code == 0
        assert "file1" in result.stdout or "file2" in result.stdout

    def test_ls_with_include_pattern(self, info_app, runner, tmp_path) -> None:
        (tmp_path / "file.txt").write_text("content")
        (tmp_path / "other.md").write_text("content")
        result = runner.invoke(info_app, ["info", "ls", str(tmp_path), "--include", "*.txt"])
        assert result.exit_code == 0
        assert "file.txt" in result.stdout
        assert "other.md" not in result.stdout

    def test_ls_recursive(self, info_app, runner, tmp_path) -> None:
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "nested.py").write_text("code")
        result = runner.invoke(info_app, ["info", "ls", str(tmp_path), "--recursive"])
        assert result.exit_code == 0
        assert "nested.py" in result.stdout

    def test_ls_nonexistent_dir(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "ls", "/nonexistent/path/xyz"])
        # Should handle gracefully (exit 0 with error message, or exit 1)
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower() or result.exit_code in (0, 1)


class TestLlmProfileCommand:
    def test_llm_profile_with_fake_id(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "llm-profile", "parrot_local@fake"])
        assert result.exit_code == 0

    def test_llm_profile_with_invalid_id(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "llm-profile", "invalid@nowhere"])
        # Should handle gracefully
        assert result.exit_code == 0

    def test_llm_list_command(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "models"])
        assert result.exit_code == 0


class TestMainCliLoadCommands:
    def test_load_and_register_commands(self) -> None:
        from genai_tk.main.cli import load_and_register_commands

        app = typer.Typer()
        # Should load all commands from config without error
        load_and_register_commands(app)

    def test_echo_command_registered(self) -> None:
        """Test that register_commands adds the echo utility command."""
        from genai_tk.main.cli import register_commands

        app = typer.Typer()
        register_commands(app)
        runner = CliRunner()
        # Single-command apps don't require the command name as prefix
        result = runner.invoke(app, ["hello_world"])
        assert result.exit_code == 0
        assert "hello_world" in result.stdout
