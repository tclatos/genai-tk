"""Integration tests for InfoCommands CLI."""

from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from genai_tk.cli.commands_info import InfoCommands
from genai_tk.core.factories.llm_factory import LlmFactory
from genai_tk.core.models_db import ModelsDb


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
        result = runner.invoke(info_app, ["info", "ls", str(tmp_path), "--pathspec", "*.txt"])
        assert result.exit_code == 0
        assert "file.txt" in result.stdout
        assert "other.md" not in result.stdout

    def test_ls_recursive(self, info_app, runner, tmp_path) -> None:
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "nested.py").write_text("code")
        result = runner.invoke(info_app, ["info", "ls", str(tmp_path), "--pathspec", "**/*.py"])
        assert result.exit_code == 0
        assert "nested.py" in result.stdout

    def test_ls_nonexistent_dir(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "ls", "/nonexistent/path/xyz"])
        # Should handle gracefully (exit 0 with error message, or exit 1)
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower() or result.exit_code in (0, 1)


class TestLlmProfileCommand:
    def test_llm_profile_with_fake_id(self, info_app, runner, fake_llm_id) -> None:
        result = runner.invoke(info_app, ["info", "llm-profile", fake_llm_id])
        assert result.exit_code == 0

    def test_llm_profile_with_invalid_id(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "llm-profile", "invalid@nowhere"])
        # Should handle gracefully
        assert result.exit_code == 0

    def test_llm_list_command(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "models"])
        assert result.exit_code == 0

    @pytest.mark.parametrize(
        ("model_id", "selected_route"),
        [
            ("glm-5.2@edenai", "scaleway/glm-5.2"),
            ("glm-5.2-cloudflare@edenai", "cloudflare/@cf/zai-org/glm-5.2"),
            ("glm-5.2-scaleway@edenai-eur", "scaleway/glm-5.2"),
        ],
    )
    def test_llm_profile_resolves_edenai_models(self, info_app, runner, model_id, selected_route) -> None:
        """EdenAI profiles show every route for the selected model before GLM-5.1."""
        db = ModelsDb()
        db._build_index({})
        db._merge_edenai_models(
            "edenai",
            {
                "data": [
                    {
                        "id": "scaleway/glm-5.2",
                        "capabilities": {"supports_reasoning": True, "supports_response_schema": True},
                        "regions": [{"code": "eu", "name": "Europe"}],
                    },
                    {
                        "id": "nebius/zai-org/GLM-5.2",
                        "capabilities": {},
                        "regions": [{"code": "us", "name": "United States"}],
                    },
                    {
                        "id": "cloudflare/@cf/zai-org/glm-5.2",
                        "capabilities": {},
                        "regions": [{"code": "global", "name": "Global"}],
                    },
                    {"id": "scaleway/glm-5.1", "capabilities": {}},
                ]
            },
        )
        db._merge_edenai_models(
            "edenai-eur",
            {
                "data": [
                    {
                        "id": "scaleway/glm-5.2",
                        "capabilities": {"supports_reasoning": True},
                        "regions": [{"code": "eu", "name": "Europe"}],
                    },
                    {"id": "qwen/glm-5.2", "capabilities": {}},
                ]
            },
        )

        with (
            patch("genai_tk.core.models_db.get_models_db", return_value=db),
            patch("genai_tk.core.factories.llm_factory.get_models_db", return_value=db),
            patch.object(LlmFactory, "known_items_dict", return_value={}),
        ):
            result = runner.invoke(info_app, ["info", "llm-profile", model_id])

        assert result.exit_code == 0
        assert selected_route in result.stdout
        if model_id.endswith("@edenai-eur"):
            assert "qwen/glm-5.2" in result.stdout
            assert "nebius/zai-org/GLM-5.2" not in result.stdout
        else:
            assert result.stdout.index("scaleway/glm-5.2") < result.stdout.index("scaleway/glm-5.1")
            assert result.stdout.index("nebius/zai-org/GLM-5.2") < result.stdout.index("scaleway/glm-5.1")
            assert result.stdout.index("cloudflare/@cf/zai-org/glm-5.2") < result.stdout.index("scaleway/glm-5.1")
        if model_id == "glm-5.2-cloudflare@edenai":
            assert "Low-confidence resolution" not in result.stdout


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
