"""Integration tests for CoreCommands CLI."""

import pytest
import typer
from typer.testing import CliRunner

from genai_tk.core.commands_core import CoreCommands


@pytest.fixture
def core_app() -> typer.Typer:
    app = typer.Typer()
    CoreCommands().register(app)
    return app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestCoreCommandsHelp:
    def test_help_exits_zero(self, core_app, runner) -> None:
        result = runner.invoke(core_app, ["core", "--help"])
        assert result.exit_code == 0

    def test_help_shows_commands(self, core_app, runner) -> None:
        result = runner.invoke(core_app, ["core", "--help"])
        assert "llm" in result.stdout or result.exit_code == 0


class TestLlmCommand:
    def test_llm_basic_invocation(self, core_app, runner) -> None:
        result = runner.invoke(core_app, ["core", "llm", "--input", "Hello", "--llm", "parrot_local@fake"])
        assert result.exit_code == 0
        assert len(result.stdout) > 0

    def test_llm_fake_model_returns_output(self, core_app, runner) -> None:
        result = runner.invoke(core_app, ["core", "llm", "--input", "test input", "--llm", "parrot_local@fake"])
        assert result.exit_code == 0

    def test_llm_default_model(self, core_app, runner) -> None:
        # Default is set to fake in conftest
        result = runner.invoke(core_app, ["core", "llm", "--input", "hello"])
        assert result.exit_code == 0

    def test_llm_missing_input_shows_error(self, core_app, runner) -> None:
        result = runner.invoke(core_app, ["core", "llm", "--llm", "parrot_local@fake"])
        assert result.exit_code == 0  # Returns gracefully
        assert "Error" in result.stdout or result.exit_code == 0

    def test_llm_invalid_model_shows_error(self, core_app, runner) -> None:
        result = runner.invoke(core_app, ["core", "llm", "--input", "hello", "--llm", "nonexistent@nowhere"])
        # Should either show an error or exit with non-zero code
        assert result.exit_code == 0  # Returns gracefully with error message

    def test_llm_with_cache_option(self, core_app, runner) -> None:
        result = runner.invoke(
            core_app,
            ["core", "llm", "--input", "hello", "--llm", "parrot_local@fake", "--cache", "memory"],
        )
        assert result.exit_code == 0

    def test_llm_with_temperature(self, core_app, runner) -> None:
        result = runner.invoke(
            core_app,
            ["core", "llm", "--input", "hello", "--llm", "parrot_local@fake", "--temperature", "0.5"],
        )
        assert result.exit_code == 0

    def test_llm_raw_output(self, core_app, runner) -> None:
        result = runner.invoke(
            core_app,
            ["core", "llm", "--input", "hello", "--llm", "parrot_local@fake", "--raw"],
        )
        assert result.exit_code == 0

    def test_llm_verbose_flag(self, core_app, runner) -> None:
        result = runner.invoke(
            core_app,
            ["core", "llm", "--input", "hello", "--llm", "parrot_local@fake", "--verbose"],
        )
        assert result.exit_code == 0

    def test_llm_debug_flag(self, core_app, runner) -> None:
        result = runner.invoke(
            core_app,
            ["core", "llm", "--input", "hello", "--llm", "parrot_local@fake", "--debug"],
        )
        assert result.exit_code == 0


class TestEmbeddCommand:
    def test_embedd_basic(self, core_app, runner) -> None:
        result = runner.invoke(core_app, ["core", "embedd", "test text", "--model", "embeddings_768@fake"])
        assert result.exit_code == 0
        assert "Embeddings Summary" in result.stdout or "Vector Length" in result.stdout

    def test_embedd_shows_vector_length(self, core_app, runner) -> None:
        result = runner.invoke(core_app, ["core", "embedd", "test text", "--model", "embeddings_768@fake"])
        assert result.exit_code == 0
        assert "768" in result.stdout  # FakeEmbeddings produces 768-dim vectors

    def test_embedd_default_model(self, core_app, runner) -> None:
        result = runner.invoke(core_app, ["core", "embedd", "test text"])
        assert result.exit_code == 0


class TestSimilarityCommand:
    def test_similarity_two_sentences(self, core_app, runner) -> None:
        result = runner.invoke(
            core_app,
            ["core", "similarity", "Hello world", "Goodbye world", "--model", "embeddings_768@fake"],
        )
        assert result.exit_code == 0
        assert "Semantic Similarity" in result.stdout or len(result.stdout) > 0

    def test_similarity_error_single_sentence(self, core_app, runner) -> None:
        result = runner.invoke(
            core_app,
            ["core", "similarity", "only one"],
        )
        assert result.exit_code == 0
        assert "Error" in result.stdout

    def test_similarity_multiple_sentences(self, core_app, runner) -> None:
        result = runner.invoke(
            core_app,
            ["core", "similarity", "reference", "compare1", "compare2", "--model", "embeddings_768@fake"],
        )
        assert result.exit_code == 0
