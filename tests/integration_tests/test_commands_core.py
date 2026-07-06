"""Integration tests for CoreCommands CLI."""

import pytest
import typer
from typer.testing import CliRunner

from genai_tk.cli.commands_core import CoreCommands


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
        assert result.exit_code == 0
        assert "llm" in result.stdout


class TestLlmCommand:
    def test_llm_basic_invocation(self, core_app, runner, fake_llm_id) -> None:
        result = runner.invoke(core_app, ["core", "llm", "--input", "Hello", "--llm", fake_llm_id])
        assert result.exit_code == 0
        assert len(result.stdout) > 0

    def test_llm_fake_model_returns_output(self, core_app, runner, fake_llm_id) -> None:
        result = runner.invoke(core_app, ["core", "llm", "--input", "test input", "--llm", fake_llm_id])
        assert result.exit_code == 0

    def test_llm_default_model(self, core_app, runner) -> None:
        # Default is set to fake in conftest
        result = runner.invoke(core_app, ["core", "llm", "--input", "hello"])
        assert result.exit_code == 0

    def test_llm_missing_input_shows_error(self, core_app, runner, fake_llm_id) -> None:
        result = runner.invoke(core_app, ["core", "llm", "--llm", fake_llm_id])
        # Missing required --input must surface an error (non-zero exit or an
        # error message), never exit 0 with empty output.
        assert result.exit_code != 0 or "Error" in result.stdout

    def test_llm_invalid_model_shows_error(self, core_app, runner) -> None:
        result = runner.invoke(core_app, ["core", "llm", "--input", "hello", "--llm", "nonexistent@nowhere"])
        # An invalid model must surface a failure — either a non-zero exit code
        # or an error/traceback visible in stdout/stderr.
        combined = result.stdout + (result.stderr or "")
        assert result.exit_code != 0 or "error" in combined.lower(), (
            f"Expected an error for invalid model, got exit_code={result.exit_code}, output={combined!r}"
        )

    @pytest.mark.parametrize(
        ("flag", "value"),
        [
            ("--cache", "memory"),
            ("--temperature", "0.5"),
            ("--raw", None),
            ("--verbose", None),
            ("--debug", None),
        ],
    )
    def test_llm_flags_accepted(self, core_app, runner, fake_llm_id, flag, value) -> None:
        """Each optional LLM flag is accepted and the command still succeeds."""
        args = ["core", "llm", "--input", "hello", "--llm", fake_llm_id, flag]
        if value is not None:
            args.append(value)
        result = runner.invoke(core_app, args)
        assert result.exit_code == 0


class TestEmbeddCommand:
    def test_embedd_basic(self, core_app, runner, fake_embeddings_id) -> None:
        result = runner.invoke(core_app, ["core", "embedd", "test text", "--model", fake_embeddings_id])
        assert result.exit_code == 0
        assert "Embeddings Summary" in result.stdout or "Vector Length" in result.stdout

    def test_embedd_shows_vector_length(self, core_app, runner, fake_embeddings_id) -> None:
        result = runner.invoke(core_app, ["core", "embedd", "test text", "--model", fake_embeddings_id])
        assert result.exit_code == 0
        assert "768" in result.stdout  # FakeEmbeddings produces 768-dim vectors

    def test_embedd_default_model(self, core_app, runner) -> None:
        result = runner.invoke(core_app, ["core", "embedd", "test text"])
        assert result.exit_code == 0

    def test_embedd_fastembed_local_model(self, core_app, runner, monkeypatch) -> None:
        class FakeFastEmbedEmbeddings:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] * 384 for _ in texts]

            def embed_query(self, text: str) -> list[float]:
                return [0.0] * 384

        import genai_tk.utils.langchain_community_repl.fastembed_embeddings as fastembed_module

        monkeypatch.setattr(fastembed_module, "FastEmbedEmbeddings", FakeFastEmbedEmbeddings)

        result = runner.invoke(core_app, ["core", "embedd", "hello", "-m", "bge-small-en@local"])
        assert result.exit_code == 0
        assert "Embeddings Summary" in result.stdout
        assert "bge-small-en@local" in result.stdout
        assert "384" in result.stdout


class TestSimilarityCommand:
    def test_similarity_two_sentences(self, core_app, runner, fake_embeddings_id) -> None:
        result = runner.invoke(
            core_app,
            ["core", "similarity", "reference", "compare1", "compare2", "--model", fake_embeddings_id],
        )
        assert result.exit_code == 0
        assert len(result.stdout) > 0
        assert "Semantic Similarity" in result.stdout

    def test_similarity_error_single_sentence(self, core_app, runner) -> None:
        result = runner.invoke(
            core_app,
            ["core", "similarity", "only one"],
        )
        assert result.exit_code == 0
        assert "Error" in result.stdout

    def test_similarity_multiple_sentences(self, core_app, runner, fake_embeddings_id) -> None:
        result = runner.invoke(
            core_app,
            ["core", "similarity", "reference", "compare1", "compare2", "--model", fake_embeddings_id],
        )
        assert result.exit_code == 0
