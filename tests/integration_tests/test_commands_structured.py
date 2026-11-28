"""Integration tests for LinkML and Structured extraction CLI commands."""

from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

try:
    from genai_tk.extra.structured.commands_linkml import LinkMLCommands
    from genai_tk.extra.structured.commands_structured import StructuredCommands
except ImportError:
    pytest.skip("linkml feature not installed", allow_module_level=True)

import genai_tk.workflow.prefect.flows.linkml_flow as flow_module
from tests.generated.resume import Resume

RESUME_SCHEMA = Path(__file__).parent.parent / "linkml_src" / "resume.yaml"
DELIVERY_NOTE_SCHEMA = Path(__file__).parent.parent / "linkml_src" / "delivery_note.yaml"
ENUMS_SCHEMA = Path(__file__).parent.parent / "linkml_src" / "enums_fixture.yaml"
GENERATED_RESUME = Path(__file__).parent.parent / "generated" / "resume.py"


@pytest.fixture
def cli_app() -> typer.Typer:
    app = typer.Typer()
    LinkMLCommands().register(app)
    StructuredCommands().register(app)
    return app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestLinkMLCommands:
    def test_linkml_help(self, cli_app: typer.Typer, runner: CliRunner) -> None:
        result = runner.invoke(cli_app, ["linkml", "--help"])
        assert result.exit_code == 0
        assert "generate" in result.stdout
        assert "validate-drift" in result.stdout
        assert "show-docs" in result.stdout
        assert "show-glossary" in result.stdout

    def test_linkml_show_docs(self, cli_app: typer.Typer, runner: CliRunner) -> None:
        result = runner.invoke(cli_app, ["linkml", "show-docs", str(RESUME_SCHEMA)])
        assert result.exit_code == 0
        assert "Resume" in result.stdout
        assert "name" in result.stdout

    def test_linkml_show_glossary(self, cli_app: typer.Typer, runner: CliRunner) -> None:
        result = runner.invoke(cli_app, ["linkml", "show-glossary", str(ENUMS_SCHEMA)])
        assert result.exit_code == 0
        assert "ProcessingStatus" in result.stdout
        assert "PENDING" in result.stdout
        assert "Awaiting initial review or ingestion." in result.stdout

    def test_linkml_validate_drift_success(self, cli_app: typer.Typer, runner: CliRunner) -> None:
        result = runner.invoke(
            cli_app,
            ["linkml", "validate-drift", str(RESUME_SCHEMA), str(GENERATED_RESUME)],
        )
        assert result.exit_code == 0
        assert "No drift detected" in result.stdout

    def test_linkml_validate_drift_failure(self, cli_app: typer.Typer, runner: CliRunner, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("# drifting code", encoding="utf-8")
        result = runner.invoke(
            cli_app,
            ["linkml", "validate-drift", str(RESUME_SCHEMA), str(bad_file)],
        )
        assert result.exit_code != 0
        assert "Drift detected" in result.stdout

    def test_linkml_generate_to_file(self, cli_app: typer.Typer, runner: CliRunner, tmp_path: Path) -> None:
        out_file = tmp_path / "gen.py"
        result = runner.invoke(
            cli_app,
            ["linkml", "generate", str(RESUME_SCHEMA), "--out", str(out_file)],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        code = out_file.read_text(encoding="utf-8")
        assert "class Resume" in code


class TestLinkMLExtractionCommands:
    def test_linkml_run_and_extract_in_help(self, cli_app: typer.Typer, runner: CliRunner) -> None:
        result = runner.invoke(cli_app, ["linkml", "--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "extract" in result.stdout

    def test_linkml_run_single_input(
        self, cli_app: typer.Typer, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_resume = Resume(
            name="Alice Smith",
            email="alice@example.com",
            experience=["Lead Developer"],
            skills=["Python", "Cloud"],
        )

        async def mock_aextract(self: Any, document_id: str, content: Any, is_pdf: bool = False) -> Resume:
            return fake_resume

        monkeypatch.setattr(flow_module.LinkMLProcessor, "aextract", mock_aextract)

        out_dir = tmp_path / "out"
        result = runner.invoke(
            cli_app,
            [
                "linkml",
                "run",
                "resume",
                "-i",
                "Alice Smith; alice@example.com",
                "--out-dir",
                str(out_dir),
                "--out-file",
                "alice.json",
            ],
        )

        assert result.exit_code == 0
        saved_file = out_dir / "Resume" / "alice.json"
        assert saved_file.exists()
        assert "Alice Smith" in saved_file.read_text(encoding="utf-8")

    def test_linkml_extract_directory(
        self, cli_app: typer.Typer, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        doc = src_dir / "person.md"
        doc.write_text("Person details", encoding="utf-8")

        fake_resume = Resume(
            name="Bob Jones",
            email="bob@example.com",
            experience=["Manager"],
            skills=["Agile"],
        )

        async def mock_aextract(self: Any, document_id: str, content: Any, is_pdf: bool = False) -> Resume:
            return fake_resume

        monkeypatch.setattr(flow_module.LinkMLProcessor, "aextract", mock_aextract)

        out_dir = tmp_path / "out"
        result = runner.invoke(
            cli_app,
            [
                "linkml",
                "extract",
                str(src_dir),
                str(out_dir),
                "--config",
                "resume",
            ],
        )

        assert result.exit_code == 0
        assert "Extraction complete" in result.stdout
        saved_file = out_dir / "Resume" / "person.json"
        assert saved_file.exists()
