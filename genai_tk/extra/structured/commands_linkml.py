"""LinkML schema, code generation, and structured extraction CLI commands.

Usage:
    uv run cli linkml run resume -i "John Smith; SW engineer"
    uv run cli linkml run delivery_note --file note.md --out-dir ./output
    uv run cli linkml extract ./docs ./output --config delivery_note --pathspec '**/*.md'
    uv run cli linkml generate schema.yaml --out generated.py
    uv run cli linkml validate-drift schema.yaml generated.py
    uv run cli linkml show-docs schema.yaml
    uv run cli linkml show-glossary schema.yaml
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from genai_tk.cli.base import CliTopCommand
from genai_tk.config_mgmt.features import require_feature

require_feature("linkml", context="cli linkml")

from genai_tk.extra.structured.linkml_contract import LinkMLContract  # noqa: E402
from genai_tk.workflow.prefect.flows.linkml_flow import (  # noqa: E402
    linkml_extraction_flow,
    linkml_single_input_flow,
)


class LinkMLCommands(CliTopCommand):
    def get_description(self) -> tuple[str, str]:
        return "linkml", "LinkML extraction, schema generation, and drift detection commands."

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command("run")
        def run(
            config_name: Annotated[
                str,
                typer.Argument(help="Name of the LinkML extraction config from YAML (e.g. resume, delivery_note)"),
            ] = "default",
            input_text: Annotated[
                str | None,
                typer.Option("--input", "-i", help="Input text or file path to process (reads stdin if omitted)"),
            ] = None,
            file_path: Annotated[
                str | None,
                typer.Option("--file", "-f", help="Input file path (PDF, text, Markdown) to process"),
            ] = None,
            schema_path: Annotated[
                str | None,
                typer.Option("--schema", help="Override path to LinkML YAML schema file"),
            ] = None,
            root_class: Annotated[
                str | None,
                typer.Option("--root-class", help="Override root class name"),
            ] = None,
            generated_module: Annotated[
                str | None,
                typer.Option("--module", help="Override Python generated module import path"),
            ] = None,
            prompt: Annotated[
                str | None,
                typer.Option("--prompt", help="Override prompt template or prompt file path"),
            ] = None,
            llm: Annotated[
                str | None,
                typer.Option(help="LLM model identifier or tag override"),
            ] = None,
            output_dir: Annotated[
                str | None,
                typer.Option("--out-dir", help="Output directory for result"),
            ] = None,
            output_file: Annotated[
                str | None,
                typer.Option("--out-file", help="Output filename (.json)"),
            ] = None,
            force: bool = typer.Option(False, "--force", help="Force reprocessing even if cached"),
        ) -> None:
            """Execute LinkML structured extraction on a single input text or file."""
            console = Console()

            if file_path:
                input_text = file_path

            if output_dir and not output_file:
                cand_name = file_path or input_text or config_name
                if cand_name and Path(os.path.expandvars(cand_name)).is_file():
                    output_file = f"{Path(os.path.expandvars(cand_name)).stem}.json"
                else:
                    output_file = f"{config_name or 'result'}.json"
            elif output_file and not output_dir:
                output_dir = "."

            if output_file and not output_file.endswith(".json"):
                console.print("[red]Error: Output filename must have .json extension[/red]")
                raise typer.Exit(1)

            # Read from stdin if no input provided
            if not input_text:
                if not sys.stdin.isatty():
                    input_text = sys.stdin.read().strip()
                else:
                    console.print("[red]Error: No input provided via --input, --file, or stdin[/red]")
                    raise typer.Exit(1)

            try:
                result, model_name, out_path = linkml_single_input_flow(
                    input_text=input_text,
                    config_name=config_name,
                    schema_path=schema_path,
                    root_class=root_class,
                    generated_module=generated_module,
                    prompt=prompt,
                    llm=llm or "default",
                    output_dir=output_dir,
                    output_file=output_file,
                    force=force,
                )
            except Exception as exc:
                console.print(f"[red]Error executing LinkML extraction: {exc}[/red]")
                logger.exception("LinkML run error")
                raise typer.Exit(1) from exc

            if output_dir and output_file:
                console.print(
                    f"[green]Saved {model_name} result to:[/green] {Path(output_dir) / (out_path or output_file)}"
                )
            else:
                json_text = result.model_dump_json(indent=2)
                console.print(Panel(json_text, title=f"[bold]{model_name}[/bold]", expand=False))

        @cli_app.command("extract")
        def extract(
            sources: Annotated[
                str,
                typer.Argument(help="Root directory or file to extract structured data from"),
            ],
            output_dir: Annotated[
                str,
                typer.Argument(help="Output directory for structured JSON files and manifest.json"),
            ],
            config_name: Annotated[
                str,
                typer.Option("--config", "-c", help="Name of structured config from YAML"),
            ] = "default",
            schema_path: Annotated[
                str | None,
                typer.Option("--schema", help="Override path to LinkML YAML schema file"),
            ] = None,
            root_class: Annotated[
                str | None,
                typer.Option("--root-class", help="Override root class name"),
            ] = None,
            generated_module: Annotated[
                str | None,
                typer.Option("--module", help="Override Python generated module import path"),
            ] = None,
            prompt: Annotated[
                str | None,
                typer.Option("--prompt", help="Override prompt template or prompt file path"),
            ] = None,
            llm: Annotated[
                str | None,
                typer.Option(help="LLM model identifier or tag override"),
            ] = None,
            pathspec: Annotated[
                list[str] | None,
                typer.Option("--pathspec", "-p", help="Glob pattern(s) to match files (e.g. '**/*.md')"),
            ] = None,
            batch_size: Annotated[
                int,
                typer.Option(help="Concurrent batch size"),
            ] = 5,
            force: bool = typer.Option(False, "--force", help="Reprocess files even if unchanged"),
        ) -> None:
            """Batch extract structured data from files into an output directory using LinkML schemas."""
            console = Console()
            try:
                manifest = linkml_extraction_flow(
                    base_dir=sources,
                    output_dir=output_dir,
                    config_name=config_name,
                    schema_path=schema_path,
                    root_class=root_class,
                    generated_module=generated_module,
                    prompt=prompt,
                    llm=llm or "default",
                    pathspecs=pathspec,
                    batch_size=batch_size,
                    force=force,
                )
                console.print(
                    f"[green]Extraction complete:[/green] {len(manifest.entries)} files processed for model '{manifest.model_name}'"
                )
            except Exception as exc:
                console.print(f"[red]Extraction failed: {exc}[/red]")
                logger.exception("LinkML extract error")
                raise typer.Exit(1) from exc

        @cli_app.command("generate")
        def generate(
            schema_path: Annotated[
                str,
                typer.Argument(help="Path to the LinkML YAML schema file"),
            ],
            output_file: Annotated[
                str | None,
                typer.Option("--out", "-o", help="Output Python file path (prints to stdout if not set)"),
            ] = None,
            extra_fields: Annotated[
                str,
                typer.Option("--extra-fields", help="Pydantic extra fields behavior (forbid, allow, ignore)"),
            ] = "forbid",
        ) -> None:
            """Generate Pydantic v2 models from a LinkML schema."""
            console = Console()
            contract = LinkMLContract(schema_path)
            code = contract.generate_pydantic_code(extra_fields=extra_fields)

            if output_file:
                out_p = Path(output_file)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_p.write_text(code, encoding="utf-8")
                console.print(f"[green]Successfully generated Pydantic models at:[/green] {out_p}")
            else:
                sys.stdout.write(code)

        @cli_app.command("validate-drift")
        def validate_drift(
            schema_path: Annotated[
                str,
                typer.Argument(help="Path to the LinkML YAML schema file"),
            ],
            target_file: Annotated[
                str,
                typer.Argument(help="Path to the generated Python file to compare against"),
            ],
            extra_fields: Annotated[
                str,
                typer.Option("--extra-fields", help="Pydantic extra fields behavior (forbid, allow, ignore)"),
            ] = "forbid",
        ) -> None:
            """Validate that generated Pydantic code matches target_file (fails on drift)."""
            console = Console()
            contract = LinkMLContract(schema_path)
            has_drift, diff = contract.check_drift(target_file, extra_fields=extra_fields)

            if has_drift:
                console.print(f"[red]Error: Drift detected between {schema_path} and {target_file}:[/red]")
                console.print(diff)
                raise typer.Exit(1)
            else:
                console.print(f"[green]No drift detected for {target_file}.[/green]")

        @cli_app.command("show-docs")
        def show_docs(
            schema_path: Annotated[
                str,
                typer.Argument(help="Path to the LinkML YAML schema file"),
            ],
        ) -> None:
            """Display neutral schema documentation (classes, fields, enums)."""
            console = Console()
            contract = LinkMLContract(schema_path)
            docs = contract.get_schema_documentation()

            console.print(f"[bold cyan]Classes ({len(docs.classes)}):[/bold cyan]")
            for cls_name, cls_desc in docs.classes.items():
                console.print(f"  [bold]{cls_name}[/bold]: {cls_desc or '(no description)'}")
                fields = docs.fields.get(cls_name, {})
                for f_name, f_desc in fields.items():
                    console.print(f"    - [italic]{f_name}[/italic]: {f_desc}")

            console.print(f"\n[bold cyan]Enums ({len(docs.enums)}):[/bold cyan]")
            for e_name, e_doc in docs.enums.items():
                console.print(f"  [bold]{e_name}[/bold]: {e_doc.description or '(no description)'}")
                for v_name, v_desc in e_doc.values.items():
                    console.print(f"    - {v_name}: {v_desc}")

        @cli_app.command("show-glossary")
        def show_glossary(
            schema_path: Annotated[
                str,
                typer.Argument(help="Path to the LinkML YAML schema file"),
            ],
        ) -> None:
            """Display the formatted enum glossary generated for extraction prompts."""
            console = Console()
            contract = LinkMLContract(schema_path)
            glossary = contract.get_enum_glossary()
            if glossary:
                console.print(glossary)
            else:
                console.print("[dim]No enums defined in this schema.[/dim]")
