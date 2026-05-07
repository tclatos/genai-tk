"""CLI commands for AI Extra functionality.

This module provides command-line interface commands for:
- Running MCP React agents
- Executing SmolAgents with custom tools
- Processing PDF files with OCR
- Running Fabric patterns

The commands are registered with a Typer CLI application and provide:
- Input/output handling (including stdin)
- Configuration of LLM parameters
- Tool integration
- Batch processing capabilities
"""

import asyncio
import sys
from typing import Annotated, Any

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table
from typer import Option

from genai_tk.cli.base import CliTopCommand


def _resolve_document_flow_params(
    *,
    root_dir: str | None,
    output_dir: str | None,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
    recursive: bool,
    batch_size: int,
    force: bool,
    workflow_config: str | None,
    extra_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve parameters for markdownize/ppt2pdf from config profile or CLI args."""
    if workflow_config:
        from genai_tk.workflow.resolver import resolve_workflow_invocation

        cli_overrides: dict[str, Any] = {}
        if root_dir is not None:
            cli_overrides["root_dir"] = root_dir
        if output_dir is not None:
            cli_overrides["output_dir"] = output_dir
        if include_patterns is not None:
            cli_overrides["include_patterns"] = include_patterns
        if exclude_patterns is not None:
            cli_overrides["exclude_patterns"] = exclude_patterns
        if recursive:
            cli_overrides["recursive"] = recursive
        if batch_size != 5:
            cli_overrides["batch_size"] = batch_size
        if extra_overrides:
            cli_overrides.update(extra_overrides)

        invocation = resolve_workflow_invocation(
            workflow_config,
            cli_overrides=cli_overrides,
            force=force,
        )
        values = invocation.values
        effective_root = values.get("root_dir")
        effective_output = values.get("output_dir")
        if not effective_root or not effective_output:
            raise typer.BadParameter("Resolved workflow invocation is missing required 'root_dir' and/or 'output_dir'.")
        return {
            "root_dir": str(effective_root),
            "output_dir": str(effective_output),
            "include_patterns": values.get("include_patterns"),
            "exclude_patterns": values.get("exclude_patterns"),
            "recursive": bool(values.get("recursive", False)),
            "batch_size": int(values.get("batch_size", 5)),
            "force": force,
            "converter": values.get("converter", "markitdown"),
        }

    if root_dir is None or output_dir is None:
        raise typer.BadParameter("root_dir and output_dir are required when --config is not used.")

    return {
        "root_dir": root_dir,
        "output_dir": output_dir,
        "include_patterns": include_patterns,
        "exclude_patterns": exclude_patterns,
        "recursive": recursive,
        "batch_size": batch_size,
        "force": force,
    }


def _print_document_flow_params(console: Console, title: str, params: dict[str, Any]) -> None:
    """Render resolved document flow parameters."""
    table = Table(title=f"{title} Resolution", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    for key, value in params.items():
        table.add_row(key, str(value))
    console.print(table)


class ExtraCommands(CliTopCommand):
    def get_description(self) -> tuple[str, str]:
        return "tools", "Utilities and extra tools"

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command()
        def gpt_researcher(
            query: Annotated[str, typer.Argument(help="Research query to investigate")],
            config_name: Annotated[
                str, typer.Option("--config", "-c", help="Configuration name from gpt_researcher.yaml")
            ] = "default",
            verbose: Annotated[bool, Option("--verbose", "-v", help="Enable verbose output")] = False,
        ) -> None:
            """
            Run GPT Researcher with configuration from gpt_researcher.yaml.

            Example:
                uv run cli tools gpt-researcher "Latest developments in AI" --config detailed
                uv run cli tools gpt-researcher "Climate change impacts" --llm gpt-4o
            """
            from genai_tk.extra.gpt_researcher_helper import run_gpt_researcher

            try:
                print(f"Running GPT Researcher with config: {config_name}")
                print(f"Query: {query}")

                # Run the research
                result = asyncio.run(run_gpt_researcher(query=query, config_name=config_name, verbose=verbose))

                print("\n" + "=" * 80)
                print("RESEARCH REPORT")
                print("=" * 80)
                print(result.report)

            except Exception as e:
                print(f"Error running GPT Researcher: {str(e)}")
                if verbose:
                    import traceback

                    traceback.print_exc()

        @cli_app.command()
        def markdownize(
            root_dir: Annotated[
                str | None,
                typer.Argument(
                    help="Root directory to search for files to convert",
                ),
            ] = None,
            output_dir: Annotated[
                str | None,
                typer.Argument(
                    help="Output directory for markdown files and manifest",
                ),
            ] = None,
            include_patterns: Annotated[
                list[str] | None,
                typer.Option(
                    "--include",
                    "-i",
                    help="Glob patterns to include (e.g., '*.pdf', '*.docx'). Default: all supported formats",
                ),
            ] = None,
            exclude_patterns: Annotated[
                list[str] | None,
                typer.Option(
                    "--exclude",
                    "-e",
                    help="Glob patterns to exclude (e.g., '*_draft.pdf')",
                ),
            ] = None,
            recursive: bool = typer.Option(False, help="Search for files recursively"),
            batch_size: int = typer.Option(5, help="Number of files to process concurrently in each batch"),
            force: bool = typer.Option(False, "--force", help="Reprocess files even if unchanged in manifest"),
            converter: str = typer.Option(
                "markitdown",
                "--converter",
                help="Converter to use: 'markitdown' (default), 'mistral' (Mistral OCR, PDFs only), 'edgeparse' (fast Rust engine, PDFs only)",
            ),
            workflow_config: Annotated[
                str | None,
                typer.Option("--config", help="Workflow or workflow profile name to resolve parameters from"),
            ] = None,
            dry_run: Annotated[
                bool, typer.Option("--dry-run", help="Resolve parameters and print plan without executing")
            ] = False,
        ) -> None:
            """Convert documents to Markdown using markitdown, Mistral OCR, or edgeparse.

            Processes files from root directory using glob patterns and saves markdown
            output plus a manifest file for incremental processing. Supports parallel
            batch processing with Prefect.

            Examples:
                ```bash
                cli tools markdownize ./docs ./output --recursive

                cli tools markdownize --config markdownize_docs --dry-run

                cli tools markdownize ./data ./output --include '*.pdf' --converter mistral

                cli tools markdownize '${paths.data}' '${paths.markdown}' --recursive --force --batch-size 10
                ```
            """
            from genai_tk.extra.markdownize_prefect_flow import markdownize_flow
            from genai_tk.extra.prefect.runtime import run_flow_ephemeral

            console = Console()
            params = _resolve_document_flow_params(
                root_dir=root_dir,
                output_dir=output_dir,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                recursive=recursive,
                batch_size=batch_size,
                force=force,
                workflow_config=workflow_config,
                extra_overrides={"converter": converter} if converter != "markitdown" else {},
            )

            if dry_run:
                _print_document_flow_params(console, "Markdownize", params)
                return

            logger.info(
                f"Starting markdownize from '{params['root_dir']}' to '{params['output_dir']}' "
                f"with batch_size {params['batch_size']}",
            )

            try:
                run_flow_ephemeral(
                    markdownize_flow,
                    root_dir=params["root_dir"],
                    output_dir=params["output_dir"],
                    include_patterns=params.get("include_patterns"),
                    exclude_patterns=params.get("exclude_patterns"),
                    recursive=params["recursive"],
                    batch_size=params["batch_size"],
                    force=params["force"],
                    converter=params.get("converter", "markitdown"),
                )
            except Exception as exc:
                logger.error("Markdownize conversion failed: {}", exc)
                raise typer.Exit(1) from exc

            logger.success("Markdownize conversion completed successfully")

        @cli_app.command()
        def ppt2pdf(
            root_dir: Annotated[
                str | None,
                typer.Argument(
                    help="Root directory to search for PowerPoint files to convert",
                ),
            ] = None,
            output_dir: Annotated[
                str | None,
                typer.Argument(
                    help="Output directory for PDF files and manifest",
                ),
            ] = None,
            include_patterns: Annotated[
                list[str] | None,
                typer.Option(
                    "--include",
                    "-i",
                    help="Glob patterns to include (e.g., '*.pptx', '*.ppt'). Default: all supported formats",
                ),
            ] = None,
            exclude_patterns: Annotated[
                list[str] | None,
                typer.Option(
                    "--exclude",
                    "-e",
                    help="Glob patterns to exclude (e.g., '*_draft.pptx')",
                ),
            ] = None,
            recursive: bool = typer.Option(False, help="Search for files recursively"),
            batch_size: int = typer.Option(5, help="Number of files to process concurrently in each batch"),
            force: bool = typer.Option(False, "--force", help="Reprocess files even if unchanged in manifest"),
            workflow_config: Annotated[
                str | None,
                typer.Option("--config", help="Workflow or workflow profile name to resolve parameters from"),
            ] = None,
            dry_run: Annotated[
                bool, typer.Option("--dry-run", help="Resolve parameters and print plan without executing")
            ] = False,
        ) -> None:
            """Convert PowerPoint files to PDF using LibreOffice headless mode.

            Processes PowerPoint files (PPT, PPTX, ODP) from root directory and converts
            them to PDF using LibreOffice. Supports parallel batch processing with Prefect
            and manifest-based tracking for incremental processing.

            Examples:
                ```bash
                cli tools ppt2pdf ./presentations ./output --recursive

                cli tools ppt2pdf --config ppt2pdf_docs --dry-run

                cli tools ppt2pdf ./data ./pdfs --include '*.pptx' --force

                cli tools ppt2pdf '${paths.rainbow_ppt}' '${paths.rainbow_pdf}' --recursive --batch-size 10
                ```
            """
            from genai_tk.extra.ppt2pdf_prefect_flow import ppt2pdf_flow
            from genai_tk.extra.prefect.runtime import run_flow_ephemeral

            console = Console()
            params = _resolve_document_flow_params(
                root_dir=root_dir,
                output_dir=output_dir,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                recursive=recursive,
                batch_size=batch_size,
                force=force,
                workflow_config=workflow_config,
            )

            if dry_run:
                _print_document_flow_params(console, "PPT to PDF", params)
                return

            logger.info(
                f"Starting ppt2pdf from '{params['root_dir']}' to '{params['output_dir']}' "
                f"with batch_size {params['batch_size']}",
            )

            try:
                run_flow_ephemeral(
                    ppt2pdf_flow,
                    root_dir=params["root_dir"],
                    output_dir=params["output_dir"],
                    include_patterns=params.get("include_patterns"),
                    exclude_patterns=params.get("exclude_patterns"),
                    recursive=params["recursive"],
                    batch_size=params["batch_size"],
                    force=params["force"],
                )
            except Exception as exc:
                logger.error("PowerPoint to PDF conversion failed: {}", exc)
                raise typer.Exit(1) from exc

            logger.success("PowerPoint to PDF conversion completed successfully")

    def unmaintained_commands(cli_app: typer.Typer) -> None:
        @cli_app.command()
        def browser_agent(
            task: Annotated[str, typer.Argument(help="The task for the browser agent to execute")],
            headless: Annotated[bool, typer.Option(help="Run browser in headless mode")] = False,
            llm: Annotated[str, Option("--llm", "-m", help="LLM identifier (ID or tag from config)")] = "default",
        ) -> None:
            """Launch a browser agent to complete a given task.

            Example:
                uv run cli browser-agent "recent news on Atos" --headless
            """
            from browser_use import Agent, BrowserSession

            from genai_tk.core.factories.llm_factory import get_llm
            from genai_tk.wip.browser_use_langchain import ChatLangchain

            print(f"Running browser agent with task: {task}")
            browser_session = BrowserSession(
                headless=headless,
                window_size={"width": 800, "height": 600},
            )

            llm_model = ChatLangchain(chat=get_llm(llm=llm))
            agent = Agent(task=task, llm=llm_model, browser_session=browser_session)
            history = asyncio.run(agent.run())
            print(history.final_result())

        @cli_app.command()
        def fabric(
            pattern: Annotated[str, Option("--pattern", "-p", help="Fabric pattern name to execute")],
            verbose: Annotated[bool, Option("--verbose", "-v", help="Enable verbose output")] = False,
            debug_mode: Annotated[bool, Option("--debug", "-d", help="Enable debug mode")] = False,
            stream: Annotated[bool, Option("--stream", "-s", help="Stream output progressively")] = False,
            # temperature: float = 0.0,
            llm: Annotated[str, Option("--llm", "-m", help="LLM identifier (ID or tag from config)")] = "default",
        ) -> None:
            """Run 'fabric' pattern on standard input.

            Pattern list is here: https://github.com/danielmiessler/fabric/tree/main/patterns
            Also described here : https://github.com/danielmiessler/fabric/blob/main/patterns/suggest_pattern/user.md

            ex: echo "artificial intelligence" | uv run cli fabric -p "create_aphorisms" --llm llama-70-groq
            """

            from genai_blueprint.ai_chains.fabric_chain import get_fabric_chain
            from langchain_core.globals import set_debug, set_verbose

            set_debug(debug_mode)
            set_verbose(verbose)

            config = {"llm": llm}
            chain = get_fabric_chain(config)
            input = repr("\n".join(sys.stdin))
            input = input.replace("{", "{{").replace("}", "}}")

            if stream:
                for s in chain.stream({"pattern": pattern, "input_data": input}, config):
                    print(s, end="", flush=True)
                    print("\n")
            else:
                result = chain.invoke({"pattern": pattern, "input_data": input}, config)
                print(result)

        @cli_app.command()
        def ocr_pdf(
            file_patterns: list[str] = typer.Argument(..., help="File patterns to match PDF files (glob patterns)"),  # noqa: B008
            output_dir: str = typer.Option("./ocr_output", help="Directory to save OCR results"),
            use_cache: bool = typer.Option(True, help="Use cached OCR results if available"),
            recursive: bool = typer.Option(False, help="Search for files recursively"),
        ) -> None:
            """Process PDF files with Mistral OCR and save the results as markdown files.

            Example:
                python -m src.ai_extra.mistral_ocr ocr_pdf "*.pdf" "data/*.pdf" --output-dir=./ocr_results
            """

            from upath import UPath

            from genai_tk.extra.loaders.mistral_ocr import process_pdf_batch

            # Collect all PDF files matching the patterns
            all_files = []
            for pattern in file_patterns:
                path = UPath(pattern)

                # Handle glob patterns
                if "*" in pattern:
                    base_dir = path.parent
                    if recursive:
                        matched_files = list(base_dir.glob(f"**/{path.name}"))
                    else:
                        matched_files = list(base_dir.glob(path.name))
                    all_files.extend(matched_files)
                else:
                    # Direct file path
                    if path.exists():
                        all_files.append(path)

            # Filter for PDF files
            pdf_files = [f for f in all_files if f.suffix.lower() == ".pdf"]

            if not pdf_files:
                logger.warning("No PDF files found matching the provided patterns.")
                return

            logger.info("Found {} PDF files to process", len(pdf_files))

            # Process the files
            output_path = UPath(output_dir)
            asyncio.run(process_pdf_batch(pdf_files, output_path, use_cache))

            logger.info("OCR processing complete. Results saved to {}", output_dir)
