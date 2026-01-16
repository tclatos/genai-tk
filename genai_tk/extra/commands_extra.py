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
from typing import Annotated, Optional

import typer
from loguru import logger
from typer import Option
from upath import UPath

from genai_tk.main.cli import CliTopCommand
from genai_tk.utils.config_mngr import global_config


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
                str,
                typer.Argument(
                    help="Root directory to search for files to convert",
                ),
            ],
            output_dir: Annotated[
                str,
                typer.Argument(
                    help="Output directory for markdown files and manifest",
                ),
            ],
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
            mistral_ocr: bool = typer.Option(
                False, "--mistral-ocr", help="Use Mistral OCR for PDF processing"
            ),
        ) -> None:
            """Convert documents to Markdown using markitdown or Mistral OCR.

            Processes files from root directory using glob patterns and saves markdown
            output plus a manifest file for incremental processing. Supports parallel
            batch processing with Prefect.

            Examples:
                ```bash
                cli tools markdownize ./docs ./output --recursive

                cli tools markdownize ./data ./output --include '*.pdf' --include '*.docx' --mistral-ocr

                cli tools markdownize '${paths.data}' '${paths.markdown}' --recursive --force --batch-size 10

                cli tools markdownize ./documents ./output --exclude '*_draft*' --recursive
                ```
            """
            from genai_tk.extra.markdownize_prefect_flow import markdownize_flow
            from genai_tk.extra.prefect.runtime import run_flow_ephemeral

            logger.info(
                f"Starting markdownize from '{root_dir}' to '{output_dir}' "
                f"with batch_size {batch_size}",
            )

            try:
                run_flow_ephemeral(
                    markdownize_flow,
                    root_dir=root_dir,
                    output_dir=output_dir,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                    recursive=recursive,
                    batch_size=batch_size,
                    force=force,
                    use_mistral_ocr=mistral_ocr,
                )
            except Exception as exc:
                logger.error(f"Markdownize conversion failed: {exc}")
                raise typer.Exit(1) from exc

            logger.success("Markdownize conversion completed successfully")

    def unmaintained_commands(cli_app: typer.Typer) -> None:
        @cli_app.command()
        def browser_agent(
            task: Annotated[str, typer.Argument(help="The task for the browser agent to execute")],
            headless: Annotated[bool, typer.Option(help="Run browser in headless mode")] = False,
            llm: Annotated[Optional[str], Option("--llm", "-m", help="LLM identifier (ID or tag from config)")] = None,
        ) -> None:
            """Launch a browser agent to complete a given task.

            Example:
                uv run cli browser-agent "recent news on Atos" --headless
            """
            from browser_use import Agent, BrowserSession

            from genai_tk.core.llm_factory import get_llm_unified
            from genai_tk.wip.browser_use_langchain import ChatLangchain

            print(f"Running browser agent with task: {task}")
            browser_session = BrowserSession(
                headless=headless,
                window_size={"width": 800, "height": 600},
            )

            llm_model = ChatLangchain(chat=get_llm_unified(llm=llm))
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
            llm: Annotated[Optional[str], Option("--llm", "-m", help="LLM identifier (ID or tag from config)")] = None,
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

            config = {"llm": llm if llm else global_config().get_str("llm.models.default")}
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

            logger.info(f"Found {len(pdf_files)} PDF files to process")

            # Process the files
            output_path = UPath(output_dir)
            asyncio.run(process_pdf_batch(pdf_files, output_path, use_cache))

            logger.info(f"OCR processing complete. Results saved to {output_dir}")
