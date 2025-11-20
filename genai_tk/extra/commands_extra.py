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
            input_path: str = typer.Argument(..., help="Input file or directory path"),
            output_dir: Optional[str] = typer.Option(None, help="Output directory for markdown files"),
            use_cache: bool = typer.Option(True, help="Use cached conversion results if available"),
            recursive: bool = typer.Option(False, help="Search for files recursively in directories"),
            mistral_ocr: bool = typer.Option(
                False, help="Use Mistral OCR for PDF files (requires genai_tk.extra.loaders.mistral_ocr)"
            ),
            force: bool = typer.Option(False, help="Overwrite existing markdown files"),
            glob_pattern: Optional[str] = typer.Option(
                None, help="Glob pattern to filter files (when input_path is a directory)"
            ),
        ) -> None:
            """Convert documents to Markdown using markitdown or Mistral OCR.

            Supports multiple file types: PDF, DOCX, PPTX, XLSX, HTML, CSV, JSON, XML, images, and more.
            Uses markitdown by default, with optional Mistral OCR for PDF files.

            Examples:
                cli markdownize document.pdf
                cli markdownize ./documents --recursive --output-dir ./markdown_output
                cli markdownize ./data --glob-pattern "*.pdf" --mistral-ocr
                cli markdownize report.docx --force
            """
            from pydantic import BaseModel
            from rich.console import Console
            from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
            from upath import UPath

            from genai_tk.utils.pydantic.kv_store import PydanticStore
            from genai_tk.utils.rich_widgets import create_error_panel, create_success_panel, create_warning_panel

            console = Console()

            # Default patterns for markitdown supported files
            # markitdown support more, but need extra install (see doc)
            # Consider : https://github.com/aspose-cells-python/aspose-cells-python/blob/main/aspose/cells/plugins/markitdown_plugin/README.md

            MARKITDOWN_FORMATS_DEFAULT = {"*.pdf", "*.docx", "*.pptx"}
            MISTRAL_OCR_FORMATS = {".pdf", ".jpeg", ".jpg", ".png", ".gif", ".bmp"}

            # Validate input path
            input_upath = UPath(input_path)
            if not input_upath.exists():
                console.print(create_error_panel("Input Error", f"Input path does not exist: {input_path}"))
                return

            # Determine output directory
            if output_dir:
                output_upath = UPath(output_dir)
            else:
                if input_upath.is_file():
                    output_upath = input_upath.parent / "markdownized"
                else:
                    output_upath = input_upath / "markdownized"

            # Create output directory if it doesn't exist
            try:
                output_upath.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                console.print(
                    create_error_panel("Directory Error", f"Cannot create output directory {output_upath}: {str(e)}")
                )
                return

            # Discover files to process
            files_to_process = []

            if input_upath.is_file():
                files_to_process.append(input_upath)
            else:
                # Directory processing
                if glob_pattern:
                    if recursive:
                        matched_files = list(input_upath.glob(f"**/{glob_pattern}"))
                    else:
                        matched_files = list(input_upath.glob(glob_pattern))
                    files_to_process.extend(matched_files)
                else:
                    for pattern in MARKITDOWN_FORMATS_DEFAULT | MISTRAL_OCR_FORMATS:
                        if recursive:
                            matched_files = list(input_upath.glob(f"**/{pattern}"))
                        else:
                            matched_files = list(input_upath.glob(pattern))
                        files_to_process.extend(matched_files)

            # Remove duplicates and filter for existing files
            files_to_process = list(set(files_to_process))
            files_to_process = [f for f in files_to_process if f.is_file()]

            if not files_to_process:
                console.print(create_warning_panel("No Files", "No files found matching the criteria."))
                return

            console.print(f"[green]Found {len(files_to_process)} files to process[/green]")

            # Check for Mistral OCR availability
            mistral_available = False
            if mistral_ocr:
                try:
                    import importlib.util

                    if importlib.util.find_spec("genai_tk.extra.loaders.mistral_ocr"):
                        mistral_available = True
                        console.print("[green]Mistral OCR enabled for PDF files[/green]")
                    else:
                        raise ImportError()
                except ImportError:
                    console.print(
                        create_warning_panel(
                            "Mistral OCR Unavailable", "Mistral OCR module not found. Using markitdown for all files."
                        )
                    )
                    mistral_ocr = False

            # Define cache model and initialize cache store
            class MarkdownContent(BaseModel):
                content: str

            cache_store = None
            if use_cache:
                try:
                    cache_store = PydanticStore(kvstore_id="file", model=MarkdownContent)
                except Exception as e:
                    console.print(create_warning_panel("Cache Warning", f"Cannot initialize cache: {str(e)}"))
                    use_cache = False

            # Process files
            processed_count = 0
            skipped_count = 0
            error_count = 0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                main_task = progress.add_task("[cyan]Processing files...", total=len(files_to_process))

                for file_path in files_to_process:
                    try:
                        progress.update(main_task, description=f"[cyan]Processing {file_path.name}...")

                        # Determine output path maintaining directory structure
                        if input_upath.is_file():
                            # Single file input
                            output_file = output_upath / f"{file_path.stem}.md"
                        else:
                            # Directory input - maintain structure
                            if recursive:
                                relative_path = file_path.relative_to(input_upath)
                                output_file = output_upath / relative_path.parent / f"{file_path.stem}.md"
                            else:
                                output_file = output_upath / f"{file_path.stem}.md"

                        # Create output subdirectory if needed
                        output_file.parent.mkdir(parents=True, exist_ok=True)

                        # Check if file already exists
                        if output_file.exists() and not force:
                            progress.update(main_task, advance=1)
                            skipped_count += 1
                            continue

                        # Check cache first
                        cache_key = str(file_path)
                        cached_content = None
                        if use_cache and cache_store:
                            try:
                                cached_obj = cache_store.load_object(cache_key)
                                if cached_obj and isinstance(cached_obj, MarkdownContent):
                                    cached_content = cached_obj.content
                            except Exception:
                                pass

                        if cached_content:
                            # Use cached content
                            with open(str(output_file), "w", encoding="utf-8") as f:
                                f.write(cached_content)
                            progress.update(main_task, advance=1)
                            processed_count += 1
                            continue

                        # Convert file
                        content = None

                        # Use Mistral OCR for PDFs if requested and available
                        if mistral_ocr and file_path.suffix.lower() in MISTRAL_OCR_FORMATS and mistral_available:
                            try:
                                from genai_tk.extra.loaders.mistral_ocr import mistral_ocr as mistral_ocr_func

                                progress.update(
                                    main_task, description=f"[blue]Using Mistral OCR for {file_path.name}..."
                                )

                                # Process with Mistral OCR
                                ocr_response = mistral_ocr_func(file_path, use_cache=False)  # Disable internal cache

                                # Convert OCR response to markdown
                                content_parts = []
                                for page in ocr_response.pages:
                                    content_parts.append(f"## Page {page.index + 1}\n\n")
                                    content_parts.append(page.markdown)
                                    content_parts.append("\n\n")

                                content = "".join(content_parts)

                            except Exception as e:
                                console.print(
                                    f"[yellow]Mistral OCR failed for {file_path.name}: {str(e)}. Falling back to markitdown.[/yellow]"
                                )

                        # Use markitdown (default or fallback)
                        if content is None:
                            try:
                                from markitdown import MarkItDown

                                progress.update(
                                    main_task, description=f"[green]Using markitdown for {file_path.name}..."
                                )

                                md = MarkItDown()
                                result = md.convert(str(file_path))
                                content = result.text_content

                            except Exception as e:
                                console.print(f"[red]Failed to convert {file_path.name}: {str(e)}[/red]")
                                error_count += 1
                                progress.update(main_task, advance=1)
                                continue

                        # Save content
                        with open(str(output_file), "w", encoding="utf-8") as f:
                            f.write(content)

                        # Cache the result
                        if use_cache and cache_store and content:
                            try:
                                md_content = MarkdownContent(content=content)
                                cache_store.save_obj(cache_key, md_content)
                            except Exception:
                                pass

                        processed_count += 1
                        progress.update(main_task, advance=1)

                    except Exception as e:
                        console.print(f"[red]Error processing {file_path.name}: {str(e)}[/red]")
                        error_count += 1
                        progress.update(main_task, advance=1)

            # Display summary
            console.print("\n")
            console.print(
                create_success_panel(
                    "Processing Complete",
                    f"Processed: {processed_count} files\n"
                    f"Skipped: {skipped_count} files\n"
                    f"Errors: {error_count} files\n"
                    f"Output directory: {output_upath}",
                )
            )

            if error_count > 0:
                console.print(
                    create_warning_panel(
                        "Errors Detected",
                        f"{error_count} files encountered errors during processing. Check the output above for details.",
                    )
                )

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
