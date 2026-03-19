"""CLI commands for authenticated web browser scraping.

Provides four subcommands under the ``browser`` group:

- ``capture`` – open browser (headful by default), authenticate, and save the session to disk
- ``scrape``  – navigate to a target page and print the extracted content (no LLM)
- ``run``     – query an LLM agent equipped with the scraper tool
- ``list``    – show available scraper configs and their targets

Example:
    ```
    # Capture Enedis session interactively (SSO/MFA friendly)
    cli browser capture enedis_production

    # Re-scrape a specific target and write to a file
    cli browser scrape enedis_production -t production_daily -o daily.txt

    # Ask the agent a question (LLM uses the scraped content)
    cli browser run "What was my solar production yesterday?" -c enedis_production
    cli browser run "Monthly total for May" -c enedis_production -t production_monthly -m gpt_41mini
    ```
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typer import Option

from genai_tk.cli.base import CliTopCommand


def _run(coro: object) -> None:
    """Run an async coroutine, suppressing the harmless 'event loop closed' warning
    that Playwright's subprocess transport emits on Python 3.12+ during cleanup."""
    import warnings

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(coro)  # type: ignore[arg-type]
    finally:
        # Drain pending callbacks so Playwright's subprocess transports can close cleanly
        try:
            loop.run_until_complete(asyncio.sleep(0))
        except Exception:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loop.close()
        asyncio.set_event_loop(None)


class BrowserCommands(CliTopCommand):
    """CLI commands for the browser-based authenticated web scraper."""

    def get_description(self) -> tuple[str, str]:
        return "browser", "Authenticated web scraping: capture sessions, scrape pages, run LLM agents"

    def register_sub_commands(self, cli_app: typer.Typer) -> None:  # noqa: C901
        @cli_app.command("capture")
        def capture(
            config_name: Annotated[str, typer.Argument(help="Scraper config name (key under web_scrapers: in YAML)")],
            headless: Annotated[
                bool,
                Option("--headless/--no-headless", help="Browser mode (default: visible, for interactive auth)"),
            ] = False,
            config_path: Annotated[
                Optional[str],
                Option("--config-path", "-p", help="Override path to the YAML config file"),
            ] = None,
            force: Annotated[
                bool,
                Option("--force", "-f", help="Delete any existing session before re-authenticating"),
            ] = False,
        ) -> None:
            """Authenticate and save the browser session to disk.

            Opens the browser (visible by default so you can complete any MFA /
            SSO pop-ups), runs the configured authentication flow, then saves
            the session storage.  Subsequent ``scrape`` and ``run`` calls
            reuse the saved session and skip re-authentication.

            Examples:
                ```
                cli browser capture enedis_production
                cli browser capture enedis_production --force
                cli browser capture enedis_production --headless
                ```
            """
            _run(_capture(config_name, headless=headless, config_path=config_path, force=force))

        @cli_app.command("scrape")
        def scrape(
            config_name: Annotated[str, typer.Argument(help="Scraper config name")],
            target: Annotated[
                Optional[str],
                Option("--target", "-t", help="Target name (defaults to first target in config)"),
            ] = None,
            output: Annotated[
                Optional[str],
                Option("--output", "-o", help="Write extracted content to a file instead of stdout"),
            ] = None,
        ) -> None:
            """Scrape a configured target page and print extracted content.

            Examples:
                ```
                cli browser scrape enedis_production
                cli browser scrape enedis_production --target production_daily
                cli browser scrape enedis_production -t production_monthly -o result.txt
                ```
            """
            _run(_scrape(config_name, target_name=target, output=output))

        @cli_app.command("run")
        def run(
            query: Annotated[str, typer.Argument(help="Question or task for the LLM agent")],
            config_name: Annotated[
                str,
                Option("--config", "-c", help="Scraper config name"),
            ],
            target: Annotated[
                Optional[str],
                Option("--target", "-t", help="Target name within the scraper config"),
            ] = None,
            llm: Annotated[
                Optional[str],
                Option("--llm", "-m", help="LLM identifier, e.g. gpt_41mini@openai"),
            ] = None,
            profile: Annotated[
                Optional[str],
                Option("--profile", "-p", help="Agent profile name (from langchain.yaml) to use as base"),
            ] = None,
        ) -> None:
            """Run an LLM agent equipped with the authenticated web scraper.

            Loads the scraper tool, optionally starts from an agent profile, and
            answers *query* using the scraped page content.

            Examples:
                ```
                cli browser run "What was my solar production yesterday?" -c enedis_production
                cli browser run "Monthly total for May" -c enedis_production -t production_monthly -m gpt_41mini
                cli browser run "Solar trends" -c enedis_production -p Research
                ```
            """
            _run(_run_agent(query, config_name=config_name, target_name=target, llm=llm, profile_name=profile))

        @cli_app.command("list")
        def list_scrapers(
            config_dir: Annotated[
                Optional[str],
                Option("--config-dir", "-d", help="Override the web_scrapers YAML directory"),
            ] = None,
        ) -> None:
            """List available web scraper configurations.

            Examples:
                ```
                cli browser list
                ```
            """
            _list_scrapers(config_dir)


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


async def _capture(config_name: str, *, headless: bool, config_path: str | None, force: bool) -> None:
    from genai_tk.tools.browser.config_loader import load_web_scraper_config
    from genai_tk.tools.browser.scraper_session import ScraperSession
    from genai_tk.tools.browser.session_manager import SessionManager

    console = Console()

    try:
        config = load_web_scraper_config(config_name, config_path)
    except Exception as exc:
        console.print(Panel(f"[red]Failed to load config '{config_name}': {exc}[/red]", style="red"))
        raise typer.Exit(1) from None

    if force:
        SessionManager.delete_session(config.auth, config_name)
        console.print("[dim]Existing session deleted (--force).[/dim]")

    # Override headless setting from CLI flag without mutating the shared config
    config = config.model_copy(update={"browser": config.browser.model_copy(update={"headless": headless})})

    console.print(f"[bold]Capturing session for [cyan]{config_name}[/cyan]...[/bold]")
    if not headless:
        console.print(
            "[dim]Browser window will open. Complete any MFA / SSO prompts; the session is saved automatically.[/dim]"
        )

    try:
        async with ScraperSession(config):
            pass  # auth + session save happen inside __aenter__
    except RuntimeError as exc:
        msg = str(exc)
        if "playwright is required" in msg:
            console.print(
                Panel(
                    f"[red]{msg}[/red]\n\nThen install the browser:\n  [bold]uv run playwright install chromium[/bold]",
                    title="Missing dependency",
                    style="red",
                )
            )
        elif "Executable doesn't exist" in msg or "playwright install" in msg.lower():
            console.print(
                Panel(
                    "[red]Playwright browser not found.[/red]\n\n"
                    "Run:\n  [bold]uv run playwright install chromium[/bold]",
                    title="Browser not installed",
                    style="red",
                )
            )
        else:
            console.print(Panel(f"[red]Session capture failed: {exc}[/red]", style="red"))
        raise typer.Exit(1) from None
    except Exception as exc:
        console.print(Panel(f"[red]Session capture failed: {exc}[/red]", style="red"))
        raise typer.Exit(1) from None

    state_path = SessionManager.state_path(config.auth, config_name)
    console.print(Panel(f"[green]Session saved → {state_path}[/green]", title="Done", border_style="green"))


async def _scrape(config_name: str, *, target_name: str | None, output: str | None) -> None:
    from genai_tk.tools.browser.config_loader import load_web_scraper_config
    from genai_tk.tools.browser.scraper_session import run_scraper

    console = Console()

    try:
        config = load_web_scraper_config(config_name)
    except Exception as exc:
        console.print(Panel(f"[red]Failed to load config '{config_name}': {exc}[/red]", style="red"))
        raise typer.Exit(1) from None

    effective_target = target_name or (config.targets[0].name if config.targets else "first")
    console.print(f"[bold]Scraping [cyan]{config_name}[/cyan] / [yellow]{effective_target}[/yellow]...[/bold]")

    try:
        content = await run_scraper(config, target_name)
    except Exception as exc:
        console.print(Panel(f"[red]Scrape failed: {exc}[/red]", style="red"))
        raise typer.Exit(1) from None

    if output:
        Path(output).write_text(content, encoding="utf-8")
        console.print(f"[green]Extracted content written to {output!r}[/green]")
    else:
        console.print(Panel(content, title="Extracted Content", border_style="cyan"))


async def _run_agent(
    query: str,
    *,
    config_name: str,
    target_name: str | None,
    llm: str | None,
    profile_name: str | None,
) -> None:
    from genai_tk.agents.langchain.agent_cli import run_langchain_agent_direct
    from genai_tk.agents.langchain.langchain_agent import LangchainAgent
    from genai_tk.agents.langchain.setup import setup_langchain
    from genai_tk.tools.browser.factory import create_web_scraper_tool

    console = Console()

    setup_langchain(llm, False, False, "memory")

    try:
        tools = create_web_scraper_tool(config_name, target_name)
    except Exception as exc:
        console.print(Panel(f"[red]Failed to create scraper tool for '{config_name}': {exc}[/red]", style="red"))
        raise typer.Exit(1) from None

    agent = LangchainAgent(profile_name=profile_name, llm=llm, tools=tools)
    console.print(f"[bold]Running agent with scraper [cyan]{config_name}[/cyan]...[/bold]")
    await run_langchain_agent_direct(query, agent)


def _list_scrapers(config_dir_override: str | None) -> None:
    console = Console()

    if config_dir_override:
        search_dir = Path(config_dir_override)
    else:
        try:
            from genai_tk.utils.config_mngr import global_config

            search_dir = Path(global_config().get_dir_path("paths.config")) / "web_scrapers"
        except Exception:
            search_dir = Path("config/basic/web_scrapers")

    if not search_dir.exists():
        console.print(f"[yellow]No web_scrapers config directory found at {search_dir}[/yellow]")
        return

    yaml_files = sorted(search_dir.glob("*.yaml")) + sorted(search_dir.glob("*.yml"))
    if not yaml_files:
        console.print(f"[yellow]No YAML files found in {search_dir}[/yellow]")
        return

    table = Table(title="Web Scraper Configurations", border_style="cyan")
    table.add_column("Config name", style="bold cyan")
    table.add_column("Auth type", style="yellow")
    table.add_column("Targets", style="green")
    table.add_column("File", style="dim")

    from genai_tk.tools.browser.config_loader import load_web_scraper_config

    for yaml_file in yaml_files:
        # Read the actual keys from the YAML instead of assuming key == filename stem
        scraper_keys = _keys_in_yaml(yaml_file)
        if not scraper_keys:
            table.add_row("[dim](no web_scrapers)[/dim]", "", "", yaml_file.name)
            continue
        for key in scraper_keys:
            try:
                cfg = load_web_scraper_config(key, config_path=yaml_file)
                targets = ", ".join(t.name for t in cfg.targets) if cfg.targets else "(none)"
                table.add_row(cfg.name, cfg.auth.type, targets, yaml_file.name)
            except Exception as exc:
                table.add_row(key, "[red]error[/red]", "[red]error[/red]", f"{yaml_file.name}: {exc}")

    console.print(table)


def _keys_in_yaml(yaml_file: Path) -> list[str]:
    """Return the list of scraper keys defined under *web_scrapers:* in *yaml_file*."""
    try:
        from omegaconf import OmegaConf

        with open(yaml_file) as fh:
            raw = fh.read()
        node = OmegaConf.create(raw)
        cfg = OmegaConf.to_container(node, resolve=False)
        if isinstance(cfg, dict):
            return list(cfg.get("web_scrapers", {}).keys())
    except Exception:
        pass
    return []
