"""CLI commands for Prefect server lifecycle management.

Provides the ``cli prefect`` command group with sub-commands to start,
stop, and inspect the local Prefect server.

Usage::

    cli prefect start           # start as background daemon
    cli prefect start -f        # start in foreground (blocking)
    cli prefect stop            # stop the background daemon
    cli prefect status          # show running / stopped + URLs
    cli prefect ui              # open Prefect UI in browser
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from genai_tk.cli.base import CliTopCommand


class PrefectCommands(CliTopCommand):
    """Manage the local Prefect server."""

    description: str = "Manage the local Prefect server (start/stop/status/ui)"

    def get_description(self) -> tuple[str, str]:
        return "prefect", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command("start")
        def start(
            foreground: Annotated[
                bool,
                typer.Option("--foreground", "-f", help="Run in foreground (blocks terminal)"),
            ] = False,
        ) -> None:
            """Start the local Prefect server.

            By default starts as a background daemon (detached process).
            Use --foreground / -f to keep it in the terminal (useful for debugging).
            Stop a background server with: cli prefect stop
            """
            from genai_tk.utils.prefect_server import prefect_server

            console = Console()
            server = prefect_server()

            if server.is_running() and not foreground:
                console.print(
                    Panel(
                        f"Prefect server already running.\n"
                        f"UI:  [cyan]{server.ui_url}[/cyan]\n"
                        f"API: [dim]{server.api_url}[/dim]",
                        title="Prefect Server",
                        border_style="green",
                    )
                )
                return

            if foreground:
                console.print(f"Starting Prefect server at [cyan]{server.ui_url}[/cyan] (foreground)...")
            else:
                console.print(f"Starting Prefect server at [cyan]{server.ui_url}[/cyan] (background daemon)...")

            server.start(foreground=foreground)

            if not foreground:
                if server.is_running():
                    console.print(
                        Panel(
                            f"[green]Prefect server started[/green]\n"
                            f"UI:  [cyan]{server.ui_url}[/cyan]\n"
                            f"API: [dim]{server.api_url}[/dim]\n\n"
                            f"Stop with: [bold]cli prefect stop[/bold]",
                            title="Prefect Server",
                            border_style="green",
                        )
                    )
                else:
                    console.print(
                        Panel(
                            f"Server started but health check failed.\n"
                            f"Check manually: [cyan]{server.api_url}/health[/cyan]",
                            title="Prefect Server",
                            border_style="yellow",
                        )
                    )

        @cli_app.command("stop")
        def stop() -> None:
            """Stop the background Prefect server daemon."""
            from genai_tk.utils.prefect_server import prefect_server

            console = Console()
            server = prefect_server()

            if not server.is_running():
                console.print(Panel("[dim]Prefect server is not running.[/dim]", border_style="dim"))
                return

            server.stop()
            console.print(Panel("[yellow]Prefect server stopped.[/yellow]", border_style="yellow"))

        @cli_app.command("status")
        def status() -> None:
            """Show Prefect server running status and URLs."""
            from genai_tk.utils.prefect_server import prefect_server

            console = Console()
            server = prefect_server()

            if server.is_running():
                console.print(
                    Panel(
                        f"[green]Running[/green]\nUI:  [cyan]{server.ui_url}[/cyan]\nAPI: [dim]{server.api_url}[/dim]",
                        title="Prefect Server",
                        border_style="green",
                    )
                )
            else:
                console.print(
                    Panel(
                        "[red]Not running[/red]\n\nStart with: [bold]cli prefect start[/bold]",
                        title="Prefect Server",
                        border_style="red",
                    )
                )

        @cli_app.command("ui")
        def ui() -> None:
            """Open the Prefect UI in the default browser."""
            from genai_tk.utils.prefect_server import prefect_server

            console = Console()
            server = prefect_server()

            if not server.is_running():
                console.print(
                    Panel(
                        "[yellow]Warning:[/yellow] Prefect server does not appear to be running.\n"
                        "Start it first with: [bold]cli prefect start[/bold]",
                        border_style="yellow",
                    )
                )

            console.print(f"Opening [cyan]{server.ui_url}[/cyan] ...")
            server.open_ui()
