"""CLI command group ``auip`` – expose agent harnesses over AUIP and AG-UI protocol.

Commands
--------
- ``cli auip serve`` – start the FastAPI AUIP / AG-UI gateway with uvicorn
- ``cli auip info``  – show configured endpoints and available agent profiles

Registration
------------
Add to ``cli.commands`` in ``app_conf.yaml``:

    ```yaml
    cli:
      commands:
        - genai_tk.protocols.auip.cli_commands.AuipCommands
    ```
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from genai_tk.cli.base import CliTopCommand
from genai_tk.protocols.auip.config import auip_config


class AuipCommands(CliTopCommand):
    """CLI command group for AUIP and AG-UI server operations."""

    description: str = "Serve agents over Agent-User Interaction Protocol (AUIP / AG-UI)"

    def get_description(self) -> tuple[str, str]:
        return "auip", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command(name="serve")
        def serve_cmd(
            host: Annotated[str, typer.Option("--host", "-h", help="Bind host IP address")] = "",
            port: Annotated[int, typer.Option("--port", "-p", help="Bind port number")] = 0,
            profile: Annotated[str, typer.Option("--profile", help="Default agent profile key override")] = "",
            reload: Annotated[
                bool, typer.Option("--reload", help="Enable uvicorn auto-reload for development")
            ] = False,
        ) -> None:
            """Start the AUIP and AG-UI FastAPI streaming server.

            Endpoints exposed:
              - POST /agent (Standard AG-UI protocol for assistant-ui and @ag-ui/client)
              - POST /api/v1/auip/stream (Canonical AUIP SSE stream)
              - GET /profiles (List available agent profiles)
              - GET /health (Liveness probe)

            Examples:
                uv run cli auip serve
                uv run cli auip serve --profile coding --port 8080
            """
            from genai_tk.protocols.auip.server import serve

            cfg = auip_config().server
            target_host = host or cfg.host
            target_port = port or cfg.port
            target_profile = profile or cfg.default_profile

            serve(host=target_host, port=target_port, profile=target_profile, reload=reload)

        @cli_app.command(name="info")
        def info_cmd() -> None:
            """Display AUIP server configuration and available agent profiles."""
            from genai_tk.agents.harness.registry import list_harness_profiles

            cfg = auip_config().server
            console = Console()

            console.print("\n[bold cyan]🚀 Agent-User Interaction Protocol (AUIP / AG-UI) Gateway[/bold cyan]\n")
            console.print(f"  [bold]Host:[/bold]            {cfg.host}")
            console.print(f"  [bold]Port:[/bold]            {cfg.port}")
            console.print(f"  [bold]Default Profile:[/bold] {cfg.default_profile}")
            console.print("  [bold]Endpoints:[/bold]")
            console.print(f"    • AG-UI SSE:       [green]http://{cfg.host}:{cfg.port}{cfg.endpoint}[/green]")
            console.print(f"    • AUIP SSE:        [green]http://{cfg.host}:{cfg.port}/api/v1/auip/stream[/green]")
            console.print(f"    • Profiles API:    [green]http://{cfg.host}:{cfg.port}/profiles[/green]")
            console.print(f"    • Health:          [green]http://{cfg.host}:{cfg.port}/health[/green]\n")

            profiles = list_harness_profiles()
            table = Table(title=f"Available Agent Profiles ({len(profiles)})")
            table.add_column("Key", style="cyan", no_wrap=True)
            table.add_column("Name", style="bold")
            table.add_column("Kind", style="magenta")
            table.add_column("Harness", style="yellow")
            table.add_column("Description")

            for p in profiles:
                is_default = " [green](default)[/green]" if p.key == cfg.default_profile else ""
                table.add_row(
                    f"{p.key}{is_default}",
                    p.name,
                    p.kind,
                    p.harness,
                    p.description[:60] + "..." if len(p.description) > 60 else p.description,
                )

            console.print(table)
            console.print()
