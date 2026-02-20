"""CLI command group ``mcp`` – expose genai-tk tools and agents as MCP servers.

Commands
--------
- ``mcp serve  --name X``  – build and serve an MCP server over stdio (blocking)
- ``mcp list   [--config]`` – print a table of all declared servers and their tools
- ``mcp generate --name X [--output file.py]`` – write a standalone server script

Registration
------------
Add to ``cli.commands`` in ``app_conf.yaml``:

    ```yaml
    cli:
      commands:
        - genai_tk.mcp.cli_commands:McpCommands
    ```
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from loguru import logger

from genai_tk.main.cli import CliTopCommand


class McpCommands(CliTopCommand):
    """CLI command group for MCP server operations."""

    description: str = "Expose tools and agents as MCP servers"

    def get_description(self) -> tuple[str, str]:
        return "mcpserver", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command()
        def start(
            name: Annotated[str, typer.Option("--name", "-n", help="Server name from config/mcp/servers.yaml")],
            config: Annotated[
                Optional[Path],
                typer.Option("--config", "-c", help="Path to servers.yaml override"),
            ] = None,
            transport: Annotated[
                str,
                typer.Option("--transport", "-t", help="Transport: stdio | sse | streamable-http"),
            ] = "stdio",
        ) -> None:
            """Start an MCP server over stdio (or sse/streamable-http).

            The server is registered in config/mcp/servers.yaml.
            It exposes LangChain tools and, optionally, an agent-as-a-tool.

            Examples:
                uv run cli mcpserver serve --name search
                uv run cli mcpserver  serve --name chinook --transport sse
            """
            from genai_tk.mcp.server_builder import serve as _serve

            try:
                _serve(name=name, config_path=config, transport=transport)
            except FileNotFoundError as e:
                logger.error(str(e))
                raise typer.Exit(1) from e
            except ValueError as e:
                logger.error(str(e))
                raise typer.Exit(1) from e

        @cli_app.command(name="list")
        def list_servers(
            config: Annotated[
                Optional[Path],
                typer.Option("--config", "-c", help="Path to servers.yaml override"),
            ] = None,
        ) -> None:
            """List all MCP servers declared in the config file.

            Prints a Rich table with server names, descriptions, tools, and
            whether an agent wrapper is configured.

            Example:
                uv run cli mcpserver list
            """
            from rich.console import Console
            from rich.table import Table

            from genai_tk.mcp.server_builder import list_servers as _list

            console = Console()

            try:
                servers = _list(config)
            except FileNotFoundError as e:
                console.print(f"[red]{e}[/red]")
                raise typer.Exit(1) from e

            if not servers:
                console.print("[yellow]No MCP servers found in config.[/yellow]")
                return

            table = Table(title="MCP Expose Servers", show_lines=True)
            table.add_column("Name", style="cyan", no_wrap=True)
            table.add_column("Description", style="white")
            table.add_column("Tools", style="green")
            table.add_column("Agent tool", style="blue")

            for s in servers:
                tools_info = "\n".join(t.factory.split(":")[-1] for t in s.tools) if s.tools else "-"
                agent_info = f"[green]✓[/green] {s.agent.name}" if s.agent and s.agent.enabled else "[dim]-[/dim]"
                table.add_row(s.name, s.description or "-", tools_info, agent_info)

            console.print(table)

        @cli_app.command()
        def generate(
            name: Annotated[str, typer.Option("--name", "-n", help="Server name from config")],
            output: Annotated[
                Optional[Path],
                typer.Option("--output", "-o", help="Output file path (default: server_<name>.py)"),
            ] = None,
            config: Annotated[
                Optional[Path],
                typer.Option("--config", "-c", help="Path to servers.yaml (embedded in script)"),
            ] = None,
        ) -> None:
            """Generate a standalone Python script for an MCP server.

            The script can be run directly with ``uv run`` or registered as a
            ``[project.scripts]`` entry in ``pyproject.toml`` for ``uvx`` use.

            Examples:
                uv run cli mcpserver generate --name search
                uv run cli mcpserver generate --name search --output scripts/mcp_search.py
                uv run cli mcpserver generate --name search --config /abs/path/servers.yaml
            """
            from rich.console import Console

            from genai_tk.mcp.script_generator import write_server_script

            console = Console()
            try:
                path = write_server_script(name, output=output, config_path=config)
                console.print(f"[green]✓[/green] Script written to [bold]{path}[/bold]")
                console.print(f"\nLaunch with:  [cyan]uv run {path.name}[/cyan]")
            except Exception as e:
                logger.error(f"Script generation failed: {e}")
                raise typer.Exit(1) from e
