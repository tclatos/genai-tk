"""CLI commands for monitoring and observability management.

Provides ``cli monitoring`` sub-commands for:
- Inspecting which backends are active and whether keys are set
- Starting / stopping a self-hosted LangFuse Docker instance
- Opening the LangFuse or LangSmith UI in the browser
- Tailing / clearing the local JSONL trace log
"""

from __future__ import annotations

import json
import os
import subprocess
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from genai_tk.cli.base import CliTopCommand


def _compose_file() -> Path:
    """Resolve path to the bundled LangFuse docker-compose file."""
    try:
        from genai_tk.config_mgmt.config_mngr import global_config

        project = global_config().get("paths.project", "")
        if project:
            p = Path(project) / "deploy" / "docker-compose.langfuse.yaml"
            if p.exists():
                return p
    except Exception:
        pass
    # Fallback: relative to this source file's package root
    return Path(__file__).parent.parent.parent / "deploy" / "docker-compose.langfuse.yaml"


class MonitoringCommands(CliTopCommand):
    """Monitoring and tracing management commands.

    Manage tracing backends (LangSmith, LangFuse, OTEL), inspect local
    JSONL logs, and control the self-hosted LangFuse Docker instance.
    """

    description: str = "Manage monitoring backends and inspect traces."

    def get_description(self) -> tuple[str, str]:
        return "monitoring", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:

        @cli_app.command("status")
        def status() -> None:
            """Show active monitoring backends and API key availability."""
            from genai_tk.utils.tracing import monitoring_config

            console = Console()
            cfg = monitoring_config()

            table = Table(title="Monitoring Backends", show_header=True, header_style="bold magenta")
            table.add_column("Backend", style="cyan", width=12)
            table.add_column("Active", width=8)
            table.add_column("Details", width=60)

            # LangSmith
            ls_active = cfg.is_active("langsmith")
            # LangSmith key can be stored as LANGSMITH_API_KEY or LANGCHAIN_API_KEY
            ls_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY", "")
            ls_project = os.environ.get("LANGSMITH_PROJECT", cfg.project)
            table.add_row(
                "langsmith",
                "[green]✓[/green]" if ls_active else "[dim]-[/dim]",
                f"project={ls_project}  key={'[green]set[/green]' if ls_key else '[red]not set[/red]'}",
            )

            # LangFuse
            lf_active = cfg.is_active("langfuse")
            lf_pk = cfg.langfuse.public_key or os.environ.get("LANGFUSE_PUBLIC_KEY", "")
            lf_sk = cfg.langfuse.secret_key or os.environ.get("LANGFUSE_SECRET_KEY", "")
            lf_host = cfg.langfuse.host or os.environ.get("LANGFUSE_HOST", "")
            table.add_row(
                "langfuse",
                "[green]✓[/green]" if lf_active else "[dim]-[/dim]",
                (f"host={lf_host}  keys={'[green]set[/green]' if (lf_pk and lf_sk) else '[red]not set[/red]'}"),
            )

            # OTEL
            otel_active = cfg.is_active("otel")
            otel_ep = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", cfg.otel.endpoint)
            table.add_row(
                "otel",
                "[green]✓[/green]" if otel_active else "[dim]-[/dim]",
                f"endpoint={otel_ep}",
            )

            # Local JSONL
            local_active = cfg.is_active("local")
            log_path = Path(cfg.local_log.path)
            if log_path.exists():
                size_kb = log_path.stat().st_size / 1024
                log_info = f"path={log_path}  size={size_kb:.1f} KB"
            else:
                log_info = f"path={log_path}  (not created yet)"
            table.add_row(
                "local",
                "[green]✓[/green]" if local_active else "[dim]-[/dim]",
                log_info,
            )

            console.print(table)
            if cfg.backends:
                console.print(f"[dim]Active backends: {', '.join(cfg.backends)}[/dim]")
            else:
                console.print("[yellow]No backends configured. Add to monitoring.backends in config.[/yellow]")

        @cli_app.command("start")
        def start(
            backend: Annotated[
                str, typer.Argument(help="Backend service to start. Currently supported: langfuse")
            ] = "langfuse",
        ) -> None:
            """Start a monitoring backend service via Docker Compose."""
            console = Console()
            if backend != "langfuse":
                console.print(f"[red]Unknown backend '{backend}'. Only 'langfuse' is currently supported.[/red]")
                raise typer.Exit(1)

            compose = _compose_file()
            if not compose.exists():
                console.print(
                    f"[red]Docker Compose file not found: {compose}[/red]\n"
                    "[dim]Run: just monitoring-start  to auto-download the compose file.[/dim]"
                )
                raise typer.Exit(1)

            console.print("[bold]Starting LangFuse via Docker Compose …[/bold]")
            try:
                subprocess.run(["docker", "compose", "-f", str(compose), "up", "-d"], check=True)
            except subprocess.CalledProcessError as exc:
                console.print(f"[red]docker compose failed: {exc}[/red]")
                raise typer.Exit(1) from exc

            from genai_tk.utils.tracing import monitoring_config

            host = monitoring_config().langfuse.host or "http://localhost:3000"
            console.print(f"[green]LangFuse started → {host}[/green]")
            console.print("[dim]Run: cli monitoring open langfuse  to open the UI[/dim]")

        @cli_app.command("stop")
        def stop(
            backend: Annotated[
                str, typer.Argument(help="Backend service to stop. Currently supported: langfuse")
            ] = "langfuse",
        ) -> None:
            """Stop a monitoring backend service via Docker Compose."""
            console = Console()
            if backend != "langfuse":
                console.print(f"[red]Unknown backend '{backend}'.[/red]")
                raise typer.Exit(1)

            compose = _compose_file()
            if not compose.exists():
                console.print(f"[red]Docker Compose file not found: {compose}[/red]")
                raise typer.Exit(1)

            console.print("[bold]Stopping LangFuse …[/bold]")
            try:
                subprocess.run(["docker", "compose", "-f", str(compose), "down"], check=True)
            except subprocess.CalledProcessError as exc:
                console.print(f"[red]docker compose failed: {exc}[/red]")
                raise typer.Exit(1) from exc
            console.print("[green]LangFuse stopped.[/green]")

        @cli_app.command("open")
        def open_ui(
            backend: Annotated[
                str, typer.Argument(help="Backend UI to open: langfuse | langsmith | otel")
            ] = "langfuse",
        ) -> None:
            """Open the monitoring backend UI in the browser."""
            from genai_tk.utils.tracing import monitoring_config

            cfg = monitoring_config()
            urls: dict[str, str] = {
                "langfuse": cfg.langfuse.host or "http://localhost:3000",
                "langsmith": "https://smith.langchain.com",
                "otel": cfg.otel.endpoint,
            }
            url = urls.get(backend)
            if not url:
                Console().print(f"[red]No URL configured for backend: {backend}[/red]")
                raise typer.Exit(1)
            Console().print(f"Opening [link={url}]{url}[/link]")
            webbrowser.open(url)

        @cli_app.command("tail")
        def tail(
            n: Annotated[int, typer.Option("--n", "-n", help="Number of recent entries to show")] = 20,
            json_out: Annotated[bool, typer.Option("--json", help="Output raw JSON lines instead of a table")] = False,
        ) -> None:
            """Print the last N entries from the local JSONL trace log."""
            import datetime

            from genai_tk.utils.tracing import monitoring_config

            console = Console()
            log_path = Path(monitoring_config().local_log.path)
            if not log_path.exists():
                console.print(f"[yellow]Log file not found: {log_path}[/yellow]")
                raise typer.Exit(0)

            all_lines = log_path.read_text(encoding="utf-8").splitlines()
            recent = all_lines[-n:] if len(all_lines) > n else all_lines
            # Most recent first
            recent = list(reversed(recent))

            if json_out:
                for line in recent:
                    typer.echo(line)
                return

            from rich.markup import escape

            table = Table(
                title=f"Last {len(recent)} LLM trace entries  {escape(str(log_path))}",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Time (local)", style="dim", min_width=19, no_wrap=True)
            table.add_column("Model", style="cyan", width=26)
            table.add_column("Framework", style="blue", width=12)
            table.add_column("In", justify="right", width=7)
            table.add_column("Out", justify="right", width=7)
            table.add_column("Cost $", justify="right", style="yellow", width=12)
            table.add_column("ms", justify="right", width=8)
            table.add_column("Prompt (snippet)", style="dim", width=30)
            table.add_column("Response (snippet)", style="dim", width=30)
            table.add_column("Error", style="red", width=22)

            def _to_local(ts_str: str) -> str:
                if not ts_str:
                    return ""
                try:
                    dt = datetime.datetime.fromisoformat(ts_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=datetime.timezone.utc)
                    local_dt = dt.astimezone()
                    return local_dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    return ts_str[:19].replace("T", " ")

            def _snip(text: str | None, width: int = 28) -> str:
                if not text:
                    return ""
                text = text.replace("\n", " ").strip()
                return (text[: width - 1] + "…") if len(text) > width else text

            for line in recent:
                try:
                    e = json.loads(line)
                    cost = e.get("cost_usd")
                    latency = e.get("latency_ms")
                    table.add_row(
                        _to_local(e.get("ts") or ""),
                        str(e.get("model", ""))[:26],
                        str(e.get("framework", "")),
                        str(e.get("tokens_in") or ""),
                        str(e.get("tokens_out") or ""),
                        f"{cost:.6f}" if cost is not None else "",
                        f"{latency:.0f}" if latency is not None else "",
                        _snip(e.get("prompt")),
                        _snip(e.get("response")),
                        str(e.get("error") or "")[:22],
                    )
                except Exception:
                    table.add_row("[red]parse error[/red]", "", "", "", "", "", "", "", "", line[:40])

            console.print(table)
            console.print(f"[dim]Total entries in log: {len(all_lines)}[/dim]")

        @cli_app.command("clear")
        def clear(
            yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt")] = False,
        ) -> None:
            """Truncate the local JSONL trace log file."""
            from genai_tk.utils.tracing import monitoring_config

            console = Console()
            log_path = Path(monitoring_config().local_log.path)
            if not log_path.exists():
                console.print("[yellow]Log file does not exist — nothing to clear.[/yellow]")
                return
            if not yes:
                typer.confirm(f"Clear all entries from {log_path}?", abort=True)
            log_path.write_text("", encoding="utf-8")
            console.print(f"[green]Log cleared: {log_path}[/green]")
