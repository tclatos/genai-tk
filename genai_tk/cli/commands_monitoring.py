"""CLI commands for monitoring and observability management.

Provides ``cli monitoring`` sub-commands for:
- Inspecting which backends are active and whether keys are set
- Enabling / disabling monitoring via a local ``.genai_tk`` state file
- Opening the LangFuse or LangSmith UI (or latest trace) in the browser
- Tailing / clearing the local JSONL trace log

Docker service management (starting/stopping LangFuse itself) is handled by
``just langfuse-server-start`` / ``just langfuse-server-stop``.
"""

from __future__ import annotations

import json
import os
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from genai_tk.cli.base import CliTopCommand

if TYPE_CHECKING:
    from genai_tk.utils.tracing import MonitoringConfig

# ── State file helpers ─────────────────────────────────────────────────────────

_STATE_KEY = "monitoring"


def _state_file() -> Path:
    """Return path to the project-level ``.genai_tk`` state file."""
    try:
        from genai_tk.config_mgmt.config_mngr import paths_config

        return paths_config().project / ".genai_tk"
    except Exception:
        return Path.cwd() / ".genai_tk"


def _read_state() -> dict:
    """Read the ``.genai_tk`` state file, returning an empty dict on error."""
    sf = _state_file()
    if not sf.exists():
        return {}
    try:
        return json.loads(sf.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(data: dict) -> None:
    """Merge *data* into the ``.genai_tk`` state file and persist it."""
    current = _read_state()
    current.update(data)
    _state_file().write_text(json.dumps(current, indent=2, default=str), encoding="utf-8")


def read_monitoring_state() -> dict:
    """Return the ``monitoring`` section of the ``.genai_tk`` state file (may be empty)."""
    return _read_state().get(_STATE_KEY, {})


# ── Trace URL helpers ──────────────────────────────────────────────────────────


def _get_latest_langfuse_trace_url(cfg: "MonitoringConfig", host: str, console: Console) -> str | None:
    """Fetch the URL of the most recent LangFuse trace via the SDK."""
    try:
        from langfuse import Langfuse

        pk = cfg.langfuse.public_key or os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        sk = cfg.langfuse.secret_key or os.environ.get("LANGFUSE_SECRET_KEY", "")
        if not (pk and sk):
            console.print("[yellow]LangFuse keys not set — cannot fetch latest trace.[/yellow]")
            return None
        lf = Langfuse(public_key=pk, secret_key=sk, host=host)
        # SDK v2/v3: list traces, take the first (most recent)
        response = lf.api.trace.list(limit=1)
        items = getattr(response, "data", None) or []
        if items:
            trace_id = items[0].id
            url = f"{host}/trace/{trace_id}"
            console.print(f"[dim]Latest trace: {trace_id}[/dim]")
            return url
        console.print("[yellow]No traces found in LangFuse.[/yellow]")
        return None
    except ImportError:
        console.print("[yellow]langfuse package not installed.[/yellow]")
        return None
    except Exception as exc:
        console.print(f"[yellow]Could not fetch latest LangFuse trace: {exc}[/yellow]")
        return None


def _get_latest_langsmith_trace_url(project: str, console: Console) -> str | None:
    """Fetch the URL of the most recent LangSmith run via the SDK."""
    try:
        from langsmith import Client

        client = Client()
        runs = list(client.list_runs(project_name=project, is_root=True, limit=1, execution_order=1))
        if runs:
            run = runs[0]
            # Prefer the run.url attribute if present; otherwise build it manually
            url = getattr(run, "url", None)
            if not url:
                url = f"https://smith.langchain.com/public/{run.id}/r"
            console.print(f"[dim]Latest trace run: {run.id}[/dim]")
            return url
        console.print("[yellow]No runs found in LangSmith project '{project}'.[/yellow]")
        return None
    except ImportError:
        console.print("[yellow]langsmith package not installed.[/yellow]")
        return None
    except Exception as exc:
        console.print(f"[yellow]Could not fetch latest LangSmith trace: {exc}[/yellow]")
        return None


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
                console.print(f"[dim]Active backends (config): {', '.join(cfg.backends)}[/dim]")
            else:
                console.print("[yellow]No backends configured. Add to monitoring.backends in config.[/yellow]")

            # Show .genai_tk state file info
            ms = read_monitoring_state()
            if ms:
                sf = _state_file()
                state_active = ms.get("active_backends", [])
                started_at = ms.get("started_at", "")
                console.print(
                    f"\n[bold]Local state[/bold] ([dim]{sf}[/dim])\n"
                    f"  backends : {', '.join(state_active) if state_active else '—'}\n"
                    f"  started  : {started_at}"
                )

        @cli_app.command("start")
        def start(
            backends: Annotated[
                str,
                typer.Argument(
                    help="Comma-separated backends to record as active (e.g. langfuse,local). "
                    "Writes to .genai_tk state file."
                ),
            ] = "",
        ) -> None:
            """Enable monitoring and record active backends in the local .genai_tk state file.

            This does NOT start Docker services — use ``just langfuse-server-start`` for that.
            """
            console = Console()
            from genai_tk.utils.tracing import monitoring_config

            cfg = monitoring_config()
            active = [b.strip() for b in backends.split(",") if b.strip()] if backends else list(cfg.backends)
            if not active:
                active = ["langfuse"]

            lf_host = cfg.langfuse.host or "http://localhost:3000"
            state: dict = {
                "active_backends": active,
                "project": cfg.project,
                "started_at": datetime.now(tz=timezone.utc).isoformat(),
                "langfuse_host": lf_host,
            }
            _write_state({_STATE_KEY: state})
            sf = _state_file()
            console.print(f"[green]Monitoring enabled[/green] → backends: {', '.join(active)}")
            console.print(f"[dim]State written to {sf}[/dim]")
            if "langfuse" in active:
                console.print(f"[dim]LangFuse UI: {lf_host}[/dim]")

        @cli_app.command("stop")
        def stop() -> None:
            """Disable monitoring by clearing the .genai_tk state file entry.

            This does NOT stop Docker services — use ``just langfuse-server-stop`` for that.
            """
            console = Console()
            data = _read_state()
            if _STATE_KEY not in data:
                console.print("[yellow]Monitoring was not enabled in .genai_tk — nothing to do.[/yellow]")
                return
            data.pop(_STATE_KEY)
            _state_file().write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            console.print(f"[green]Monitoring disabled.[/green] State file: {_state_file()}")

        @cli_app.command("open")
        def open_ui(
            backend: Annotated[
                str, typer.Argument(help="Backend UI to open: langfuse | langsmith | otel")
            ] = "langfuse",
            trace: Annotated[
                bool,
                typer.Option("--trace", "-t", help="Open the latest trace URL instead of the dashboard home."),
            ] = False,
            trace_id: Annotated[
                str | None,
                typer.Option("--trace-id", help="Open a specific trace by ID."),
            ] = None,
        ) -> None:
            """Open the monitoring backend UI or a trace URL in the browser.

            With ``--trace``, fetches and opens the latest trace from the active backend.
            With ``--trace-id <id>``, opens that specific trace directly.
            """
            from genai_tk.utils.tracing import monitoring_config

            console = Console()
            cfg = monitoring_config()

            if backend == "langfuse":
                host = cfg.langfuse.host or os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
                host = host.rstrip("/")

                if trace_id:
                    url = f"{host}/trace/{trace_id}"
                elif trace:
                    url = _get_latest_langfuse_trace_url(cfg, host, console)
                    if url is None:
                        url = f"{host}/traces"
                else:
                    url = host

            elif backend == "langsmith":
                project = cfg.project or os.environ.get("LANGSMITH_PROJECT", "default")
                if trace_id:
                    url = f"https://smith.langchain.com/public/{trace_id}/r"
                elif trace:
                    url = _get_latest_langsmith_trace_url(project, console)
                    if url is None:
                        url = f"https://smith.langchain.com/projects/p/{project}"
                else:
                    url = "https://smith.langchain.com"

            elif backend == "otel":
                url = cfg.otel.endpoint
            else:
                console.print(f"[red]No URL configured for backend: {backend}[/red]")
                raise typer.Exit(1)

            console.print(f"Opening [link={url}]{url}[/link]")
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
