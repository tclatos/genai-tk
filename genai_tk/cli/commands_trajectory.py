"""CLI commands for inspecting recorded agent trajectories (Phase 2).

Provides ``cli trajectory`` sub-commands over the local trajectory store
written by :mod:`genai_tk.utils.nemo_relay_setup` and read by
:mod:`genai_tk.utils.trajectory_store`:

- ``list`` — enumerate recorded runs.
- ``show <id>`` — render a trajectory (tree / json / messages / dot).
- ``tail`` — live tail of the current run's ATOF events.
- ``replay <id>`` — replay events in order with relative timings.
- ``export <id>`` — export a trajectory in a given format.
- ``diff <id1> <id2>`` — structural diff of two runs.
- ``skills <id>`` — show skill.load marks.
- ``stats`` — aggregate stats across runs.
- ``prune`` — retention.
- ``view`` — launch the Harbor ATIF web viewer on the store (lazy; no-op if
  ``harbor`` is not installed).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree as RichTree

from genai_tk.cli.base import CliTopCommand
from genai_tk.utils.trajectory_store import TrajectoryStore


class TrajectoryCommands(CliTopCommand):
    """Commands for inspecting recorded agent trajectories."""

    description: str = "Inspect recorded agent trajectories (ATOF)."

    def get_description(self) -> tuple[str, str]:
        return "trajectory", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        store = TrajectoryStore

        @cli_app.command("list")
        def list_cmd(
            profile: Annotated[str | None, typer.Option("--profile", help="Filter by profile name")] = None,
            since: Annotated[str | None, typer.Option("--since", help="ISO datetime; only runs started after")] = None,
            status: Annotated[str | None, typer.Option("--status", help="Filter by status (ok|error)")] = None,
        ) -> None:
            """List recorded runs from the trajectory store index."""
            console = Console()
            s = store()
            since_dt = _parse_since(since)
            runs = s.list_runs(profile=profile, since=since_dt, status=status)
            if not runs:
                console.print("[yellow]No recorded runs found.[/yellow]")
                console.print(f"[dim]store: {s.root}[/dim]")
                return
            table = Table(title="Recorded trajectories", show_header=True, header_style="bold magenta")
            table.add_column("Run id", style="cyan", no_wrap=True)
            table.add_column("Profile", style="blue")
            table.add_column("Started", style="dim")
            table.add_column("LLM", justify="right")
            table.add_column("Tools", justify="right")
            table.add_column("Tokens (in/out)", justify="right")
            table.add_column("Status")
            for r in runs:
                table.add_row(
                    r.run_id,
                    r.profile,
                    _short_ts(r.started_at),
                    str(r.n_llm_calls),
                    str(r.n_tool_calls),
                    f"{r.total_prompt_tokens}/{r.total_completion_tokens}",
                    "[green]ok[/green]" if r.status == "ok" else "[red]error[/red]",
                )
            console.print(table)
            console.print(f"[dim]{len(runs)} run(s) · store: {s.root}[/dim]")

        @cli_app.command("show")
        def show_cmd(
            run_id: Annotated[str, typer.Argument(help="Run id (root agent scope uuid)")],
            fmt: Annotated[
                str,
                typer.Option("--format", "-f", help="tree | json | messages | dot"),
            ] = "tree",
        ) -> None:
            """Render a trajectory."""
            console = Console()
            traj = store().get(run_id)
            if traj is None:
                console.print(f"[red]Run '{run_id}' not found in store {store().root}[/red]")
                raise typer.Exit(1)
            if fmt == "json":
                console.print_json(json.dumps(_trajectory_dict(traj)))
            elif fmt == "messages":
                _print_messages(console, store().messages(run_id))
            elif fmt == "dot":
                console.print(_to_dot(traj))
            else:
                _print_tree(console, traj)

        @cli_app.command("tail")
        def tail_cmd(
            n: Annotated[int, typer.Option("--n", "-n", help="Number of recent events to show")] = 20,
        ) -> None:
            """Print the last N ATOF events from the most recent run."""
            console = Console()
            s = store()
            runs = s.list_runs()
            if not runs:
                console.print("[yellow]No recorded runs to tail.[/yellow]")
                return
            run_id = runs[0].run_id
            events = s.get(run_id)
            evs = events.events if events else []
            recent = evs[-n:] if len(evs) > n else evs
            table = Table(
                title=f"Last {len(recent)} events · run {run_id}", show_header=True, header_style="bold magenta"
            )
            table.add_column("Time", style="dim")
            table.add_column("Kind")
            table.add_column("Cat")
            table.add_column("Phase")
            table.add_column("Name", style="cyan")
            for e in recent:
                table.add_row(
                    _short_ts(str(e.get("timestamp") or "")),
                    str(e.get("kind") or ""),
                    str(e.get("category") or ""),
                    str(e.get("scope_category") or ""),
                    str(e.get("name") or ""),
                )
            console.print(table)

        @cli_app.command("replay")
        def replay_cmd(
            run_id: Annotated[str, typer.Argument(help="Run id to replay")],
            delay: Annotated[
                float,
                typer.Option("--delay", help="Seconds to pause between events (0 = no wait)"),
            ] = 0.0,
        ) -> None:
            """Replay a run's events in order with relative timings."""
            import time

            console = Console()
            traj = store().get(run_id)
            if traj is None:
                console.print(f"[red]Run '{run_id}' not found.[/red]")
                raise typer.Exit(1)
            evs = traj.events
            t0 = _parse_iso(str(evs[0].get("timestamp") or "")) if evs else None
            for e in evs:
                ts = _parse_iso(str(e.get("timestamp") or ""))
                rel = (ts - t0).total_seconds() if ts and t0 else 0.0
                console.print(
                    f"[dim]{rel:7.2f}s[/dim] {e.get('kind'):<5} "
                    f"{str(e.get('category') or ''):<8} "
                    f"{str(e.get('scope_category') or ''):<5} {e.get('name')}"
                )
                if delay > 0:
                    time.sleep(delay)

        @cli_app.command("export")
        def export_cmd(
            run_id: Annotated[str, typer.Argument(help="Run id to export")],
            fmt: Annotated[
                str,
                typer.Option("--format", "-f", help="atif | atof | messages | otel"),
            ] = "atof",
            out: Annotated[str | None, typer.Option("--out", "-o", help="Write to file (default: stdout)")] = None,
        ) -> None:
            """Export a trajectory in a given format."""
            console = Console()
            s = store()
            traj = s.get(run_id)
            if traj is None:
                console.print(f"[red]Run '{run_id}' not found.[/red]")
                raise typer.Exit(1)
            if fmt == "atof":
                payload = json.dumps({"events": traj.events}, indent=2)
            elif fmt == "messages":
                payload = json.dumps(s.messages(run_id), indent=2)
            elif fmt == "atif":
                payload = json.dumps(_to_atif(traj), indent=2)
            elif fmt == "otel":
                payload = json.dumps(_to_otel_spans(traj), indent=2)
            else:
                console.print(f"[red]Unknown format: {fmt}[/red]")
                raise typer.Exit(1)
            if out:
                Path(out).write_text(payload, encoding="utf-8")
                console.print(f"[green]Exported[/green] {run_id} ({fmt}) → {out}")
            else:
                console.print(payload)

        @cli_app.command("diff")
        def diff_cmd(
            run_id_a: Annotated[str, typer.Argument(help="First run id")],
            run_id_b: Annotated[str, typer.Argument(help="Second run id")],
        ) -> None:
            """Structural diff of two runs (tools, skills, steps, tokens)."""
            console = Console()
            d = store().diff(run_id_a, run_id_b)
            if "error" in d:
                console.print(f"[red]{d['error']}[/red]")
                raise typer.Exit(1)
            console.print_json(json.dumps(d))

        @cli_app.command("skills")
        def skills_cmd(run_id: Annotated[str, typer.Argument(help="Run id")]) -> None:
            """Show skills loaded during a run (skill.load marks)."""
            console = Console()
            loads = store().skills(run_id)
            if not loads:
                console.print("[yellow]No skill.load marks recorded for this run.[/yellow]")
                return
            table = Table(title=f"Skills loaded · run {run_id}", show_header=True, header_style="bold magenta")
            table.add_column("Skill", style="cyan")
            table.add_column("Source", style="dim")
            table.add_column("Tool", style="dim")
            table.add_column("Time", style="dim")
            for sl in loads:
                table.add_row(sl.skill_name, sl.source or "—", sl.tool_name or "—", _short_ts(sl.timestamp))
            console.print(table)

        @cli_app.command("stats")
        def stats_cmd(
            since: Annotated[str | None, typer.Option("--since", help="ISO datetime cutoff")] = None,
        ) -> None:
            """Aggregate stats across runs (tokens, tool/skill frequency, latency)."""
            console = Console()
            d = store().stats(since=_parse_since(since))
            console.print_json(json.dumps(d, default=str))

        @cli_app.command("prune")
        def prune_cmd(
            keep_last: Annotated[
                int | None, typer.Option("--keep-last", help="Keep only the N most recent runs")
            ] = None,
            older_than_days: Annotated[
                int | None, typer.Option("--older-than", help="Delete runs older than N days")
            ] = None,
            yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
        ) -> None:
            """Delete runs not matching the retention policy."""
            console = Console()
            if keep_last is None and older_than_days is None:
                console.print("[yellow]Specify --keep-last and/or --older-than.[/yellow]")
                raise typer.Exit(1)
            if not yes:
                typer.confirm("Prune recorded trajectories?", abort=True)
            n = store().prune(keep_last=keep_last, older_than_days=older_than_days)
            console.print(f"[green]Pruned {n} run(s).[/green]")

        @cli_app.command("view")
        def view_cmd() -> None:
            """Launch the Harbor ATIF web viewer on the store (if installed).

            The trajectory store root holds one subdirectory per recorded run,
            which is harbor's jobs layout, so ``--jobs`` is passed explicitly to
            skip harbor's folder-type auto-detection (which fails on the store).
            """
            console = Console()
            s = store()
            try:
                subprocess.run(["harbor", "view", "--jobs", str(s.root)], check=False)  # noqa: S603,S607
            except FileNotFoundError:
                console.print(
                    "[yellow]harbor not installed.[/yellow] Install it with "
                    "[dim]uv tool install harbor[/dim] to use the web trajectory viewer."
                )


# ── Render helpers ───────────────────────────────────────────────────────────


def _print_tree(console: Console, traj: Any) -> None:
    """Render a trajectory as a scope timeline tree."""
    tree = RichTree(f"[bold cyan]{traj.profile}[/bold cyan] [dim]{traj.run_id}[/dim]")
    node = tree.add(f"[dim]started[/dim] {_short_ts(traj.started_at)} · status={traj.status}")
    for lc in traj.llm_calls:
        label = f"[blue]llm[/blue] {lc.model or '?'}"
        if lc.tool_calls:
            label += f" → {', '.join(tc.get('name', '') for tc in lc.tool_calls if isinstance(tc, dict))}"
        if lc.message:
            label += f" [dim]{_snip(lc.message, 60)}[/dim]"
        node.add(label)
    for tc in traj.tool_calls:
        node.add(f"[magenta]tool[/magenta] {tc.name} [dim]→ {_snip(tc.result or '', 60)}[/dim]")
    for sl in traj.skill_loads:
        node.add(f"[yellow]skill.load[/yellow] {sl.skill_name} [dim]({sl.source or '?'})[/dim]")
    console.print(tree)


def _print_messages(console: Console, messages: list[dict[str, Any]]) -> None:
    """Print OpenAI-format messages."""
    for m in messages:
        role = m.get("role", "")
        console.print(f"[bold]{role}[/bold]: {m.get('content', '')}")
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            console.print(f"  [dim]tool_call[/dim] {fn.get('name', '')}({fn.get('arguments', '')})")


def _to_dot(traj: Any) -> str:
    """Render the scope tree as a Graphviz dot string (best-effort flat timeline)."""
    lines = ["digraph trajectory {", "  rankdir=LR;"]
    for e in traj.events:
        if e.get("kind") == "scope" and e.get("scope_category") == "start":
            uuid = str(e.get("uuid") or "")
            parent = e.get("parent_uuid")
            label = f"{e.get('category') or ''}\\n{e.get('name') or ''}".replace('"', '\\"')
            lines.append(f'  "{uuid}" [label="{label}"];')
            if parent:
                lines.append(f'  "{parent}" -> "{uuid}";')
    lines.append("}")
    return "\n".join(lines)


def _trajectory_dict(traj: Any) -> dict[str, Any]:
    """Serialize a Trajectory to a JSON-friendly dict."""
    return {
        "run_id": traj.run_id,
        "profile": traj.profile,
        "started_at": traj.started_at,
        "ended_at": traj.ended_at,
        "status": traj.status,
        "total_prompt_tokens": traj.total_prompt_tokens,
        "total_completion_tokens": traj.total_completion_tokens,
        "llm_calls": [
            {
                "model": lc.model,
                "usage": lc.usage,
                "tool_calls": lc.tool_calls,
                "message": lc.message,
            }
            for lc in traj.llm_calls
        ],
        "tool_calls": [{"name": tc.name, "args": tc.args, "result": tc.result} for tc in traj.tool_calls],
        "skill_loads": [
            {"skill_name": sl.skill_name, "source": sl.source, "tool_name": sl.tool_name} for sl in traj.skill_loads
        ],
    }


def _to_atif(traj: Any) -> dict[str, Any]:
    """Project a Trajectory to an ATIF-v1.7-ish object (steps from messages)."""
    from genai_tk.utils.trajectory_store import TrajectoryStore

    msgs = TrajectoryStore().messages(traj.run_id)
    steps = []
    for i, m in enumerate(msgs, 1):
        step: dict[str, Any] = {"step_id": i, "source": m.get("role", "user"), "message": m.get("content", "")}
        if m.get("tool_calls"):
            step["tool_calls"] = m["tool_calls"]
        steps.append(step)
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": traj.run_id,
        "agent": {"name": traj.profile, "model_name": (traj.llm_calls[0].model if traj.llm_calls else None)},
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": traj.total_prompt_tokens,
            "total_completion_tokens": traj.total_completion_tokens,
            "total_steps": len(steps),
        },
    }


def _to_otel_spans(traj: Any) -> list[dict[str, Any]]:
    """Project ATOF scopes to a flat list of OpenTelemetry-ish span dicts."""
    spans: list[dict[str, Any]] = []
    for e in traj.events:
        if e.get("kind") == "scope":
            spans.append(
                {
                    "trace_id": traj.run_id,
                    "span_id": e.get("uuid"),
                    "parent_span_id": e.get("parent_uuid"),
                    "name": e.get("name"),
                    "category": e.get("category"),
                    "scope_category": e.get("scope_category"),
                    "timestamp": e.get("timestamp"),
                }
            )
    return spans


def _parse_iso(ts: str) -> datetime | None:
    """Parse an ISO-8601 timestamp (tolerant of a trailing Z)."""
    try:
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE101
        return None


def _short_ts(ts: str) -> str:
    """Compact local timestamp."""
    dt = _parse_iso(ts)
    if dt is None:
        return ts[:19]
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _snip(text: str, width: int = 60) -> str:
    text = text.replace("\n", " ").strip()
    return (text[: width - 1] + "…") if len(text) > width else text


def _parse_since(since: str | None) -> datetime | None:
    if not since:
        return None
    try:
        dt = datetime.fromisoformat(since)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        raise typer.BadParameter(f"Invalid datetime: {since}") from None  # noqa: EM101
