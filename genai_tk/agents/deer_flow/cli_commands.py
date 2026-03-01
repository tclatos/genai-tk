"""CLI commands for Deer-flow agents (embedded client mode).

Loads DeerFlow in-process via DEER_FLOW_PATH/backend — no HTTP servers needed.
The --web option still starts the backend servers for the Next.js frontend.

Usage examples:
    cli deerflow --list
    cli deerflow "Explain quantum computing"
    cli deerflow -p "Coder" --chat
    cli deerflow -p "Research Assistant" --mode ultra --trace "Analyse AI trends"
    cli deerflow -p "Web Browser" --llm gpt_41mini@openai "Go to atos.net"
    cli deerflow -p "Research Assistant" --subagent --plan-mode "Build a report"
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from loguru import logger
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from pydantic import BaseModel
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from genai_tk.main.cli import CliTopCommand

console = Console()

# Human-readable labels for meaningful graph nodes
_NODE_LABELS: dict[str, str] = {
    # Lead-agent sub-graph nodes
    "planner": "Planning",
    "reporter": "Writing report",
    "researcher": "Researching",
    "coder": "Writing code",
    "model": "Thinking",
    "tools": "Using tools",
    # Tool nodes
    "search_tool": "Searching",
    "web_search": "Searching the web",
    "tavily_search": "Searching (Tavily)",
    "python_repl": "Running code",
    "bash": "Running shell command",
    "file_read": "Reading file",
    "file_write": "Writing file",
    "browser": "Browsing",
    # Subagent nodes
    "subagent": "Running sub-agent",
    "reflection": "Reflecting",
}


# ---------------------------------------------------------------------------
# LLM identifier -> deer-flow model_name
# ---------------------------------------------------------------------------


def _resolve_model_name(llm_identifier: str) -> str:
    """Resolve a GenAI Toolkit LLM identifier to a deer-flow model name.

    Delegates to ``LlmFactory.resolve_llm_identifier_safe`` which handles
    exact IDs, config tags, hyphen/underscore normalisation, and fuzzy
    models.dev resolution in a single call.

    The returned ID is used both as the ``model_name`` parameter for the
    embedded client and as the ``selected_llm`` filter in
    ``config_bridge.generate_deer_flow_models``.

    Args:
        llm_identifier: LLM ID, tag, or compact alias (e.g. ``gpt_oss120@openrouter``).

    Returns:
        Resolved LLM ID string.
    """
    from genai_tk.core.llm_factory import LlmFactory

    llm_id, error_msg = LlmFactory.resolve_llm_identifier_safe(llm_identifier)
    if error_msg:
        console.print(f"[red]{error_msg}[/red]")
        raise typer.Exit(1)
    return llm_id


# ---------------------------------------------------------------------------
# Server boot helper (sync wrapper used once per run)
# ---------------------------------------------------------------------------


def _validate_and_normalize_sandbox(sandbox: str) -> str:
    """Validate sandbox provider string.

    Args:
        sandbox: Raw sandbox string from the profile.

    Returns:
        Normalized sandbox string.
    """
    normalized = (sandbox or "").strip().lower() or "local"
    if normalized not in {"local", "docker"}:
        console.print(
            f"[red]Invalid sandbox value:[/red] '{sandbox}'. Expected 'local' or 'docker'. "
            "Update config/agents/deerflow.yaml."
        )
        raise typer.Exit(1)
    return normalized


def _check_docker_available() -> bool:
    """Return True if Docker appears usable by this user."""
    from shutil import which

    if which("docker") is None:
        return False

    try:
        # `docker ps` is a practical permission check (fails if daemon isn't reachable).
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True, timeout=3, check=False)
        return result.returncode == 0
    except Exception:
        return False


def _verify_written_sandbox(config_path: Path, expected_sandbox: str) -> None:
    """Verify the generated deer-flow config.yaml matches the profile sandbox.

    Args:
        config_path: Path to the generated deer-flow config.yaml.
        expected_sandbox: Expected sandbox setting (local or docker).
    """
    try:
        import yaml

        raw = yaml.safe_load(config_path.read_text()) or {}
        use_str = ((raw.get("sandbox") or {}).get("use") or "").strip()
    except Exception as e:
        logger.debug(f"Could not verify sandbox provider in {config_path}: {e}")
        return

    expected = expected_sandbox.lower()
    if expected == "docker" and "aio_sandbox_provider" not in use_str:
        console.print(
            "[yellow]Warning:[/yellow] Profile sandbox is 'docker' but generated config does not look like a Docker sandbox. "
            f"config.yaml={config_path} sandbox.use={use_str!r}"
        )
    if expected == "local" and "LocalSandboxProvider" not in use_str:
        console.print(
            "[yellow]Warning:[/yellow] Profile sandbox is 'local' but generated config does not look like LocalSandboxProvider. "
            f"config.yaml={config_path} sandbox.use={use_str!r}"
        )


async def _prepare_profile(
    profile_name: str,
    llm_override: str | None,
    extra_mcp: list[str],
    mode_override: str | None,
    verbose: bool,
    *,
    start_servers: bool = False,
) -> tuple["DeerFlowProfile", str | None, "Path"]:  # noqa: F821
    """Load, validate and prepare a profile, then write the deer-flow config.

    For embedded (terminal) mode ``start_servers=False`` (default): only writes
    ``config.yaml`` + ``extensions_config.json`` — no server restart is done.
    For ``--web`` mode pass ``start_servers=True`` to also restart the backend
    servers so the Next.js frontend has live processes to connect to.

    Args:
        profile_name: Profile name from deerflow.yaml.
        llm_override: LLM identifier override (ID or tag).
        extra_mcp: Additional MCP server names.
        mode_override: Mode override string.
        verbose: Enable DEBUG-level logging.
        start_servers: When True, restart deer-flow backend servers after config write.

    Returns:
        Tuple of (prepared DeerFlowProfile, resolved model_name or None, config_path).
    """
    from genai_tk.agents.deer_flow.config_bridge import setup_deer_flow_config
    from genai_tk.agents.deer_flow.profile import (
        DeerFlowError,
        load_deer_flow_profiles,
        validate_mcp_servers,
        validate_mode,
        validate_profile_name,
    )
    from genai_tk.utils.config_mngr import global_config

    if verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        )

    config_dir = global_config().get_dir_path("paths.config")
    config_path = str(config_dir / "agents" / "deerflow.yaml")

    try:
        profiles = load_deer_flow_profiles(config_path)
        profile = validate_profile_name(profile_name, profiles)
    except DeerFlowError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    try:
        if mode_override:
            profile.mode = validate_mode(mode_override)
        if extra_mcp:
            validated = validate_mcp_servers(extra_mcp)
            profile.mcp_servers = list(set(profile.mcp_servers + validated))
        profile.sandbox = _validate_and_normalize_sandbox(profile.sandbox)
    except DeerFlowError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    model_name: str | None = None
    if llm_override:
        model_name = _resolve_model_name(llm_override)
    elif profile.llm:
        model_name = _resolve_model_name(profile.llm)

    with console.status("Preparing Deer-flow config...", spinner="dots"):
        config_path, _ext_path = setup_deer_flow_config(
            mcp_server_names=profile.mcp_servers,
            skill_directories=profile.skill_directories,
            sandbox=profile.sandbox,
            selected_llm=model_name,
        )
        _verify_written_sandbox(config_path, profile.sandbox)

    if profile.sandbox == "docker" and not _check_docker_available():
        console.print(
            "[yellow]Warning:[/yellow] Profile sandbox is 'docker', but Docker does not look available for this user. "
            "Sandboxed code execution may fail."
        )

    if start_servers:
        await _ensure_server(profile.auto_start, profile.deer_flow_path, profile.langgraph_url, profile.gateway_url)

    return profile, model_name, config_path


async def _ensure_server(
    profile_auto_start: bool,
    deer_flow_path: str | None,
    langgraph_url: str,
    gateway_url: str,
) -> None:
    """Restart the server so the freshly generated config.yaml is picked up.

    Performs the equivalent of ``make clean && make dev``: kills processes,
    stops sandbox containers, clears logs, then starts fresh.  LangGraph runs
    with ``--no-reload``, so a restart is required for every config change.

    If ``profile_auto_start`` is False, the server must already be running; a
    reminder to restart manually is shown instead.

    Args:
        profile_auto_start: Whether the profile allows auto-starting the server.
        deer_flow_path: Override path to deer-flow clone (falls back to DEER_FLOW_PATH env).
        langgraph_url: LangGraph server base URL.
        gateway_url: Gateway API base URL.
    """
    from genai_tk.agents.deer_flow.server_manager import DeerFlowServerManager

    if not profile_auto_start:
        mgr = DeerFlowServerManager(
            langgraph_url=langgraph_url,
            gateway_url=gateway_url,
        )
        if not await mgr.is_running():
            console.print(
                "[red]Deer-flow server is not running.[/red] Start it manually or set auto_start: true in the profile."
            )
            raise typer.Exit(1)
        console.print(
            "[yellow]Note:[/yellow] Config was regenerated — restart the Deer-flow server manually to pick up changes."
        )
        return

    df_path = deer_flow_path or os.environ.get("DEER_FLOW_PATH", "")
    if not df_path:
        console.print(
            "[red]Cannot auto-start:[/red] DEER_FLOW_PATH is not set. Point it to your deer-flow clone directory."
        )
        raise typer.Exit(1)

    with console.status("Restarting Deer-flow (stop → clean → start)...", spinner="dots"):
        mgr = DeerFlowServerManager(
            deer_flow_path=df_path,
            langgraph_url=langgraph_url,
            gateway_url=gateway_url,
        )
        try:
            await mgr.restart()
            df_root = df_path
            console.print(f"[green]✓ Deer-flow servers started[/green]  [dim]Logs → {df_root}/logs/[/dim]")
        except Exception as e:
            console.print(f"[red]Failed to restart Deer-flow servers:[/red] {e}")
            raise typer.Exit(1) from e


# ---------------------------------------------------------------------------
# Apply skills from profile
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Core streaming function (single turn)
# ---------------------------------------------------------------------------


async def _stream_message(
    client: "EmbeddedDeerFlowClient",  # noqa: F821
    thread_id: str,
    user_input: str,
    model_name: str | None,
    mode: str,
    show_trace: bool,
    *,
    subagent_enabled: bool | None = None,
    plan_mode: bool | None = None,
) -> str:
    """Send a message and stream the response to the terminal.

    Displays a spinner while the agent thinks, then renders the full AI
    response (embedded mode delivers complete messages, not char-by-char tokens).
    Tool calls and results are shown as they arrive.

    Args:
        client: Embedded deer-flow client instance.
        thread_id: Conversation thread ID.
        user_input: User message.
        model_name: Deer-flow model name override (None = client default).
        mode: Agent mode string.
        show_trace: Show node-level execution trace.
        subagent_enabled: Override subagent flag for this turn.
        plan_mode: Override plan_mode flag for this turn.

    Returns:
        Accumulated full response text.
    """
    from genai_tk.agents.deer_flow.embedded_client import (
        ErrorEvent,
        NodeEvent,
        TokenEvent,
        ToolCallEvent,
        ToolResultEvent,
    )

    full_text = ""
    current_node = ""

    _PANEL_TITLE = "[bold white on royal_blue1] Assistant [/bold white on royal_blue1]"

    def _text_panel(text: str) -> Panel:
        return Panel(Text(text), title=_PANEL_TITLE, border_style="royal_blue1")

    def _md_panel(text: str) -> Panel:
        return Panel(Markdown(text), title=_PANEL_TITLE, border_style="royal_blue1", padding=(1, 2))

    with Live(Spinner("dots", text=" Thinking..."), console=console, refresh_per_second=10) as live:
        async for event in client.stream_message(
            thread_id=thread_id,
            user_input=user_input,
            model_name=model_name,
            mode=mode,
            subagent_enabled=subagent_enabled,
            plan_mode=plan_mode,
        ):
            if isinstance(event, NodeEvent):
                node_label = _NODE_LABELS.get(event.node)
                if event.node != current_node:
                    current_node = event.node
                    if node_label:  # known/meaningful node
                        if not full_text:
                            live.update(Spinner("dots", text=f" {node_label}..."))
                        else:
                            console.log(f"[dim]→ {node_label}[/dim]")
                    elif show_trace:  # unknown node — only with --trace
                        console.log(f"[dim italic]→ {event.node}[/dim italic]")
            elif isinstance(event, ToolCallEvent):
                if event.tool_name:  # always show tool calls — useful progress indicator
                    args_preview = str(event.args)[:120].replace("\n", " ") if event.args else ""
                    console.log(
                        f"[dim cyan]⚙ tool:[/dim cyan] [cyan]{event.tool_name}[/cyan] [dim]{args_preview}[/dim]"
                    )
            elif isinstance(event, ToolResultEvent):
                if show_trace and event.tool_name:
                    result_preview = event.content[:200].replace("\n", " ") if event.content else ""
                    console.log(
                        f"[dim green]✓ result:[/dim green] [dim]{event.tool_name}[/dim] [dim italic]{result_preview}[/dim italic]"
                    )
            elif isinstance(event, TokenEvent):
                full_text += event.data
                live.update(_text_panel(full_text))
            elif isinstance(event, ErrorEvent):
                live.update(Text(f"[red]Error: {event.message}[/red]"))
                console.print(f"[red]Agent error:[/red] {event.message}")
                return full_text

        # Switch to markdown rendering before Live exits so the final
        # persistent output is nicely formatted (no second print needed).
        if full_text:
            live.update(_md_panel(full_text))

    return full_text


# ---------------------------------------------------------------------------
# List profiles helper
# ---------------------------------------------------------------------------


def _get_default_profile_name() -> str | None:
    """Return the name of the first available profile, or None."""
    try:
        from genai_tk.agents.deer_flow.profile import load_deer_flow_profiles
        from genai_tk.utils.config_mngr import global_config

        config_dir = global_config().get_dir_path("paths.config")
        config_path = str(config_dir / "agents" / "deerflow.yaml")
        profiles = load_deer_flow_profiles(config_path)
        return profiles[0].name if profiles else None
    except Exception:
        return None


def _list_profiles() -> None:
    """Print a Rich table of all profiles."""
    from genai_tk.agents.deer_flow.profile import load_deer_flow_profiles
    from genai_tk.utils.config_mngr import global_config

    config_dir = global_config().get_dir_path("paths.config")
    config_path = str(config_dir / "agents" / "deerflow.yaml")

    try:
        profiles = load_deer_flow_profiles(config_path)
    except Exception as e:
        console.print(f"[red]Error loading profiles:[/red] {e}")
        raise typer.Exit(1) from e

    if not profiles:
        console.print(f"[yellow]No profiles found in {config_path}[/yellow]")
        return

    try:
        default_name = global_config().get("deerflow.default_profile")
    except Exception:
        default_name = None

    table = Table(title=f"Deer-flow Profiles  ({config_path})")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Mode", style="magenta")
    table.add_column("Tool Groups", style="green")
    table.add_column("MCP Servers", style="blue")
    table.add_column("LangGraph URL", style="dim")

    for p in profiles:
        name = f"* {p.name}" if default_name and p.name == default_name else p.name
        table.add_row(
            name,
            p.mode or "flash",
            ", ".join(p.tool_groups) or "-",
            ", ".join(p.mcp_servers) or "-",
            p.langgraph_url,
        )

    console.print(table)
    if default_name:
        console.print("[dim]* = default profile (used when -p is not specified)[/dim]")


# ---------------------------------------------------------------------------
# Single-shot run
# ---------------------------------------------------------------------------


async def _run_single_shot(
    profile_name: str,
    user_input: str,
    llm_override: str | None,
    extra_mcp: list[str],
    mode_override: str | None,
    verbose: bool,
    show_trace: bool = False,
    subagent_enabled: bool | None = None,
    plan_mode: bool | None = None,
) -> None:
    """Execute one query and exit.

    Args:
        profile_name: Profile to load.
        user_input: User query text.
        llm_override: LLM identifier override.
        extra_mcp: Additional MCP server names.
        mode_override: Mode override string.
        show_trace: Show node-level trace lines.
        verbose: Enable DEBUG logging.
        subagent_enabled: Override subagent flag (None = use profile setting).
        plan_mode: Override plan_mode flag (None = use profile setting).
    """
    import uuid

    from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

    profile, model_name, config_path = await _prepare_profile(
        profile_name, llm_override, extra_mcp, mode_override, verbose
    )

    llm_display = model_name or "(profile default)"
    tools_display = ", ".join(profile.tool_groups) if profile.tool_groups else "(none)"
    console.print(
        f"[cyan]Profile:[/cyan] {profile.name}  [cyan]Mode:[/cyan] {profile.mode}  [cyan]LLM:[/cyan] {llm_display}"
    )
    console.print(f"[cyan]Tools:[/cyan] {tools_display}  [cyan]Sandbox:[/cyan] {profile.sandbox}")
    if profile.mcp_servers:
        console.print(f"[cyan]MCP:[/cyan] {', '.join(profile.mcp_servers)}")
    console.print()

    with console.status("Loading deer-flow agent...", spinner="dots"):
        client = EmbeddedDeerFlowClient(config_path=config_path, model_name=model_name)
    thread_id = str(uuid.uuid4())

    await _stream_message(
        client=client,
        thread_id=thread_id,
        user_input=user_input,
        model_name=model_name,
        mode=profile.mode,
        show_trace=show_trace,
        subagent_enabled=subagent_enabled if subagent_enabled is not None else profile.subagent_enabled,
        plan_mode=plan_mode if plan_mode is not None else profile.plan_mode,
    )


# ---------------------------------------------------------------------------
# Chat REPL
# ---------------------------------------------------------------------------


async def _run_chat_mode(
    profile_name: str,
    llm_override: str | None,
    extra_mcp: list[str],
    mode_override: str | None,
    initial_input: str | None,
    verbose: bool,
    show_trace: bool = False,
    subagent_enabled: bool | None = None,
    plan_mode: bool | None = None,
) -> None:
    """Interactive multi-turn chat REPL.

    A fresh thread ID is generated at session start; the SqliteSaver
    checkpointer in EmbeddedDeerFlowClient persists multi-turn state for
    the life of the session.  /clear starts a new thread within the same
    session.

    Args:
        profile_name: Profile to load.
        llm_override: LLM identifier override.
        extra_mcp: Additional MCP server names.
        mode_override: Mode override string.
        show_trace: Show node-level trace.
        initial_input: Optional first message before entering the REPL loop.
        verbose: Enable DEBUG logging.
        subagent_enabled: Override subagent flag (None = use profile setting).
        plan_mode: Override plan_mode flag (None = use profile setting).
    """
    import uuid

    from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

    profile, model_name, config_path = await _prepare_profile(
        profile_name, llm_override, extra_mcp, mode_override, verbose
    )

    llm_display = model_name or "(profile default)"
    tools_display = ", ".join(profile.tool_groups) if profile.tool_groups else "(none)"
    console.print(Panel.fit("Deer-flow Interactive Chat", style="bold cyan"))
    console.print(
        f"[cyan]Profile:[/cyan] {profile.name}  [cyan]Mode:[/cyan] {profile.mode}  [cyan]LLM:[/cyan] {llm_display}"
    )
    console.print(f"[cyan]Tools:[/cyan] {tools_display}  [cyan]Sandbox:[/cyan] {profile.sandbox}")
    if profile.mcp_servers:
        console.print(f"[cyan]MCP:[/cyan] {', '.join(profile.mcp_servers)}")
    console.print()
    console.print("[dim]Commands: /help  /info  /mode <flash|thinking|pro|ultra>  /trace  /clear  /quit[/dim]")
    console.print("[dim]Use ↑↓ arrows for history. Ctrl+C or /quit to exit.[/dim]")
    console.print()

    current_mode = profile.mode
    effective_subagent = subagent_enabled if subagent_enabled is not None else profile.subagent_enabled
    effective_plan_mode = plan_mode if plan_mode is not None else profile.plan_mode

    with console.status("Loading deer-flow agent...", spinner="dots"):
        client = EmbeddedDeerFlowClient(config_path=config_path, model_name=model_name)
    thread_id = str(uuid.uuid4())
    session: PromptSession = PromptSession(history=FileHistory(str(Path(".deerflow.input.history"))))
    prompt_style = Style.from_dict({"prompt": "bold green"})

    def _prompt_text() -> str:
        return f"[{current_mode}] >>> "

    if initial_input:
        console.print(Panel(initial_input, title="[bold blue]You[/bold blue]", border_style="blue"))
        await _stream_message(
            client,
            thread_id,
            initial_input,
            model_name,
            current_mode,
            show_trace,
            subagent_enabled=effective_subagent,
            plan_mode=effective_plan_mode,
        )
        console.print()

    while True:
        try:
            with patch_stdout():
                user_input = await session.prompt_async(
                    _prompt_text(), style=prompt_style, auto_suggest=AutoSuggestFromHistory()
                )
            user_input = user_input.strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "/q"):
            console.print("[bold yellow]Goodbye![/bold yellow]")
            break
        elif cmd == "/clear":
            import uuid

            thread_id = str(uuid.uuid4())
            console.print("[yellow]New conversation thread started.[/yellow]")
            continue
        elif cmd == "/help":
            console.print(
                Panel(
                    "/help                       show this help\n"
                    "/info                       show current agent configuration\n"
                    "/mode flash|thinking|pro|ultra  switch reasoning mode (no restart)\n"
                    "/trace                      toggle node-level execution trace\n"
                    "/clear                      start a fresh conversation thread\n"
                    "/quit  /exit  /q            exit",
                    title="[cyan]Chat Commands[/cyan]",
                    border_style="cyan",
                )
            )
            continue
        elif cmd == "/info":
            skills_display = ", ".join(profile.skills) if profile.skills else "(all enabled)"
            console.print(
                Panel(
                    f"Profile   : {profile.name}\n"
                    f"Mode      : {current_mode}\n"
                    f"LLM       : {llm_display}\n"
                    f"Tools     : {tools_display}\n"
                    f"Sandbox   : {profile.sandbox}\n"
                    f"Skills    : {skills_display}\n"
                    f"MCP       : {', '.join(profile.mcp_servers) or 'none'}\n"
                    f"Thread    : {thread_id}\n"
                    f"Subagent  : {effective_subagent}\n"
                    f"Plan mode : {effective_plan_mode}\n"
                    f"Trace     : {'ON' if show_trace else 'OFF'}",
                    title="[bold cyan]Agent Configuration[/bold cyan]",
                    border_style="cyan",
                )
            )
            continue
        elif cmd.startswith("/mode"):
            parts = user_input.split(None, 1)
            if len(parts) < 2 or not parts[1].strip():
                console.print(f"[yellow]Current mode: {current_mode}[/yellow]  (flash | thinking | pro | ultra)")
            else:
                new_mode = parts[1].strip().lower()
                if new_mode not in ("flash", "thinking", "pro", "ultra"):
                    console.print(f"[red]Unknown mode:[/red] {new_mode}. Choose: flash | thinking | pro | ultra")
                else:
                    current_mode = new_mode
                    console.print(f"[green]Mode switched to:[/green] {current_mode}")
            continue
        elif cmd == "/trace":
            show_trace = not show_trace
            console.print(f"[yellow]Node trace: {'ON' if show_trace else 'OFF'}[/yellow]")
            continue
        elif user_input.startswith("/"):
            console.print(f"[yellow]Unknown command: {user_input!r}[/yellow]  Type /help for available commands.")
            continue

        try:
            console.print(Panel(user_input, title="[bold blue]You[/bold blue]", border_style="blue"))
            await _stream_message(
                client,
                thread_id,
                user_input,
                model_name,
                current_mode,
                show_trace,
                subagent_enabled=effective_subagent,
                plan_mode=effective_plan_mode,
            )
            console.print()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Use /quit to exit.[/yellow]")
        except Exception as e:
            console.print(Panel(f"[red]Error: {e}[/red]", title="[bold red]Error[/bold red]", border_style="red"))
            logger.exception("Agent execution error")


# ---------------------------------------------------------------------------
# Web client launcher
# ---------------------------------------------------------------------------


async def _open_web_client(
    profile_name: str,
    llm_override: str | None,
    extra_mcp: list[str],
    mode_override: str | None,
    verbose: bool,
    web_port: int = 3000,
    start_timeout_seconds: float = 180.0,
) -> None:
    """Start the deer-flow backend then launch the Next.js web client.

    Ensures the backend server is running (auto-starting it if needed), sets
    ``NEXT_PUBLIC_BACKEND_BASE_URL`` / ``NEXT_PUBLIC_LANGGRAPH_BASE_URL`` from the
    profile, and starts ``node_modules/.bin/next dev`` inside
    ``DEER_FLOW_PATH/frontend``.  The binary is invoked directly (instead of via
    ``pnpm dev``) to avoid pnpm's self-version-management mechanism, which tries to
    download an exact pnpm release into the (possibly root-owned) store and causes
    EACCES errors.  Blocks until the subprocess exits or the user presses Ctrl+C.

    Args:
        profile_name: Profile to load from ``deerflow.yaml``.
        llm_override: LLM identifier override (e.g. ``gpt_41mini@openai``).
        extra_mcp: Additional MCP server names to enable.
        mode_override: Mode override string (``flash|thinking|pro|ultra``).
        verbose: Enable DEBUG-level logging.
        web_port: Port for the Next.js dev server (default 3000).
        start_timeout_seconds: Seconds to wait for the dev server to begin listening.

    Example:
        ```python
        asyncio.run(_open_web_client("Research Assistant", None, [], None, False))
        ```
    """
    import asyncio as _aio
    import webbrowser

    profile, model_name, _config_path = await _prepare_profile(
        profile_name, llm_override, extra_mcp, mode_override, verbose, start_servers=True
    )

    # Locate frontend directory
    df_path = profile.deer_flow_path or os.environ.get("DEER_FLOW_PATH", "")
    if not df_path:
        console.print(
            "[red]Cannot start web client:[/red] DEER_FLOW_PATH is not set. Point it to your deer-flow clone directory."
        )
        raise typer.Exit(1)

    frontend_dir = Path(df_path) / "frontend"
    if not frontend_dir.exists():
        console.print(f"[red]Frontend directory not found:[/red] {frontend_dir}")
        raise typer.Exit(1)

    web_url = f"http://localhost:{web_port}"

    # Pass backend URLs via environment variables consumed by Next.js.
    env = os.environ.copy()
    env["NEXT_PUBLIC_BACKEND_BASE_URL"] = profile.gateway_url
    env["NEXT_PUBLIC_LANGGRAPH_BASE_URL"] = profile.langgraph_url

    logs_dir = Path(df_path) / "logs"
    logs_dir.mkdir(exist_ok=True)
    frontend_log = logs_dir / "frontend.log"

    console.print(
        f"[cyan]Profile:[/cyan] {profile.name}  [cyan]Mode:[/cyan] {profile.mode}  [cyan]Sandbox:[/cyan] {profile.sandbox}"
    )
    console.print(f"[cyan]Backend:[/cyan] {profile.gateway_url}  [cyan]LangGraph:[/cyan] {profile.langgraph_url}")
    console.print(f"[cyan]Frontend log:[/cyan] {frontend_log}")
    console.print(f"[cyan]Starting deer-flow web client …[/cyan] {web_url}")

    # Build the command to start the Next.js dev server.
    # We invoke node_modules/.bin/next directly instead of 'pnpm dev' to bypass
    # pnpm's "manage-package-manager-versions" mechanism, which unconditionally
    # tries to self-install pnpm@<version> into the (potentially root-owned) store
    # and hangs or fails with EACCES.  The dev script in package.json is parsed so
    # the command stays in sync with upstream changes.
    next_bin = frontend_dir / "node_modules" / ".bin" / "next"
    if not next_bin.exists():
        console.print(f"[red]node_modules not found.[/red] Run [bold]pnpm install[/bold] inside {frontend_dir} first.")
        raise typer.Exit(1)
    import json as _json

    pkg_json = frontend_dir / "package.json"
    dev_script_raw: str = "next dev"
    try:
        dev_script_raw = _json.loads(pkg_json.read_text()).get("scripts", {}).get("dev", dev_script_raw)
    except Exception:
        pass
    # Replace the bare 'next' with the absolute path to the local binary so we
    # never rely on whatever 'next' is on PATH.
    dev_args = dev_script_raw.split()
    if dev_args and dev_args[0] == "next":
        dev_args[0] = str(next_bin)
    dev_cmd = [*dev_args, "--port", str(web_port)]

    log_fp = open(frontend_log, "w", encoding="utf-8")  # noqa: WPS515
    try:
        proc = await _aio.create_subprocess_exec(
            *dev_cmd,
            cwd=str(frontend_dir),
            env=env,
            stdout=_aio.subprocess.PIPE,
            stderr=_aio.subprocess.STDOUT,
        )
    except Exception:
        log_fp.close()
        raise

    async def _stream_frontend_output() -> None:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode(errors="replace").rstrip("\n")
            log_fp.write(text + "\n")
            log_fp.flush()
            console.print(text, markup=False)

    output_task = _aio.create_task(_stream_frontend_output())

    # Poll until the port is accepting connections (or the process dies).
    opened = False
    deadline = _aio.get_event_loop().time() + start_timeout_seconds
    while _aio.get_event_loop().time() < deadline:
        if proc.returncode is not None:
            await output_task
            console.print(
                f"[red]next dev exited early (code {proc.returncode}). Web client did not start.[/red] "
                f"See: {frontend_log}"
            )
            raise typer.Exit(1)

        try:
            _, writer = await _aio.wait_for(_aio.open_connection("localhost", web_port), timeout=1.0)
            writer.close()
            await writer.wait_closed()
            opened = True
            break
        except (ConnectionRefusedError, _aio.TimeoutError, OSError):
            await _aio.sleep(1)

    if opened:
        webbrowser.open(web_url)
        console.print(f"[green]Opened {web_url} in browser.[/green] Press [bold]Ctrl+C[/bold] to stop.")
    else:
        console.print(
            f"[yellow]Timed out waiting for {web_url} after {start_timeout_seconds:.0f}s.[/yellow] "
            f"The dev server may still be starting. See: {frontend_log}"
        )
        webbrowser.open(web_url)

    try:
        await proc.wait()
        await output_task
    except (KeyboardInterrupt, _aio.CancelledError):
        proc.terminate()
        try:
            await _aio.wait_for(proc.wait(), timeout=5)
        except _aio.TimeoutError:
            proc.kill()
        try:
            await _aio.wait_for(output_task, timeout=2)
        except Exception:
            output_task.cancel()
        console.print("\n[yellow]Web client stopped.[/yellow]")
    finally:
        log_fp.close()


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


class DeerFlowCommands(CliTopCommand, BaseModel):
    """CLI commands for Deer-flow agents (embedded client mode)."""

    description: str = "Deer-flow agent commands for interactive AI with advanced reasoning"

    def get_description(self) -> tuple[str, str]:
        """Return command name and description."""
        return "deerflow", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        """Register deerflow subcommands."""

        @cli_app.callback(invoke_without_command=True)
        def main(  # noqa: C901
            ctx: typer.Context,
            input_text: Annotated[
                Optional[str],
                typer.Argument(help="Query text. Omit to use --chat or stdin."),
            ] = None,
            profile: Annotated[
                Optional[str],
                typer.Option("--profile", "-p", help="Profile name from deerflow.yaml"),
            ] = None,
            chat: Annotated[
                bool,
                typer.Option("--chat", "-s", help="Interactive multi-turn chat REPL."),
            ] = False,
            llm: Annotated[
                Optional[str],
                typer.Option("--llm", "-m", help="LLM override (e.g. gpt_41mini@openai)"),
            ] = None,
            mcp: Annotated[
                list[str],
                typer.Option("--mcp", help="Extra MCP server names (repeatable)"),
            ] = [],
            mode: Annotated[
                Optional[str],
                typer.Option("--mode", help="Mode override: flash|thinking|pro|ultra"),
            ] = None,
            trace: Annotated[
                bool,
                typer.Option("--trace", help="Show node-level execution trace."),
            ] = False,
            list_profiles: Annotated[
                bool,
                typer.Option("--list", help="List profiles and exit."),
            ] = False,
            verbose: Annotated[
                bool,
                typer.Option("--verbose", "-v", help="Enable DEBUG logging."),
            ] = False,
            web: Annotated[
                bool,
                typer.Option(
                    "--web",
                    help=(
                        "Start the native deer-flow Next.js web client. "
                        "Launches DEER_FLOW_PATH/frontend/node_modules/.bin/next dev with the "
                        "profile's backend URLs and opens the browser. Requires DEER_FLOW_PATH."
                    ),
                ),
            ] = False,
            subagent: Annotated[
                bool,
                typer.Option("--subagent", "-A", help="Enable subagent delegation. Overrides profile setting."),
            ] = False,
            plan_mode: Annotated[
                bool,
                typer.Option("--plan-mode", "-P", help="Enable TodoList planning mode. Overrides profile setting."),
            ] = False,
        ) -> None:
            """Run Deer-flow agents in-process (embedded mode).

            DeerFlow is loaded directly via DEER_FLOW_PATH/backend. No server
            processes are required for terminal usage. The --web option still
            starts the backend servers for the Next.js frontend.

            Examples:
                cli deerflow --list
                cli deerflow -p "Research Assistant" "Explain quantum computing"
                cli deerflow -p "Coder" --chat
                cli deerflow -p "Research Assistant" --trace "Analyse AI trends"
                cli deerflow -p "Coder" --llm gpt_41mini@openai --mode pro "Refactor my code"
                cli deerflow -p "Research Assistant" --subagent --plan-mode "Study AI trends"
                echo "What is RAG?" | cli deerflow -p "Research Assistant"
                cli deerflow -p "Research Assistant" --web
            """
            if list_profiles:
                _list_profiles()
                return

            if not profile:
                try:
                    from genai_tk.utils.config_mngr import global_config

                    profile = global_config().get("deerflow.default_profile")
                except Exception:
                    profile = None

                if profile:
                    console.print(f"[dim]Using default profile: {profile}[/dim]")
                else:
                    console.print("[red]--profile/-p is required (or --list to see options)[/red]")
                    raise typer.Exit(1)

            if web:
                try:
                    asyncio.run(
                        _open_web_client(
                            profile_name=profile,
                            llm_override=llm,
                            extra_mcp=list(mcp),
                            mode_override=mode,
                            verbose=verbose,
                        )
                    )
                except KeyboardInterrupt:
                    console.print("\n[yellow]Interrupted[/yellow]")
                    raise typer.Exit(0) from None
                except typer.Exit:
                    raise
                except Exception as e:
                    console.print(f"\n[red]Error:[/red] {e}")
                    logger.exception("Deer-flow web client error")
                    raise typer.Exit(1) from e
                return

            if not input_text and not chat:
                if not sys.stdin.isatty():
                    input_text = sys.stdin.read().strip()
                    if not input_text:
                        console.print("[red]No input provided[/red]")
                        raise typer.Exit(1)
                else:
                    console.print("[yellow]No input. Add text, use stdin, or --chat.[/yellow]")
                    raise typer.Exit(1)

            try:
                if chat:
                    asyncio.run(
                        _run_chat_mode(
                            profile_name=profile,
                            llm_override=llm,
                            extra_mcp=list(mcp),
                            mode_override=mode,
                            show_trace=trace,
                            initial_input=input_text,
                            verbose=verbose,
                            subagent_enabled=subagent or None,
                            plan_mode=plan_mode or None,
                        )
                    )
                else:
                    asyncio.run(
                        _run_single_shot(
                            profile_name=profile,
                            user_input=input_text,
                            llm_override=llm,
                            extra_mcp=list(mcp),
                            mode_override=mode,
                            show_trace=trace,
                            verbose=verbose,
                            subagent_enabled=subagent or None,
                            plan_mode=plan_mode or None,
                        )
                    )
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")
                raise typer.Exit(0) from None
            except typer.Exit:
                raise
            except Exception as e:
                console.print(f"\n[red]Error:[/red] {e}")
                logger.exception("Deer-flow CLI error")
                raise typer.Exit(1) from e
