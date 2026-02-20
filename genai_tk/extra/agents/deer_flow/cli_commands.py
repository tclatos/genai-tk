"""CLI commands for Deer-flow agents (HTTP client mode).

Connects to a running Deer-flow server via its HTTP API
instead of importing the library in-process.

The server is auto-started if it is not already running (requires DEER_FLOW_PATH).

Usage examples:
    cli deerflow --list
    cli deerflow "Explain quantum computing"
    cli deerflow -p "Coder" --chat
    cli deerflow -p "Research Assistant" --mode ultra --trace "Analyse AI trends"
    cli deerflow -p "Web Browser" --llm gpt_41mini@openai "Go to atos.net"
"""

from __future__ import annotations

import asyncio
import os
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
    "planner": "Planning",
    "reporter": "Writing report",
    "researcher": "Researching",
    "coder": "Writing code",
    "model": "Thinking",
    "tools": "Using tools",
    "search_tool": "Searching",
    "web_search": "Searching the web",
    "tavily_search": "Searching (Tavily)",
    "python_repl": "Running code",
}


# ---------------------------------------------------------------------------
# Mode -> configurable flags
# ---------------------------------------------------------------------------


def _mode_to_configurable(mode: str) -> dict:
    """Map a mode name to deer-flow configurable flags.

    Args:
        mode: One of flash, thinking, pro, ultra.

    Returns:
        Dict with thinking_enabled, is_plan_mode, subagent_enabled.
    """
    m = mode.lower()
    if m == "flash":
        return {"thinking_enabled": False, "is_plan_mode": False, "subagent_enabled": False}
    if m == "thinking":
        return {"thinking_enabled": True, "is_plan_mode": False, "subagent_enabled": False}
    if m == "pro":
        return {"thinking_enabled": True, "is_plan_mode": True, "subagent_enabled": False}
    if m == "ultra":
        return {"thinking_enabled": True, "is_plan_mode": True, "subagent_enabled": True}
    return {"thinking_enabled": False, "is_plan_mode": False, "subagent_enabled": False}


# ---------------------------------------------------------------------------
# LLM identifier -> deer-flow model_name
# ---------------------------------------------------------------------------


def _resolve_model_name(llm_identifier: str) -> str:
    """Resolve a GenAI Toolkit LLM identifier to a deer-flow model name.

    Deer-flow model names are the GenAI Toolkit model IDs
    (set during config generation in config_bridge.py).  Tags like
    fast_model are resolved to their ID first.

    Args:
        llm_identifier: LLM ID or tag, e.g. gpt_41mini@openai or fast_model.

    Returns:
        Deer-flow model name string.
    """
    from genai_tk.core.llm_factory import LlmFactory

    llm_id, error_msg = LlmFactory.resolve_llm_identifier_safe(llm_identifier)
    if error_msg:
        console.print(error_msg)
        raise typer.Exit(1)
    return llm_id


# ---------------------------------------------------------------------------
# Server boot helper (sync wrapper used once per run)
# ---------------------------------------------------------------------------


async def _ensure_server(
    profile_auto_start: bool,
    deer_flow_path: str | None,
    langgraph_url: str,
    gateway_url: str,
) -> None:
    """Check server health and auto-start if needed.

    Args:
        profile_auto_start: Whether the profile allows auto-starting the server.
        deer_flow_path: Override path to deer-flow clone (falls back to DEER_FLOW_PATH env).
        langgraph_url: LangGraph server base URL.
        gateway_url: Gateway API base URL.
    """
    from genai_tk.extra.agents.deer_flow.client import DeerFlowClient
    from genai_tk.extra.agents.deer_flow.server_manager import DeerFlowServerManager

    client = DeerFlowClient(langgraph_url=langgraph_url, gateway_url=gateway_url)
    is_up = await client.health_check()

    if is_up:
        logger.debug("Deer-flow server already running")
        return

    if not profile_auto_start:
        console.print(
            "[red]Deer-flow server is not running.[/red] Start it manually or set auto_start: true in the profile."
        )
        raise typer.Exit(1)

    df_path = deer_flow_path or os.environ.get("DEER_FLOW_PATH", "")
    if not df_path:
        console.print(
            "[red]Cannot auto-start:[/red] DEER_FLOW_PATH is not set. Point it to your deer-flow clone directory."
        )
        raise typer.Exit(1)

    with console.status("Starting Deer-flow servers...", spinner="dots"):
        mgr = DeerFlowServerManager(
            deer_flow_path=df_path,
            langgraph_url=langgraph_url,
            gateway_url=gateway_url,
        )
        try:
            await mgr.start()
            console.print("[green]Deer-flow servers started[/green]")
        except Exception as e:
            console.print(f"[red]Failed to start Deer-flow servers:[/red] {e}")
            raise typer.Exit(1) from e


# ---------------------------------------------------------------------------
# Apply skills from profile
# ---------------------------------------------------------------------------


async def _apply_profile_skills(
    profile_skills: list[str],
    skill_directories: list[str],
    gateway_url: str,
) -> None:
    """Activate skills on the running server according to a profile.

    Args:
        profile_skills: Explicit skill list from profile (category/name format).
        skill_directories: Directories to auto-discover skills from.
        gateway_url: Gateway API base URL.
    """
    from genai_tk.extra.agents.deer_flow.client import DeerFlowClient
    from genai_tk.extra.agents.deer_flow.config_bridge import load_skills_from_directories

    skills: list[str] = list(profile_skills)
    if skill_directories:
        skills = load_skills_from_directories(skill_directories)
    if not skills:
        return

    client = DeerFlowClient(gateway_url=gateway_url)

    # Fetch the set of skill names the server actually knows about, so we
    # can silently skip any local skills that don't exist on this server
    # version (e.g. "vercel-deploy-claimable" vs "vercel-deploy").
    try:
        server_skills = {s["name"] for s in await client.list_skills()}
    except Exception:
        server_skills = None  # can't fetch — try enabling anyway

    for skill in skills:
        # API expects bare name (e.g. "consulting-analysis"), not "category/name"
        skill_name = skill.split("/")[-1]
        if server_skills is not None and skill_name not in server_skills:
            logger.debug(f"Skill '{skill_name}' not available on this server — skipping")
            continue
        try:
            await client.set_skill(skill_name, enabled=True)
            logger.debug(f"Enabled skill: {skill_name}")
        except Exception as e:
            logger.warning(f"Could not enable skill '{skill_name}': {e}")


# ---------------------------------------------------------------------------
# Core streaming function (single turn)
# ---------------------------------------------------------------------------


async def _stream_message(
    langgraph_url: str,
    gateway_url: str,
    thread_id: str,
    user_input: str,
    model_name: str | None,
    mode: str,
    show_trace: bool,
) -> str:
    """Send a message and stream the response to the terminal.

    Tokens are printed as they arrive via Rich Live.
    If show_trace is True, node names are printed between token sections.

    Args:
        langgraph_url: LangGraph server URL.
        gateway_url: Gateway API URL.
        thread_id: Conversation thread ID.
        user_input: User message.
        model_name: Deer-flow model name override (None = server default).
        mode: Agent mode string.
        show_trace: Show node-level execution trace.

    Returns:
        Accumulated full response text.
    """
    from genai_tk.extra.agents.deer_flow.client import (
        DeerFlowClient,
        ErrorEvent,
        NodeEvent,
        TokenEvent,
        ToolCallEvent,
        ToolResultEvent,
    )

    flags = _mode_to_configurable(mode)
    client = DeerFlowClient(langgraph_url=langgraph_url, gateway_url=gateway_url)

    full_text = ""
    current_node = ""

    _PANEL_TITLE = "[bold white on royal_blue1] Assistant [/bold white on royal_blue1]"

    def _text_panel(text: str) -> Panel:
        return Panel(Text(text), title=_PANEL_TITLE, border_style="royal_blue1")

    def _md_panel(text: str) -> Panel:
        return Panel(Markdown(text), title=_PANEL_TITLE, border_style="royal_blue1", padding=(1, 2))

    with Live(Spinner("dots", text=" Thinking..."), console=console, refresh_per_second=10) as live:
        async for event in client.stream_run(
            thread_id=thread_id,
            user_input=user_input,
            model_name=model_name,
            thinking_enabled=flags["thinking_enabled"],
            is_plan_mode=flags["is_plan_mode"],
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
                args_preview = str(event.args)[:120].replace("\n", " ") if event.args else ""
                console.log(
                    f"[dim cyan]⚙ tool call:[/dim cyan] [cyan]{event.tool_name}[/cyan] [dim]{args_preview}[/dim]"
                )
            elif isinstance(event, ToolResultEvent):
                if show_trace:
                    result_preview = event.content[:200].replace("\n", " ") if event.content else ""
                    console.log(
                        f"[dim green]✓ tool result:[/dim green] [dim]{event.tool_name}[/dim] [dim italic]{result_preview}[/dim italic]"
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
        from genai_tk.extra.agents.deer_flow.profile import load_deer_flow_profiles
        from genai_tk.utils.config_mngr import global_config

        config_dir = global_config().get_dir_path("paths.config")
        config_path = str(config_dir / "agents" / "deerflow.yaml")
        profiles = load_deer_flow_profiles(config_path)
        return profiles[0].name if profiles else None
    except Exception:
        return None


def _list_profiles() -> None:
    """Print a Rich table of all profiles."""
    from genai_tk.extra.agents.deer_flow.profile import load_deer_flow_profiles
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
    show_trace: bool = True,
    stream_enabled: bool = True,  # always streams; kept for call-site compat
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
    """
    from genai_tk.extra.agents.deer_flow.client import DeerFlowClient
    from genai_tk.extra.agents.deer_flow.config_bridge import setup_deer_flow_config
    from genai_tk.extra.agents.deer_flow.profile import (
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
    except DeerFlowError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    model_name: str | None = None
    if llm_override:
        model_name = _resolve_model_name(llm_override)
    elif profile.llm:
        model_name = _resolve_model_name(profile.llm)

    with console.status("Preparing Deer-flow config...", spinner="dots"):
        setup_deer_flow_config(
            mcp_server_names=profile.mcp_servers,
            skill_directories=profile.skill_directories,
            sandbox=profile.sandbox,
        )

    await _ensure_server(profile.auto_start, profile.deer_flow_path, profile.langgraph_url, profile.gateway_url)
    await _apply_profile_skills(profile.skills, profile.skill_directories, profile.gateway_url)

    llm_display = model_name or "(server default)"
    tools_display = ", ".join(profile.tool_groups) if profile.tool_groups else "(none)"
    console.print(
        f"[cyan]Profile:[/cyan] {profile.name}  [cyan]Mode:[/cyan] {profile.mode}  [cyan]LLM:[/cyan] {llm_display}"
    )
    console.print(f"[cyan]Tools:[/cyan] {tools_display}  [cyan]Sandbox:[/cyan] {profile.sandbox}")
    if profile.mcp_servers:
        console.print(f"[cyan]MCP:[/cyan] {', '.join(profile.mcp_servers)}")
    console.print()

    client = DeerFlowClient(langgraph_url=profile.langgraph_url, gateway_url=profile.gateway_url)
    thread_id = await client.create_thread()

    await _stream_message(
        langgraph_url=profile.langgraph_url,
        gateway_url=profile.gateway_url,
        thread_id=thread_id,
        user_input=user_input,
        model_name=model_name,
        mode=profile.mode,
        show_trace=show_trace,
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
    show_trace: bool = True,
    stream_enabled: bool = True,  # always streams; kept for call-site compat
) -> None:
    """Interactive multi-turn chat REPL.

    Args:
        profile_name: Profile to load.
        llm_override: LLM identifier override.
        extra_mcp: Additional MCP server names.
        mode_override: Mode override string.
        show_trace: Show node-level trace.
        initial_input: Optional first message before entering the REPL loop.
        verbose: Enable DEBUG logging.
    """
    from genai_tk.extra.agents.deer_flow.client import DeerFlowClient
    from genai_tk.extra.agents.deer_flow.config_bridge import setup_deer_flow_config
    from genai_tk.extra.agents.deer_flow.profile import (
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
    except DeerFlowError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    model_name: str | None = None
    if llm_override:
        model_name = _resolve_model_name(llm_override)
    elif profile.llm:
        model_name = _resolve_model_name(profile.llm)

    with console.status("Preparing Deer-flow config...", spinner="dots"):
        setup_deer_flow_config(
            mcp_server_names=profile.mcp_servers,
            skill_directories=profile.skill_directories,
            sandbox=profile.sandbox,
        )

    await _ensure_server(profile.auto_start, profile.deer_flow_path, profile.langgraph_url, profile.gateway_url)
    await _apply_profile_skills(profile.skills, profile.skill_directories, profile.gateway_url)

    llm_display = model_name or "(server default)"
    tools_display = ", ".join(profile.tool_groups) if profile.tool_groups else "(none)"
    console.print(Panel.fit("Deer-flow Interactive Chat", style="bold cyan"))
    console.print(
        f"[cyan]Profile:[/cyan] {profile.name}  [cyan]Mode:[/cyan] {profile.mode}  [cyan]LLM:[/cyan] {llm_display}"
    )
    console.print(f"[cyan]Tools:[/cyan] {tools_display}  [cyan]Sandbox:[/cyan] {profile.sandbox}")
    if profile.mcp_servers:
        console.print(f"[cyan]MCP:[/cyan] {', '.join(profile.mcp_servers)}")
    console.print()
    console.print("[dim]Commands: /quit /exit /q  /clear  /help  /info  /trace[/dim]")
    console.print("[dim]Use up/down arrows to navigate history[/dim]")
    console.print()

    client = DeerFlowClient(langgraph_url=profile.langgraph_url, gateway_url=profile.gateway_url)
    thread_id = await client.create_thread()
    session: PromptSession = PromptSession(history=FileHistory(str(Path(".deerflow.input.history"))))
    prompt_style = Style.from_dict({"prompt": "bold green"})

    if initial_input:
        console.print(Panel(initial_input, title="[bold blue]You[/bold blue]", border_style="blue"))
        await _stream_message(
            profile.langgraph_url, profile.gateway_url, thread_id, initial_input, model_name, profile.mode, show_trace
        )
        console.print()

    while True:
        try:
            with patch_stdout():
                user_input = await session.prompt_async(
                    ">>> ", style=prompt_style, auto_suggest=AutoSuggestFromHistory()
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
            thread_id = await client.create_thread()
            console.print("[yellow]New conversation thread started[/yellow]")
            continue
        elif cmd == "/help":
            console.print(
                Panel(
                    "/help    show this help\n"
                    "/info    show agent configuration\n"
                    "/clear   start a new conversation thread\n"
                    "/trace   toggle node-level trace\n"
                    "/quit    exit",
                    title="[cyan]Commands[/cyan]",
                    border_style="cyan",
                )
            )
            continue
        elif cmd == "/info":
            skills_display = ", ".join(profile.skills) if profile.skills else "(all enabled)"
            console.print(
                Panel(
                    f"Profile   : {profile.name}\n"
                    f"Mode      : {profile.mode}\n"
                    f"LLM       : {llm_display}\n"
                    f"Tools     : {tools_display}\n"
                    f"Sandbox   : {profile.sandbox}\n"
                    f"Skills    : {skills_display}\n"
                    f"MCP       : {', '.join(profile.mcp_servers) or 'none'}\n"
                    f"Thread    : {thread_id}\n"
                    f"Trace     : {'ON' if show_trace else 'OFF'}\n"
                    f"LangGraph : {profile.langgraph_url}\n"
                    f"Gateway   : {profile.gateway_url}",
                    title="[bold cyan]Agent Configuration[/bold cyan]",
                    border_style="cyan",
                )
            )
            continue
        elif cmd == "/trace":
            show_trace = not show_trace
            console.print(f"[yellow]Node trace: {'ON' if show_trace else 'OFF'}[/yellow]")
            continue
        elif user_input.startswith("/"):
            console.print(f"[yellow]Unknown command: {user_input}[/yellow]")
            continue

        try:
            console.print(Panel(user_input, title="[bold blue]You[/bold blue]", border_style="blue"))
            await _stream_message(
                profile.langgraph_url, profile.gateway_url, thread_id, user_input, model_name, profile.mode, show_trace
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
) -> None:
    """Start the deer-flow backend then launch the Next.js web client.

    Ensures the backend server is running (auto-starting it if needed), sets
    ``NEXT_PUBLIC_BACKEND_BASE_URL`` / ``NEXT_PUBLIC_LANGGRAPH_BASE_URL`` from the
    profile, starts ``pnpm dev`` inside ``DEER_FLOW_PATH/frontend``, and opens the
    browser.  Blocks until the subprocess exits or the user presses Ctrl+C.

    Args:
        profile_name: Profile to load from ``deerflow.yaml``.
        llm_override: LLM identifier override (e.g. ``gpt_41mini@openai``).
        extra_mcp: Additional MCP server names to enable.
        mode_override: Mode override string (``flash|thinking|pro|ultra``).
        verbose: Enable DEBUG-level logging.
        web_port: Port for the Next.js dev server (default 3000).

    Example:
        ```python
        asyncio.run(_open_web_client("Research Assistant", None, [], None, False))
        ```
    """
    import asyncio as _aio
    import webbrowser

    from genai_tk.extra.agents.deer_flow.config_bridge import setup_deer_flow_config
    from genai_tk.extra.agents.deer_flow.profile import (
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
    except DeerFlowError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    with console.status("Preparing Deer-flow config...", spinner="dots"):
        setup_deer_flow_config(
            mcp_server_names=profile.mcp_servers,
            skill_directories=profile.skill_directories,
            sandbox=profile.sandbox,
        )

    await _ensure_server(profile.auto_start, profile.deer_flow_path, profile.langgraph_url, profile.gateway_url)
    await _apply_profile_skills(profile.skills, profile.skill_directories, profile.gateway_url)

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

    # Pass backend URLs via environment variables.
    # Also redirect PNPM_HOME into the frontend directory so pnpm can write its
    # toolchain files without hitting system-wide permission restrictions.
    pnpm_home = str(frontend_dir / ".pnpm-home")
    Path(pnpm_home).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["NEXT_PUBLIC_BACKEND_BASE_URL"] = profile.gateway_url
    env["NEXT_PUBLIC_LANGGRAPH_BASE_URL"] = profile.langgraph_url
    env["PNPM_HOME"] = pnpm_home

    console.print(f"[cyan]Profile:[/cyan] {profile.name}  [cyan]Mode:[/cyan] {profile.mode}")
    console.print(f"[cyan]Backend:[/cyan] {profile.gateway_url}  [cyan]LangGraph:[/cyan] {profile.langgraph_url}")
    console.print(f"[cyan]Starting deer-flow web client …[/cyan] {web_url}")

    proc = await _aio.create_subprocess_exec(
        "pnpm",
        "dev",
        "--port",
        str(web_port),
        cwd=str(frontend_dir),
        env=env,
    )

    # Poll until the port is accepting connections (or the process dies).
    opened = False
    deadline = _aio.get_event_loop().time() + 60
    while _aio.get_event_loop().time() < deadline:
        # Bail immediately if pnpm already exited with an error
        if proc.returncode is not None:
            console.print(f"[red]pnpm dev exited early (code {proc.returncode}). Web client did not start.[/red]")
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
        console.print(f"[yellow]Timed out waiting for {web_url}. The dev server may still be starting.[/yellow]")
        webbrowser.open(web_url)

    try:
        await proc.wait()
    except (KeyboardInterrupt, _aio.CancelledError):
        proc.terminate()
        try:
            await _aio.wait_for(proc.wait(), timeout=5)
        except _aio.TimeoutError:
            proc.kill()
        console.print("\n[yellow]Web client stopped.[/yellow]")


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


class DeerFlowCommands(CliTopCommand, BaseModel):
    """CLI commands for Deer-flow agents (HTTP client mode)."""

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
                        "Launches pnpm dev in DEER_FLOW_PATH/frontend with the profile's backend URLs "
                        "and opens the browser. Requires DEER_FLOW_PATH."
                    ),
                ),
            ] = False,
        ) -> None:
            """Run Deer-flow agents via HTTP API.

            The Deer-flow server is auto-started when not already running
            (requires DEER_FLOW_PATH to point to the deer-flow clone).

            Examples:
                cli deerflow --list
                cli deerflow -p "Research Assistant" "Explain quantum computing"
                cli deerflow -p "Coder" --chat
                cli deerflow -p "Research Assistant" --trace "Analyse AI trends"
                cli deerflow -p "Coder" --llm gpt_41mini@openai --mode pro "Refactor my code"
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
