"""CLI commands for Deer-flow agents (embedded client mode).

Loads DeerFlow in-process via DEER_FLOW_PATH/backend — no HTTP servers needed.

Usage examples:
    cli deerflow --list
    cli deerflow "Explain quantum computing"
    cli deerflow -p "Coder" --chat
    cli deerflow -p "Research Assistant" --mode ultra --trace "Analyse AI trends"
    cli deerflow -p "Web Browser" --llm gpt_41mini@openai "Go to atos.net"
    cli deerflow -p "Research Assistant" --subagent --plan-mode "Build a report"
    cli deerflow -p "Research Assistant" --generate-config
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional, cast

import typer
from loguru import logger
from pydantic import BaseModel
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from genai_tk.cli.base import CliTopCommand

if TYPE_CHECKING:
    from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient
    from genai_tk.agents.deer_flow.profile import DeerFlowProfile, DeerFlowSandbox

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
    if llm_id is None:
        console.print("[red]Could not resolve model identifier[/red]")
        raise typer.Exit(1)
    return llm_id


# ---------------------------------------------------------------------------
# Server boot helper (sync wrapper used once per run)
# ---------------------------------------------------------------------------


def _validate_and_normalize_sandbox(sandbox: str) -> DeerFlowSandbox:
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
    return cast("DeerFlowSandbox", normalized)


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


def _check_agent_sandbox_importable() -> bool:
    """Return True if the ``agent-sandbox`` package is importable."""
    try:
        import agent_sandbox  # noqa: F401

        return True
    except ImportError:
        return False


def _validate_docker_sandbox() -> None:
    """Raise :class:`DockerSandboxError` if Docker sandbox prerequisites are unmet.

    Checks two things:
    1. The ``docker`` CLI is present and the daemon is reachable.
    2. The ``agent-sandbox`` Python package is importable (needed by DeerFlow's
       ``AioSandbox`` to talk to the container's HTTP API).
    """
    from genai_tk.agents.deer_flow.profile import DockerSandboxError

    reasons: list[str] = []
    if not _check_docker_available():
        reasons.append("Docker is not available (docker CLI not found or daemon not running)")
    if not _check_agent_sandbox_importable():
        reasons.append("'agent-sandbox' package is not installed — install with: uv add agent-sandbox")
    if reasons:
        raise DockerSandboxError(reasons)


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
    sandbox_override: str | None = None,
) -> tuple[DeerFlowProfile, str | None, Path]:
    """Load, validate and prepare a profile, then write the deer-flow config.

    Writes ``config.yaml`` + ``extensions_config.json`` ready for use by
    the embedded client.

    Args:
        profile_name: Profile name from deerflow.yaml.
        llm_override: LLM identifier override (ID or tag).
        extra_mcp: Additional MCP server names.
        mode_override: Mode override string.
        verbose: Enable DEBUG-level logging.
        sandbox_override: Override sandbox type (None = use profile setting).

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

    os.environ["LANGSMITH_PROJECT"] = f"DeerFlow-tk-{profile.name}"

    try:
        if mode_override:
            profile.mode = validate_mode(mode_override)
        if extra_mcp:
            validated = validate_mcp_servers(extra_mcp)
            profile.mcp_servers = list(set(profile.mcp_servers + validated))
        if sandbox_override:
            profile.sandbox = sandbox_override
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

    if profile.sandbox == "docker":
        try:
            _validate_docker_sandbox()
        except DeerFlowError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1) from e

    return profile, model_name, config_path


# ---------------------------------------------------------------------------
# Apply skills from profile
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LangSmith trace URL helper
# ---------------------------------------------------------------------------


def _try_get_last_trace_url(project_name: str, run_start: float) -> str | None:
    """Return the LangSmith URL for the most recently completed root run.

    Builds the project URL with ``?peek=<run_id>&peeked_trace=<run_id>`` so that
    the link opens the trace in the side-panel — the same behaviour as clicking a
    run row in the LangSmith UI.

    Returns ``None`` silently when tracing is disabled, auth fails, or no run
    was flushed yet.

    Args:
        project_name: LangSmith project name (``LANGSMITH_PROJECT`` value).
        run_start: Epoch time just before the run started — used to filter runs
            so we only match runs that began after this point.
    """
    try:
        from datetime import datetime, timezone

        from langsmith.utils import get_api_url, get_host_url, tracing_is_enabled

        if not tracing_is_enabled() or not project_name:
            return None

        import langsmith as _ls

        _ls_client = _ls.Client()
        _start_dt = datetime.fromtimestamp(run_start, tz=timezone.utc)
        runs = list(
            _ls_client.list_runs(
                project_name=project_name,
                is_root=True,
                start_time=_start_dt,
                limit=1,
                select=["id", "session_id", "name", "status", "start_time"],
            )
        )
        if runs:
            run_id = str(runs[0].id)
            proj_obj = _ls_client.read_project(project_name=project_name)
            tid = _ls_client._get_optional_tenant_id()
            host = get_host_url(None, get_api_url(None))
            project_base = f"{host}/o/{tid}/projects/p/{proj_obj.id}"
            return f"{project_base}?peek={run_id}&peeked_trace={run_id}"
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Core streaming function (single turn)
# ---------------------------------------------------------------------------


async def _stream_message(
    client: EmbeddedDeerFlowClient,
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
    from genai_tk.agents.rich_display import ToolStepRenderer, assistant_panel

    full_text = ""
    current_node = ""
    consecutive_sandbox_errors = 0
    _MAX_SANDBOX_ERRORS = 3
    # Mutable container so the ticker coroutine can read the current start time.
    _llm_ctx: list[float | None] = [time.time()]

    # Check if RichToolCallMiddleware is injected — if so, it already renders
    # tool calls/results via AgentMiddleware hooks; we skip event-based display
    # to avoid duplicates.
    _has_rich_middleware = any(type(m).__name__ == "RichToolCallMiddleware" for m in (client._middlewares_kwarg or []))
    tool_renderer = ToolStepRenderer(console) if not _has_rich_middleware else None

    _PANEL_TITLE = "[bold white on royal_blue1] Assistant [/bold white on royal_blue1]"

    def _text_panel(text: str) -> Panel:
        return Panel(Text(text), title=_PANEL_TITLE, border_style="royal_blue1")

    def _md_panel(text: str) -> Panel:
        return Panel(Markdown(text), title=_PANEL_TITLE, border_style="royal_blue1", padding=(1, 2))

    def _log_llm_elapsed() -> None:
        """Log a dim 'Thought for X.Xs' line and clear the timer."""
        start = _llm_ctx[0]
        if start is not None:
            elapsed = time.time() - start
            if elapsed >= 1.0:
                console.log(f"[dim]🤔 Thought for {elapsed:.1f}s[/dim]")
        _llm_ctx[0] = None

    async def _thinking_ticker() -> None:
        """Update spinner text with elapsed seconds while the LLM is working."""
        while True:
            await asyncio.sleep(0.5)
            start = _llm_ctx[0]
            if start is not None:
                elapsed = time.time() - start
                if elapsed >= 2.0:
                    live.update(Spinner("dots", text=f" Thinking... ({elapsed:.0f}s)"))

    with Live(Spinner("dots", text=" Thinking..."), console=console, refresh_per_second=10) as live:
        ticker = asyncio.create_task(_thinking_ticker())
        try:
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
                    if event.tool_name and tool_renderer:
                        _log_llm_elapsed()
                        tool_renderer.log_tool_call(event.tool_name, event.args)
                elif isinstance(event, ToolResultEvent):
                    _llm_ctx[0] = time.time()  # LLM resumes thinking after tool returns
                    if event.tool_name:
                        content = event.content or ""
                        is_error = content.startswith(("Error", "error"))
                        if tool_renderer:
                            tool_renderer.log_tool_result(event.tool_name, content, is_error=is_error)
                        # Detect sandbox failures and abort after repeated errors
                        if is_error and "Sandbox" in content and "failed" in content:
                            consecutive_sandbox_errors += 1
                            if consecutive_sandbox_errors >= _MAX_SANDBOX_ERRORS:
                                console.print(
                                    f"\n[bold red]Aborting:[/bold red] {consecutive_sandbox_errors} consecutive "
                                    "sandbox errors — the Docker sandbox is unavailable. "
                                    "Try: [cyan]docker ps[/cyan] to check containers, or run with [cyan]--sandbox local[/cyan]."
                                )
                                return full_text
                        else:
                            consecutive_sandbox_errors = 0
                    live.update(Spinner("dots", text=" Thinking..."))
                elif isinstance(event, TokenEvent):
                    if not full_text:  # first token — log how long the LLM thought
                        _log_llm_elapsed()
                    # DeerFlow uses stream_mode="values" — each TokenEvent carries
                    # the FULL text of one AIMessage, not individual tokens.
                    # Multiple AIMessages may arrive (intermediate agent +
                    # final reporter); only the *last* one is the real response.
                    full_text = event.data
                    live.update(_text_panel(full_text))
                elif isinstance(event, ErrorEvent):
                    live.update(Text(f"[red]Error: {event.message}[/red]"))
                    console.print(f"[red]Agent error:[/red] {event.message}")
                    return full_text
        finally:
            ticker.cancel()

        # Switch to markdown rendering before Live exits so the final
        # persistent output is nicely formatted (no second print needed).
        if full_text:
            from genai_tk.agents.deer_flow.embedded_client import strip_reasoning_markers

            full_text = strip_reasoning_markers(full_text)
            live.update(assistant_panel(full_text))

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
    table.add_column("Middlewares", style="yellow")

    for p in profiles:
        name = f"* {p.name}" if default_name and p.name == default_name else p.name
        table.add_row(
            name,
            p.mode or "flash",
            ", ".join(p.tool_groups) or "-",
            ", ".join(p.mcp_servers) or "-",
            ", ".join(m.rsplit(".", 1)[-1] for m in p.middlewares) or "-",
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
    sandbox_override: str | None = None,
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
        sandbox_override: Override sandbox type (None = use profile setting).
    """
    from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

    profile, model_name, config_path = await _prepare_profile(
        profile_name, llm_override, extra_mcp, mode_override, verbose, sandbox_override=sandbox_override
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

    if profile.sandbox == "docker":
        _cleanup_stale_sandbox_containers()

    with console.status("Loading deer-flow agent...", spinner="dots"):
        middlewares = _build_cli_middlewares(profile.middlewares)
        available_skills = set(profile.available_skills) if profile.available_skills is not None else None
        client = EmbeddedDeerFlowClient(
            config_path=config_path,
            model_name=model_name,
            middlewares=middlewares,
            available_skills=available_skills,
        )

    # Deterministic thread_id so the Docker sandbox container is reused
    # across single-shot runs (each random UUID would spin up a new container).
    # Clear old checkpointer state so previous (possibly failed) runs don't
    # pollute the LLM's conversation history.
    thread_id = _stable_thread_id()
    client.clear_thread(thread_id)

    # Snapshot existing output files so we only show new ones after the run.
    files_before = client.snapshot_output_files(thread_id)

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

    # Show only files created during this run
    if profile.sandbox == "docker":
        new_files = client.new_output_files(thread_id, files_before)
        _show_output_files(new_files)


def _stable_thread_id() -> str:
    """Return a deterministic thread ID for sandbox container reuse."""
    import hashlib

    return hashlib.sha256(b"genai-tk-deerflow-single").hexdigest()[:16]


def _build_cli_middlewares(profile_middlewares: list[str]) -> list:
    """Instantiate profile middlewares and prepend RichToolCallMiddleware.

    Mirrors the langchain agent default: every CLI run gets Rich tool-call
    tracing automatically, regardless of what the profile config lists.
    """
    from genai_tk.agents.langchain.middleware.rich_middleware import RichToolCallMiddleware
    from genai_tk.utils.import_utils import instantiate_from_qualified_names

    user_mws = instantiate_from_qualified_names(profile_middlewares, logger=logger)
    # Avoid duplicates if the profile already lists RichToolCallMiddleware.
    if not any(isinstance(m, RichToolCallMiddleware) for m in user_mws):
        user_mws.insert(0, RichToolCallMiddleware(console=console))
    return user_mws


def _show_output_files(files: list[Path]) -> None:
    """Print resolved paths for the given output files."""
    if not files:
        return
    console.print()
    console.print("[cyan]Output files:[/cyan]")
    for f in files:
        console.print(f"  {f.resolve()}")


def _cleanup_stale_sandbox_containers() -> None:
    """Stop any leftover deer-flow sandbox containers from previous runs.

    Prevents port exhaustion and stale container accumulation.  Runs
    ``docker stop`` on all containers matching the ``deer-flow-sandbox``
    prefix except the one belonging to the current stable thread_id.
    """
    import hashlib
    import subprocess

    keep_suffix = hashlib.sha256(b"genai-tk-deerflow-single").hexdigest()[:8]
    keep_name = f"deer-flow-sandbox-{keep_suffix}"
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", "name=deer-flow-sandbox"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if not result.stdout.strip():
            return
        # Get container names to decide which to stop
        inspect = subprocess.run(
            ["docker", "ps", "--filter", "name=deer-flow-sandbox", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        stale = [name.strip() for name in inspect.stdout.strip().splitlines() if name.strip() != keep_name]
        if stale:
            logger.debug(f"Stopping {len(stale)} stale sandbox container(s): {stale}")
            for name in stale:
                subprocess.run(["docker", "stop", name], capture_output=True, timeout=15)
    except Exception as e:
        logger.debug(f"Could not clean stale sandbox containers: {e}")


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
    sandbox_override: str | None = None,
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
        sandbox_override: Override sandbox type (None = use profile setting).
    """
    from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

    profile, model_name, config_path = await _prepare_profile(
        profile_name, llm_override, extra_mcp, mode_override, verbose, sandbox_override=sandbox_override
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
    last_trace_url: str | None = None  # URL of the most recently completed LangSmith trace

    if profile.sandbox == "docker":
        _cleanup_stale_sandbox_containers()

    with console.status("Loading deer-flow agent...", spinner="dots"):
        middlewares = _build_cli_middlewares(profile.middlewares)
        available_skills = set(profile.available_skills) if profile.available_skills is not None else None
        client = EmbeddedDeerFlowClient(
            config_path=config_path,
            model_name=model_name,
            middlewares=middlewares,
            available_skills=available_skills,
        )
    thread_id = _stable_thread_id()
    client.clear_thread(thread_id)
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.styles import Style

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
            available_models = [m.get("name", "?") for m in client.list_models()]
            available_skills = [s.get("name", "?") for s in client.list_skills()]
            # Resolve the skills mount path for diagnostics
            try:
                from omegaconf import OmegaConf as _OC

                from genai_tk.utils.config_mngr import get_raw_config, paths_config

                _raw = get_raw_config()
                _dirs = _OC.select(_raw, "deerflow.skills.directories")
                if _dirs:
                    _p = list(_dirs)[0]
                    if "${paths.project}" in _p:
                        _p = _p.replace("${paths.project}", str(paths_config().project))
                    skills_path_display = _p
                else:
                    skills_path_display = "(not configured)"
            except Exception:
                skills_path_display = "(unknown)"
            # Build LangSmith section when tracing is enabled
            langsmith_line = ""
            try:
                from langsmith.utils import get_api_url, get_host_url, get_tracer_project, tracing_is_enabled

                if tracing_is_enabled():
                    import langsmith as _ls

                    _ls_project = get_tracer_project()
                    _ls_host = get_host_url(None, get_api_url(None))
                    try:
                        _ls_client = _ls.Client()
                        _ls_proj_obj = _ls_client.read_project(project_name=_ls_project)
                        _ls_tid = _ls_client._get_optional_tenant_id()
                        if _ls_tid and _ls_proj_obj.id:
                            _ls_project_base = f"{_ls_host}/o/{_ls_tid}/projects/p/{_ls_proj_obj.id}"
                        else:
                            _ls_project_base = f"{_ls_host}/projects"
                        # Last run: build project URL with ?peek=<id>&peeked_trace=<id>
                        # which opens the trace in a side-panel — same as clicking a run
                        # in the LangSmith UI.
                        _ls_run_id: str | None = None
                        try:
                            _ls_runs = list(
                                _ls_client.list_runs(
                                    project_name=_ls_project,
                                    is_root=True,
                                    limit=1,
                                    select=["id", "session_id", "name", "status", "start_time"],
                                )
                            )
                            if _ls_runs:
                                _ls_run_id = str(_ls_runs[0].id)
                        except Exception:
                            pass
                        if _ls_run_id:
                            _ls_url = f"{_ls_project_base}?peek={_ls_run_id}&peeked_trace={_ls_run_id}"
                        else:
                            _ls_url = _ls_project_base
                    except Exception:
                        _ls_project_base = f"{_ls_host}/projects"
                        _ls_url = last_trace_url or _ls_project_base
                    _label = "Last trace" if _ls_run_id else "Project"
                    langsmith_line = (
                        f"LangSmith  : [link={_ls_url}]{_label} ↗[/link]  [dim](project: {_ls_project})[/dim]\n"
                    )
            except Exception:
                pass
            console.print(
                Panel(
                    f"Profile   : {profile.name}\n"
                    f"Mode      : {current_mode}\n"
                    f"LLM       : {llm_display}\n"
                    f"Tools     : {tools_display}\n"
                    f"Sandbox   : {profile.sandbox}\n"
                    f"Skills    : {skills_display}\n"
                    f"Skills dir: {skills_path_display}\n"
                    f"MCP       : {', '.join(profile.mcp_servers) or 'none'}\n"
                    f"Thread    : {thread_id}\n"
                    f"Subagent  : {effective_subagent}\n"
                    f"Plan mode : {effective_plan_mode}\n"
                    f"Trace     : {'ON' if show_trace else 'OFF'}\n"
                    + langsmith_line
                    + f"Models    : {', '.join(available_models) or 'none'}\n"
                    f"Avail skills: {', '.join(available_skills) or 'none'}",
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
            _run_start = time.time()
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
            # Capture the LangSmith trace URL for the run that just completed.
            last_trace_url = _try_get_last_trace_url(os.environ.get("LANGSMITH_PROJECT", ""), _run_start)
            if last_trace_url:
                console.print(f"[dim]🔗 [link={last_trace_url}]Open trace ↗[/link][/dim]")
            console.print()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Use /quit to exit.[/yellow]")
        except Exception as e:
            console.print(Panel(f"[red]Error: {e}[/red]", title="[bold red]Error[/bold red]", border_style="red"))
            logger.exception("Agent execution error")


# ---------------------------------------------------------------------------
# Config generation helper (for --generate-config)
# ---------------------------------------------------------------------------


def _generate_config_and_print_instructions(
    profile_name: str,
    llm_override: str | None,
    extra_mcp: list[str],
    mode_override: str | None,
    verbose: bool,
) -> None:
    """Generate DeerFlow config files and print instructions to launch the native stack.

    Writes ``config.yaml`` and ``extensions_config.json`` to the DeerFlow backend
    directory (or a temp directory when ``DEER_FLOW_PATH`` is not set), then prints
    step-by-step instructions to start the standard DeerFlow backend and frontend.

    Args:
        profile_name: Profile to load from ``deerflow.yaml``.
        llm_override: LLM identifier override.
        extra_mcp: Additional MCP server names to enable.
        mode_override: Mode override string.
        verbose: Enable DEBUG-level logging.
    """
    import asyncio as _aio

    async def _run() -> None:
        profile, _model_name, config_path = await _prepare_profile(
            profile_name, llm_override, extra_mcp, mode_override, verbose
        )
        ext_config_path = config_path.parent / "extensions_config.json"
        df_path = os.environ.get("DEER_FLOW_PATH", "")

        console.print("\n[green]✓ Config files generated:[/green]")
        console.print(f"  config.yaml          → [cyan]{config_path}[/cyan]")
        console.print(f"  extensions_config.json → [cyan]{ext_config_path}[/cyan]")
        console.print()
        console.print("[bold]To launch the native DeerFlow stack:[/bold]")
        console.print()
        if df_path:
            console.print("[bold cyan]Backend (terminal 1):[/bold cyan]")
            console.print(f"  cd {df_path}/backend")
            console.print("  langgraph dev")
            console.print()
            console.print("[bold cyan]Frontend (terminal 2):[/bold cyan]")
            console.print(f"  cd {df_path}/frontend")
            console.print("  pnpm dev")
            console.print()
            console.print("[dim]Then open http://localhost:3000 in your browser.[/dim]")
        else:
            console.print(
                "[yellow]DEER_FLOW_PATH is not set.[/yellow] "
                "Set it to your deer-flow clone, then run:\n"
                "  cd $DEER_FLOW_PATH/backend && langgraph dev\n"
                "  cd $DEER_FLOW_PATH/frontend && pnpm dev"
            )

    _aio.run(_run())


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
                typer.Argument(help="Query text. Omit to use --chat, --input, or stdin."),
            ] = None,
            input: Annotated[
                Optional[str],
                typer.Option("--input", "-i", help="Input query (alternative to positional arg)."),
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
            generate_config: Annotated[
                bool,
                typer.Option(
                    "--generate-config",
                    "-G",
                    help=(
                        "Generate DeerFlow config.yaml and extensions_config.json, then print "
                        "instructions to start the native DeerFlow backend (langgraph dev) and "
                        "frontend (pnpm dev). Use this to connect the standard DeerFlow web UI."
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
            sandbox: Annotated[
                Optional[str],
                typer.Option("--sandbox", "-S", help="Sandbox override: local | docker"),
            ] = None,
        ) -> None:
            """Run Deer-flow agents in-process (embedded mode).

            DeerFlow is loaded directly via DEER_FLOW_PATH/backend. No server processes
            are required for terminal usage.

            Use --generate-config to write the native DeerFlow config files and get
            instructions for launching the standard DeerFlow backend and frontend.

            Examples:
                cli deerflow --list
                cli deerflow -p "Research Assistant" "Explain quantum computing"
                cli deerflow -p "Coder" --chat
                cli deerflow -p "Research Assistant" --trace "Analyse AI trends"
                cli deerflow -p "Coder" --llm gpt_41mini@openai --mode pro "Refactor my code"
                cli deerflow -p "Research Assistant" --subagent --plan-mode "Study AI trends"
                echo "What is RAG?" | cli deerflow -p "Research Assistant"
                cli deerflow -p "Research Assistant" --generate-config
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

            if generate_config:
                try:
                    _generate_config_and_print_instructions(
                        profile_name=profile,
                        llm_override=llm,
                        extra_mcp=list(mcp),
                        mode_override=mode,
                        verbose=verbose,
                    )
                except typer.Exit:
                    raise
                except Exception as e:
                    console.print(f"\n[red]Error:[/red] {e}")
                    logger.exception("Deer-flow config generation error")
                    raise typer.Exit(1) from e
                return

            # --input/-i option is an alias for the positional input_text argument
            input_text = input_text or input
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
                user_input_text = input_text or ""
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
                            sandbox_override=sandbox or None,
                        )
                    )
                else:
                    if not user_input_text:
                        console.print("[red]No input provided[/red]")
                        raise typer.Exit(1)
                    asyncio.run(
                        _run_single_shot(
                            profile_name=profile,
                            user_input=user_input_text,
                            llm_override=llm,
                            extra_mcp=list(mcp),
                            mode_override=mode,
                            show_trace=trace,
                            verbose=verbose,
                            subagent_enabled=subagent or None,
                            plan_mode=plan_mode or None,
                            sandbox_override=sandbox or None,
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
