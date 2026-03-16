"""CLI commands for Deep Agent (deepagents-cli) integration.

Exposes deepagents-cli as ``cli agents deepagent`` command group.
Uses in-process integration: LangChain models are created via genai-tk's
``LlmFactory`` and passed directly to ``deepagents_cli.agent.create_cli_agent()``,
bypassing deepagents-cli's own model-creation pipeline.

Usage examples:
    cli agents deepagent                             # TUI with default settings
    cli agents deepagent --llm fast_model            # TUI with specific model
    cli agents deepagent --profile coder             # TUI with named profile
    cli agents deepagent task "Fix the failing tests" # Non-interactive task
    cli agents deepagent task --profile researcher "Summarise AI trends"
    cli agents deepagent list                        # List configured agents
    cli agents deepagent reset --agent mybot         # Reset agent state
    cli agents deepagent skills list                 # List agent skills
    cli agents deepagent threads list                # List chat threads
"""

from __future__ import annotations

import asyncio
import sys
from typing import Annotated, Optional

import typer
from loguru import logger
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from genai_tk.cli.base import CliTopCommand

console = Console()


# ---------------------------------------------------------------------------
# Profile resolution helpers
# ---------------------------------------------------------------------------


def _load_config():
    """Load deepagent config (lazy import to avoid side effects at import time)."""
    from genai_tk.agents.deepagent_cli.models import load_deepagent_config

    return load_deepagent_config()


def _resolve_profile_settings(
    profile_name: str | None,
    llm_override: str | None,
    auto_approve_override: bool | None,
):
    """Resolve final launch settings by merging profile defaults with CLI overrides.

    Priority: CLI flag > profile field > global config default.

    Args:
        profile_name: Profile name from --profile flag.
        llm_override: LLM identifier from --llm flag.
        auto_approve_override: True/False when flag is explicitly set, else None.

    Returns:
        Tuple of (config, profile_or_none, effective_llm, effective_profile_fields).
    """
    from genai_tk.agents.deepagent_cli.models import DeepagentProfile

    config = _load_config()

    profile: DeepagentProfile | None = None
    if profile_name:
        profile = config.get_profile(profile_name)
        if profile is None:
            names = [p.name for p in config.profiles]
            console.print(
                f"[red]Profile not found:[/red] {profile_name!r}. Available: {', '.join(names) or '(none defined)'}"
            )
            raise typer.Exit(1)

    # Select profile or build a synthetic one from global config defaults
    if profile is None:
        if config.default_profile:
            profile = config.get_profile(config.default_profile)
        if profile is None:
            profile = DeepagentProfile(
                name="__default__",
                llm=config.default_model,
                auto_approve=config.auto_approve,
                enable_memory=config.enable_memory,
                enable_skills=config.enable_skills,
                enable_shell=config.enable_shell,
                shell_allow_list=config.shell_allow_list,
                sandbox=config.sandbox,
                system_prompt=config.system_prompt,
            )

    # Apply CLI overrides
    effective_llm = llm_override or profile.llm
    effective_auto_approve = auto_approve_override if auto_approve_override is not None else profile.auto_approve

    return config, profile, effective_llm, effective_auto_approve


def _list_profiles() -> None:
    """Print a Rich table of all configured profiles."""
    config = _load_config()

    if not config.profiles:
        console.print("[yellow]No profiles defined in config/basic/agents/deepagent.yaml[/yellow]")
        return

    table = Table(title="Deep Agent Profiles")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("LLM", style="magenta")
    table.add_column("Auto-approve", style="yellow")
    table.add_column("Memory", style="green")
    table.add_column("Shell", style="blue")
    table.add_column("Sandbox", style="dim")
    table.add_column("Description", style="dim")

    default = config.default_profile
    for p in config.profiles:
        name = f"* {p.name}" if default and p.name == default else p.name
        table.add_row(
            name,
            p.llm or "(global default)",
            "✓" if p.auto_approve else "✗",
            "✓" if p.enable_memory else "✗",
            "✓" if p.enable_shell else "✗",
            p.sandbox,
            p.description,
        )

    console.print(table)
    if default:
        console.print("[dim]* = default profile[/dim]")


# ---------------------------------------------------------------------------
# Async TUI runner (in-process)
# ---------------------------------------------------------------------------


async def _run_tui_async(
    *,
    llm_id: str | None,
    config,
    profile,
    auto_approve: bool,
    assistant_id: str,
    thread_id: str | None,
    initial_prompt: str | None,
) -> int:
    """Launch the deepagents Textual TUI with a genai-tk-resolved model.

    Args:
        llm_id: Resolved LLM identifier (tag or ID), or None for global default.
        config: Loaded DeepagentConfig.
        profile: Active DeepagentProfile.
        auto_approve: Whether to skip HITL approval.
        assistant_id: Deepagents agent name.
        thread_id: Resume a specific thread, or None for a new session.
        initial_prompt: Pre-fill the chat input with this text.

    Returns:
        Exit code (0 = success).
    """
    from deepagents_cli.agent import create_cli_agent
    from deepagents_cli.app import run_textual_app
    from deepagents_cli.sessions import get_checkpointer
    from deepagents_cli.tools import fetch_url, http_request

    from genai_tk.agents.deepagent_cli.llm_bridge import resolve_model_from_profile
    from genai_tk.agents.deepagent_cli.sandbox_bridge import effective_sandbox_type, sandbox_context
    from genai_tk.agents.deepagent_cli.toml_bridge import write_genai_tk_provider

    # Populate the TUI /model switcher with the YAML-curated models list.
    write_genai_tk_provider(config.switcher_models)

    # Build tool list (always include base HTTP tools; add web_search if configured)
    tools = [http_request, fetch_url]
    if "web_search" in profile.tools:
        try:
            from deepagents_cli.tools import web_search

            tools.append(web_search)
        except Exception:
            logger.debug("web_search tool not available (requires TAVILY_API_KEY)")

    model = resolve_model_from_profile(profile, llm_id, config)

    async with sandbox_context(profile, config) as sandbox_backend:
        async with get_checkpointer() as checkpointer:
            agent, backend = create_cli_agent(
                model=model,
                assistant_id=assistant_id,
                tools=tools,
                auto_approve=auto_approve,
                enable_memory=profile.enable_memory,
                enable_skills=profile.enable_skills,
                enable_shell=profile.enable_shell,
                system_prompt=profile.system_prompt,
                checkpointer=checkpointer,
                sandbox=sandbox_backend,
            )

            result = await run_textual_app(
                agent=agent,
                assistant_id=assistant_id,
                backend=backend,
                auto_approve=auto_approve,
                thread_id=thread_id,
                initial_prompt=initial_prompt,
                checkpointer=checkpointer,
                tools=tools,
                sandbox_type=effective_sandbox_type(profile, sandbox_backend),
            )
            return result.return_code


# ---------------------------------------------------------------------------
# Async non-interactive task runner (in-process)
# ---------------------------------------------------------------------------


async def _run_task_async(
    *,
    message: str,
    llm_id: str | None,
    config,
    profile,
    auto_approve: bool,
    assistant_id: str,
    quiet: bool,
) -> int:
    """Run a single task non-interactively using the deepagents agent graph.

    Creates the agent via ``create_cli_agent`` with the genai-tk model and
    invokes it once, streaming the response to stdout.

    Args:
        message: The user task / prompt.
        llm_id: Resolved LLM identifier or None for global default.
        config: Loaded DeepagentConfig.
        profile: Active DeepagentProfile.
        auto_approve: Auto-approve all HITL tool requests.
        assistant_id: Deepagents agent name.
        quiet: Suppress tool diagnostics.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    import uuid

    from deepagents_cli.agent import create_cli_agent
    from deepagents_cli.sessions import get_checkpointer
    from deepagents_cli.tools import fetch_url, http_request
    from langchain_core.messages import AIMessage
    from langgraph.types import Command

    from genai_tk.agents.deepagent_cli.llm_bridge import resolve_model_from_profile
    from genai_tk.agents.deepagent_cli.sandbox_bridge import sandbox_context

    tools = [http_request, fetch_url]
    if "web_search" in profile.tools:
        try:
            from deepagents_cli.tools import web_search

            tools.append(web_search)
        except Exception:
            logger.debug("web_search tool not available (requires TAVILY_API_KEY)")

    # In non-interactive mode: disable shell unless allow-list is set
    shell_allow_list = profile.shell_allow_list
    effective_shell = profile.enable_shell and bool(shell_allow_list)

    model = resolve_model_from_profile(profile, llm_id, config)

    if not quiet:
        console.print(
            f"[dim]Agent:[/dim] {assistant_id}  "
            f"[dim]Memory:[/dim] {'on' if profile.enable_memory else 'off'}  "
            f"[dim]Skills:[/dim] {'on' if profile.enable_skills else 'off'}  "
            f"[dim]Shell:[/dim] {'on' if effective_shell else 'off (use --profile with shell_allow_list)'}"
        )

    async with sandbox_context(profile, config) as sandbox_backend:
        async with get_checkpointer() as checkpointer:
            agent, _backend = create_cli_agent(
                model=model,
                assistant_id=assistant_id,
                tools=tools,
                auto_approve=True,  # Non-interactive always auto-approves
                enable_memory=profile.enable_memory,
                enable_skills=profile.enable_skills,
                enable_shell=effective_shell,
                system_prompt=profile.system_prompt,
                checkpointer=checkpointer,
                sandbox=sandbox_backend,
            )

            thread_id = str(uuid.uuid4())
            run_config: dict = {"configurable": {"thread_id": thread_id}}
            stream_input: dict | Command = {"messages": [{"role": "user", "content": message}]}

            full_response: list[str] = []
            iterations = 0
            max_hitl_iterations = 10

            while True:
                async for chunk in agent.astream(
                    stream_input,
                    stream_mode=["messages", "updates"],
                    subgraphs=True,
                    config=run_config,
                    durability="exit",
                ):
                    if not isinstance(chunk, tuple) or len(chunk) != 3:
                        continue
                    namespace, chunk_mode, data = chunk
                    if namespace:  # Skip sub-agent output
                        continue

                    if chunk_mode == "updates" and isinstance(data, dict) and "__interrupt__" in data:
                        # Auto-approve all HITL interrupts (auto_approve=True above handles most,
                        # but we catch any remaining ones here)
                        interrupts = data["__interrupt__"]
                        if interrupts:
                            hitl_response: dict = {}
                            for interrupt_obj in interrupts:
                                hitl_response[interrupt_obj.id] = {"decisions": [{"type": "approve"}]}
                            stream_input = Command(resume=hitl_response)
                            iterations += 1
                            if iterations > max_hitl_iterations:
                                console.print("[red]HITL iteration limit reached, aborting.[/red]")
                                return 1
                            break  # Re-enter stream loop with resume command
                    elif chunk_mode == "messages" and isinstance(data, tuple) and len(data) == 2:
                        msg_obj, meta = data
                        if meta and meta.get("lc_source") == "summarization":
                            continue
                        if isinstance(msg_obj, AIMessage):
                            content = msg_obj.content
                            if isinstance(content, str) and content:
                                sys.stdout.write(content)
                                sys.stdout.flush()
                                full_response.append(content)
                            elif isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        text = block.get("text", "")
                                        if text:
                                            sys.stdout.write(text)
                                            sys.stdout.flush()
                                            full_response.append(text)
                        elif not quiet and hasattr(msg_obj, "name"):
                            # Tool result — show brief notification
                            pass  # keep output clean in non-interactive mode
                else:
                    break  # Normal stream end — exit while loop

            if full_response:
                sys.stdout.write("\n")
                sys.stdout.flush()

            if not quiet:
                console.print("[green]✓ Task completed[/green]")

            return 0


# ---------------------------------------------------------------------------
# Command class
# ---------------------------------------------------------------------------


class DeepagentCommands(CliTopCommand, BaseModel):
    """CLI commands for deepagents-cli integration (in-process)."""

    description: str = "Deep Agents: persistent coding agent with memory, skills and tool use"

    def get_description(self) -> tuple[str, str]:
        """Return command name and description."""
        return "deepagent", self.description

    def register(self, cli_app: typer.Typer) -> None:
        """Register this command group, disabling no_args_is_help so the TUI launches by default."""
        command_name, description = self.get_description()
        sub_app = typer.Typer(no_args_is_help=False, help=description)
        self.register_sub_commands(sub_app)
        cli_app.add_typer(sub_app, name=command_name)

    def register_sub_commands(self, cli_app: typer.Typer) -> None:  # noqa: C901
        """Register deepagent subcommands."""

        @cli_app.callback(invoke_without_command=True)
        def main(  # noqa: C901
            ctx: typer.Context,
            profile: Annotated[
                Optional[str],
                typer.Option("--profile", "-p", help="Profile name from deepagent.yaml"),
            ] = None,
            llm: Annotated[
                Optional[str],
                typer.Option("--llm", "-m", help="LLM override (tag or ID, e.g. fast_model, gpt41mini@openai)"),
            ] = None,
            agent: Annotated[
                str,
                typer.Option("--agent", "-a", help="Agent name (maps to ~/.deepagents/<name>/)"),
            ] = "agent",
            auto_approve: Annotated[
                bool,
                typer.Option("--auto-approve", help="Skip human-in-the-loop tool approval prompts"),
            ] = False,
            resume: Annotated[
                Optional[str],
                typer.Option("--resume", "-r", help="Resume a thread by ID (or omit for most recent)"),
            ] = None,
            prompt: Annotated[
                Optional[str],
                typer.Option("--prompt", "-q", help="Pre-fill the chat input with this text at startup"),
            ] = None,
            list_profiles: Annotated[
                bool,
                typer.Option("--list-profiles", "-l", help="List configured profiles and exit"),
            ] = False,
        ) -> None:
            """Launch the Deep Agents interactive TUI (default when no subcommand is given).

            Uses the genai-tk LlmFactory to create the model — bypassing
            deepagents-cli's own model pipeline — then hands off to the
            Textual TUI for the interactive session.

            Examples:
                cli agents deepagent
                cli agents deepagent --llm fast_model
                cli agents deepagent --profile coder --agent mybot
                cli agents deepagent --resume -m "Continue the refactor"
                cli agents deepagent --list-profiles
            """
            if ctx.invoked_subcommand is not None:
                return  # A subcommand was invoked — let it handle execution

            if list_profiles:
                _list_profiles()
                return

            try:
                _cfg, _profile, effective_llm, effective_auto_approve = _resolve_profile_settings(
                    profile, llm, auto_approve or None
                )
            except typer.Exit:
                raise
            except Exception as e:
                console.print(f"[red]Config error:[/red] {e}")
                raise typer.Exit(1) from e

            if profile:
                console.print(
                    f"[dim]Profile:[/dim] {_profile.name}  [dim]LLM:[/dim] {effective_llm or '(global default)'}"
                )

            try:
                exit_code = asyncio.run(
                    _run_tui_async(
                        llm_id=effective_llm,
                        config=_cfg,
                        profile=_profile,
                        auto_approve=effective_auto_approve,
                        assistant_id=agent,
                        thread_id=resume,
                        initial_prompt=prompt,
                    )
                )
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")
                raise typer.Exit(0) from None
            except typer.Exit:
                raise
            except Exception as e:
                console.print(f"\n[red]Error:[/red] {e}")
                logger.exception("deepagent TUI error")
                raise typer.Exit(1) from e
            raise typer.Exit(exit_code)

        @cli_app.command("task")
        def task_cmd(
            message: Annotated[str, typer.Argument(help="Task description to execute non-interactively")],
            profile: Annotated[
                Optional[str],
                typer.Option("--profile", "-p", help="Profile name from deepagent.yaml"),
            ] = None,
            llm: Annotated[
                Optional[str],
                typer.Option("--llm", "-m", help="LLM override (tag or ID)"),
            ] = None,
            agent: Annotated[
                str,
                typer.Option("--agent", "-a", help="Agent name"),
            ] = "agent",
            auto_approve: Annotated[
                bool,
                typer.Option("--auto-approve", help="Auto-approve tool requests"),
            ] = True,
            quiet: Annotated[
                bool,
                typer.Option("--quiet", "-q", help="Suppress diagnostic output"),
            ] = False,
        ) -> None:
            """Run a single task non-interactively and exit.

            Uses the same in-process agent as the TUI but streams response
            to stdout without the Textual interface. Shell commands are
            disabled by default — set shell_allow_list in the profile to
            enable them.

            Examples:
                cli agents deepagent task "Write a Python hello-world script"
                cli agents deepagent task --profile coder "Fix the failing tests"
                cli agents deepagent task -q "List all TODO comments" > todos.txt
            """
            try:
                _cfg, _profile, effective_llm, effective_auto_approve = _resolve_profile_settings(
                    profile, llm, auto_approve or None
                )
            except typer.Exit:
                raise
            except Exception as e:
                console.print(f"[red]Config error:[/red] {e}")
                raise typer.Exit(1) from e

            try:
                exit_code = asyncio.run(
                    _run_task_async(
                        message=message,
                        llm_id=effective_llm,
                        config=_cfg,
                        profile=_profile,
                        auto_approve=effective_auto_approve,
                        assistant_id=agent,
                        quiet=quiet,
                    )
                )
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")
                raise typer.Exit(0) from None
            except typer.Exit:
                raise
            except Exception as e:
                console.print(f"\n[red]Error:[/red] {e}")
                logger.exception("deepagent task error")
                raise typer.Exit(1) from e
            raise typer.Exit(exit_code)

        @cli_app.command("list")
        def list_cmd(
            verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show full paths")] = False,
        ) -> None:
            """List all configured deepagent agent directories.

            Example:
                cli agents deepagent list
            """
            try:
                from deepagents_cli.main import list_agents

                list_agents(verbose=verbose)
            except TypeError:
                # Older versions may not accept verbose
                from deepagents_cli.main import list_agents  # type: ignore[assignment]

                list_agents()

        @cli_app.command("reset")
        def reset_cmd(
            agent: Annotated[
                str,
                typer.Option("--agent", "-a", help="Agent name to reset"),
            ] = "agent",
            target: Annotated[
                Optional[str],
                typer.Option("--target", "-t", help="Reset target: memories | skills | threads | all"),
            ] = None,
        ) -> None:
            """Reset agent state (memories, skills, threads, or all).

            Example:
                cli agents deepagent reset --agent mybot --target memories
            """
            try:
                from deepagents_cli.main import reset_agent

                reset_agent(assistant_id=agent, target=target)
            except Exception as e:
                console.print(f"[red]Reset failed:[/red] {e}")
                raise typer.Exit(1) from e

        # ── Skills sub-group ─────────────────────────────────────────────────
        skills_app = typer.Typer(no_args_is_help=True, help="Manage deepagent skills")
        cli_app.add_typer(skills_app, name="skills")

        @skills_app.command("list")
        def skills_list(
            agent: Annotated[str, typer.Option("--agent", "-a", help="Agent name")] = "agent",
        ) -> None:
            """List installed agent skills.

            Example:
                cli agents deepagent skills list
            """
            try:
                from deepagents_cli.skills import execute_skills_command

                execute_skills_command(["list", "--agent", agent])
            except Exception as e:
                console.print(f"[red]Skills list failed:[/red] {e}")
                raise typer.Exit(1) from e

        @skills_app.command("create")
        def skills_create(
            name: Annotated[str, typer.Argument(help="Skill name")],
            agent: Annotated[str, typer.Option("--agent", "-a", help="Agent name")] = "agent",
        ) -> None:
            """Create a new skill scaffold.

            Example:
                cli agents deepagent skills create web-research
            """
            try:
                from deepagents_cli.skills import execute_skills_command

                execute_skills_command(["create", name, "--agent", agent])
            except Exception as e:
                console.print(f"[red]Skill creation failed:[/red] {e}")
                raise typer.Exit(1) from e

        @skills_app.command("info")
        def skills_info(
            name: Annotated[str, typer.Argument(help="Skill name")],
            agent: Annotated[str, typer.Option("--agent", "-a", help="Agent name")] = "agent",
        ) -> None:
            """Show details for a specific skill.

            Example:
                cli agents deepagent skills info web-research
            """
            try:
                from deepagents_cli.skills import execute_skills_command

                execute_skills_command(["info", name, "--agent", agent])
            except Exception as e:
                console.print(f"[red]Skill info failed:[/red] {e}")
                raise typer.Exit(1) from e

        # ── Threads sub-group ────────────────────────────────────────────────
        threads_app = typer.Typer(no_args_is_help=True, help="Manage deepagent conversation threads")
        cli_app.add_typer(threads_app, name="threads")

        @threads_app.command("list")
        def threads_list(
            agent: Annotated[str, typer.Option("--agent", "-a", help="Agent name")] = "agent",
        ) -> None:
            """List conversation threads for an agent.

            Example:
                cli agents deepagent threads list
            """
            try:
                from deepagents_cli.main import list_threads_command

                list_threads_command(assistant_id=agent)
            except Exception as e:
                console.print(f"[red]Thread list failed:[/red] {e}")
                raise typer.Exit(1) from e

        @threads_app.command("delete")
        def threads_delete(
            thread_id: Annotated[str, typer.Argument(help="Thread ID to delete")],
            agent: Annotated[str, typer.Option("--agent", "-a", help="Agent name")] = "agent",
        ) -> None:
            """Delete a conversation thread.

            Example:
                cli agents deepagent threads delete abc123
            """
            try:
                from deepagents_cli.main import delete_thread_command

                delete_thread_command(assistant_id=agent, thread_id=thread_id)
            except Exception as e:
                console.print(f"[red]Thread delete failed:[/red] {e}")
                raise typer.Exit(1) from e
