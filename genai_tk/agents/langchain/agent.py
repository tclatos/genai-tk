"""LangChain agent runner utilities (react, deep, custom)."""

import webbrowser
from pathlib import Path

from langchain_core.language_models.base import LanguageModelInput, LanguageModelOutput
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from genai_tk.agents.langchain.config import AgentProfileConfig
from genai_tk.agents.langchain.factory import create_langchain_agent
from genai_tk.utils.markdown import looks_like_markdown


def _render_final_message(result: LanguageModelOutput | dict, console: Console) -> None:
    """Render the final assistant message using Rich, handling LangGraph state or direct messages."""
    if isinstance(result, dict) and "messages" in result:
        out_messages = result["messages"]
        final_message = out_messages[-1] if out_messages else None
    else:
        final_message = result

    if final_message is None:
        return

    content = getattr(final_message, "content", str(final_message))
    if isinstance(content, list):
        content = "\n".join(str(block) for block in content)

    if looks_like_markdown(str(content)):
        body = Markdown(str(content))
    else:
        body = str(content)

    console.print(
        Panel(
            body,
            title="[bold white on royal_blue1] Assistant [/bold white on royal_blue1]",
            border_style="royal_blue1",
        )
    )


async def run_langchain_agent_shell(
    profile: AgentProfileConfig,
    llm_override: str | None = None,
    extra_tools: list[BaseTool] | None = None,
    extra_mcp_servers: list[str] | None = None,
) -> None:
    """Run an interactive shell for sending prompts to any LangChain-based agent.

    The agent is created once before entering the shell loop.
    Type ``/quit`` to exit.

    Args:
        profile: Resolved agent profile (includes type, tools, middleware, etc.)
        llm_override: LLM identifier taking precedence over profile.llm
        extra_tools: Additional tools to pass to the agent
        extra_mcp_servers: Additional MCP server names to connect
    """
    console = Console()

    # Info banner
    welcome_text = Text(f"🤖 {profile.name} Agent  [{profile.type}]", style="bold cyan")
    if profile.mcp_servers or extra_mcp_servers:
        all_servers = list(profile.mcp_servers) + list(extra_mcp_servers or [])
        welcome_text.append(f"\nMCP servers: {', '.join(all_servers)}", style="green")

    console.print(Panel(welcome_text, title="Welcome", border_style="bright_blue"))
    console.print("[dim]Commands: /help, /quit, /trace\nUse up/down arrows to navigate prompt history[/dim]\n")

    with console.status("[bold green]Initializing agent...[/bold green]"):
        agent = await create_langchain_agent(
            profile,
            llm_override=llm_override,
            extra_tools=extra_tools,
            extra_mcp_servers=extra_mcp_servers,
            force_memory_checkpointer=True,
        )

    history_file = Path(".blueprint.input.history")
    session = PromptSession(history=FileHistory(str(history_file)))

    try:
        while True:
            try:
                with patch_stdout():
                    prompt_style = Style.from_dict({"prompt": "bold cyan"})
                    user_input = await session.prompt_async(
                        ">>> ", style=prompt_style, auto_suggest=AutoSuggestFromHistory()
                    )

                user_input = user_input.strip()
                if user_input.lower() in ["/quit", "/exit", "/q"]:
                    console.print("\n[bold yellow]Goodbye! 👋[/bold yellow]")
                    break
                if user_input == "/help":
                    console.print(
                        Panel(
                            "/help   – show this help\n"
                            "/quit   – exit the shell\n"
                            "/trace  – open last LangSmith trace in browser",
                            title="[bold cyan]Commands[/bold cyan]",
                            border_style="cyan",
                        )
                    )
                    continue
                if user_input == "/trace":
                    webbrowser.open("https://smith.langchain.com/")
                    continue
                if user_input.startswith("/") and user_input not in {"/quit", "/exit", "/q", "/help", "/trace"}:
                    console.print(f"[red]Unknown command: {user_input}[/red]")
                    continue
                if not user_input:
                    continue

                console.print(Panel(user_input, title="[bold blue]User[/bold blue]", border_style="blue"))

                with console.status("[bold green]Agent is thinking...\n[/bold green]"):
                    result = await agent.ainvoke({"messages": user_input}, {"configurable": {"thread_id": "1"}})

                _render_final_message(result, console)
                console.print()

            except KeyboardInterrupt:
                console.print("\n[bold yellow]Received keyboard interrupt. Exiting...[/bold yellow]")
                break
            except Exception as e:  # pragma: no cover
                console.print(
                    Panel(f"[red]Error: {str(e)}[/red]", title="[bold red]Error[/bold red]", border_style="red")
                )
    finally:
        pass  # MCP client lifecycle is managed inside factory


async def run_langchain_agent_direct(
    query: LanguageModelInput,
    profile: AgentProfileConfig,
    llm_override: str | None = None,
    extra_tools: list[BaseTool] | None = None,
    extra_mcp_servers: list[str] | None = None,
    stream: bool = False,
) -> None:
    """Execute a single query using a LangChain-based agent and render output with Rich.

    Args:
        query: The user query to execute
        profile: Resolved agent profile
        llm_override: LLM identifier taking precedence over profile.llm
        extra_tools: Additional tools to pass to the agent
        extra_mcp_servers: Additional MCP server names to connect
        stream: If True, stream intermediate steps (deep agents only)
    """
    console = Console()

    with console.status("[bold green]Initializing agent...[/bold green]"):
        agent = await create_langchain_agent(
            profile,
            llm_override=llm_override,
            extra_tools=extra_tools,
            extra_mcp_servers=extra_mcp_servers,
            force_memory_checkpointer=False,
        )

    system_prompt = profile.system_prompt or profile.pre_prompt
    messages: list[HumanMessage] = []
    if system_prompt:
        messages.append(HumanMessage(content=system_prompt))
    messages.append(HumanMessage(content=query))

    if stream:
        async for chunk in agent.astream({"messages": messages}):
            _render_final_message(chunk, console)
    else:
        result = await agent.ainvoke({"messages": messages})
        _render_final_message(result, console)
