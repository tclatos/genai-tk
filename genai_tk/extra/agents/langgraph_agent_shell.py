import webbrowser
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from genai_tk.core.llm_factory import get_llm
from genai_tk.core.mcp_client import get_mcp_servers_dict
from genai_tk.utils.agent_middleware import create_rich_agent_middlewares
from genai_tk.utils.markdown import looks_like_markdown


async def run_langgraph_agent_shell(
    llm_id: str | None,
    tools: list[BaseTool] = [],
    mcp_server_names: list[str] = [],
    system_prompt: str | None = None,
) -> None:
    """Run an interactive shell for sending prompts to a LanggGraph ReAct agent.

    The MCP servers are started once before entering the shell loop.
    The user can type /quit to exit the shell.

    Args:
        llm_id: Optional ID of the language model to use
        server_filter: Optional list of server names to include in the agent
    """
    console = Console()

    # Display welcome banner
    welcome_text = Text("🤖 LangGraph Agent", style="bold cyan")
    if mcp_server_names:
        welcome_text.append(f"\nConnected to MCP servers: {', '.join(mcp_server_names)}", style="green")

    console.print(Panel(welcome_text, title="Welcome", border_style="bright_blue"))
    console.print("[dim]Commands: /help, /quit, /trace\nUse up/down arrows to navigate prompt history[/dim]\n")

    model = get_llm(llm=llm_id)
    if mcp_server_names:
        with console.status("[bold green]Connecting to MCP servers..."):
            client = MultiServerMCPClient(get_mcp_servers_dict(mcp_server_names))
            tools = tools + await client.get_tools()
            console.print("[green]✓ MCP servers connected[/green]\n")

    config = {"configurable": {"thread_id": "1"}}
    middleware = create_rich_agent_middlewares(console=console)
    if system_prompt:
        agent = create_agent(
            model,
            tools,
            system_prompt=system_prompt,
            checkpointer=MemorySaver(),
            middleware=middleware,
        )
    else:
        agent = create_agent(
            model,
            tools,
            checkpointer=MemorySaver(),
            middleware=middleware,
        )

    # Set up prompt history
    history_file = Path(".blueprint.input.history")
    session = PromptSession(history=FileHistory(str(history_file)))

    while True:
        try:
            with patch_stdout():
                prompt_style = Style.from_dict(
                    {
                        "prompt": "bold cyan",
                    }
                )
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

            # Display user prompt with styling
            console.print(Panel(user_input, title="[bold blue]User[/bold blue]", border_style="blue"))

            # Process the response
            with console.status("[bold green]Agent is thinking...\n[/bold green]"):
                result = await agent.ainvoke({"messages": user_input}, config)

            # Render the final assistant message
            final_message = None
            if isinstance(result, dict) and "messages" in result:
                out_messages = result["messages"]
                final_message = out_messages[-1] if out_messages else None
            else:
                final_message = result

            if final_message is not None:
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

            console.print()  # Add spacing between interactions

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Received keyboard interrupt. Exiting...[/bold yellow]")
            break
        except Exception as e:
            console.print(Panel(f"[red]Error: {str(e)}[/red]", title="[bold red]Error[/bold red]", border_style="red"))
