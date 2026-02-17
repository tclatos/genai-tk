import webbrowser
from collections.abc import Sequence
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.language_models.base import LanguageModelInput, LanguageModelOutput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
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

from genai_tk.cli.rich_langchain_middleware import create_rich_agent_middlewares
from genai_tk.core.llm_factory import get_llm
from genai_tk.core.mcp_client import get_mcp_servers_dict
from genai_tk.utils.markdown import looks_like_markdown


def _render_final_message(result: object, console: Console) -> None:
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


async def _prepare_rich_agent(
    llm_id: str | None,
    *,
    base_tools: Sequence[BaseTool] | None = None,
    mcp_server_names: Sequence[str] | None = None,
    system_prompt: str | None = None,
    console: Console | None = None,
    use_memory: bool = False,
    single_tool_mode: bool = False,
) -> tuple[BaseChatModel, object, MultiServerMCPClient | None, list[BaseTool]]:
    """Create a LangChain agent with optional MCP tools and Rich middleware.

    Returns the model, agent, optional MCP client (for cleanup) and full tool list.
    """
    rich_console = console or Console()

    model = get_llm(llm=llm_id)
    all_tools: list[BaseTool] = list(base_tools or [])

    client: MultiServerMCPClient | None = None
    if mcp_server_names:
        with rich_console.status("[bold green]Connecting to MCP servers..."):
            client = MultiServerMCPClient(get_mcp_servers_dict(list(mcp_server_names)))
            mcp_tools = await client.get_tools()
            all_tools.extend(mcp_tools)
            rich_console.print("[green]✓ MCP servers connected[/green]\n")

    middleware = create_rich_agent_middlewares(console=rich_console, single_tool_mode=single_tool_mode)

    agent_kwargs: dict = {
        "model": model,
        "tools": all_tools,
        "middleware": middleware,
    }

    if use_memory:
        agent_kwargs["checkpointer"] = MemorySaver()

    if system_prompt:
        agent_kwargs["system_prompt"] = system_prompt

    agent = create_agent(**agent_kwargs)
    return model, agent, client, all_tools


async def run_langchain_agent_shell(
    llm_id: str | None,
    tools: list[BaseTool] | None = None,
    mcp_server_names: list[str] | None = None,
    system_prompt: str | None = None,
) -> None:
    """Run an interactive shell for sending prompts to a LangChain ReAct agent.

    The MCP servers are started once before entering the shell loop.
    The user can type /quit to exit the shell.

    Args:
        llm_id: Optional ID of the language model to use
        mcp_server_names: Optional list of MCP server names to include in the agent
        system_prompt: Optional system prompt / pre-prompt
    """
    console = Console()

    # Display welcome banner
    welcome_text = Text("🤖 LangChain Agent", style="bold cyan")
    if mcp_server_names:
        welcome_text.append(f"\nConnected to MCP servers: {', '.join(mcp_server_names)}", style="green")

    console.print(Panel(welcome_text, title="Welcome", border_style="bright_blue"))
    console.print("[dim]Commands: /help, /quit, /trace\nUse up/down arrows to navigate prompt history[/dim]\n")

    # Create agent with memory suitable for a chat shell
    _, agent, client, _ = await _prepare_rich_agent(
        llm_id,
        base_tools=tools or [],
        mcp_server_names=mcp_server_names or [],
        system_prompt=system_prompt,
        console=console,
        use_memory=True,
    )

    # Set up prompt history
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

                # Display user prompt with styling
                console.print(Panel(user_input, title="[bold blue]User[/bold blue]", border_style="blue"))

                # Process the response (LangGraph style: string messages)
                with console.status("[bold green]Agent is thinking...\n[/bold green]"):
                    result = await agent.ainvoke({"messages": user_input}, {"configurable": {"thread_id": "1"}})

                _render_final_message(result, console)
                console.print()  # Add spacing between interactions

            except KeyboardInterrupt:
                console.print("\n[bold yellow]Received keyboard interrupt. Exiting...[/bold yellow]")
                break
            except Exception as e:  # pragma: no cover - defensive logging
                console.print(
                    Panel(f"[red]Error: {str(e)}[/red]", title="[bold red]Error[/bold red]", border_style="red")
                )
    finally:
        if client is not None and hasattr(client, "close"):
            await client.close()


async def run_langchain_agent_direct(
    query: str,
    llm_id: str | None = None,
    mcp_server_names: list[str] | None = None,
    additional_tools: list[BaseTool] | None = None,
    pre_prompt: str | None = None,
    single_tool_mode: bool = False,
) -> None:
    """Execute a single query using MCP tools with a ReAct agent and Rich output.

    This function consolidates the one-shot ReAct agent previously implemented
    as `call_react_agent` in `genai_tk.core.mcp_client`.

    Args:
        query: The user query to execute
        llm_id: Optional ID of the language model to use
        mcp_server_names: Optional list of MCP server names to include
        additional_tools: Optional list of additional tools to provide to the agent
        pre_prompt: Optional system prompt
        single_tool_mode: If True, agent will stop after calling one tool and return raw result
    """
    from loguru import logger

    console = Console()

    # Create agent without conversational memory for direct calls
    model, agent, client, all_tools = await _prepare_rich_agent(
        llm_id,
        base_tools=additional_tools or [],
        mcp_server_names=mcp_server_names or [],
        system_prompt=pre_prompt,
        console=console,
        use_memory=False,
        single_tool_mode=single_tool_mode,
    )

    try:
        tool_names = [getattr(t, "name", str(type(t).__name__)) for t in all_tools]
        logger.info(f"ReAct agent created with {len(all_tools)} tools: {', '.join(tool_names)}")

        # In single-tool mode, use direct executor (pragmatic approach)
        if single_tool_mode and len(all_tools) == 1:
            from genai_tk.cli.rich_langchain_middleware import (
                RichToolCallMiddleware,
                SingleToolExecutorMiddleware,
            )

            executor = SingleToolExecutorMiddleware()
            rich_middleware = RichToolCallMiddleware(console=console)

            # Execute tool with Rich display via middleware
            await executor.execute_single_tool(
                tool=all_tools[0],
                model=model,
                query=query,
                pre_prompt=pre_prompt,
                tool_wrapper=rich_middleware.awrap_tool_call,
            )
            return

        # Normal agent flow
        messages: list[HumanMessage] = []
        if pre_prompt:
            messages.append(HumanMessage(content=pre_prompt))
        messages.append(HumanMessage(content=query))

        # Invoke the agent and render the final answer with Rich
        result = await agent.ainvoke({"messages": messages})
        _render_final_message(result, console)

    finally:
        if client is not None and hasattr(client, "close"):
            await client.close()
