"""CLI commands for AI Extra functionality."""

import asyncio
import sys
from typing import Annotated

import typer
from loguru import logger
from typer import Option

from genai_tk.cli.base import CliTopCommand


class ExtraCommands(CliTopCommand):
    def get_description(self) -> tuple[str, str]:
        return "tools", "Utilities and extra tools"

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command()
        def gpt_researcher(
            query: Annotated[str, typer.Argument(help="Research query to investigate")],
            config_name: Annotated[
                str, typer.Option("--config", "-c", help="Configuration name from gpt_researcher.yaml")
            ] = "default",
            verbose: Annotated[bool, Option("--verbose", "-v", help="Enable verbose output")] = False,
        ) -> None:
            """Run GPT Researcher with configuration from gpt_researcher.yaml.

            Example:
                uv run cli tools gpt-researcher "Latest developments in AI" --config detailed
            """
            from genai_tk.extra.gpt_researcher_helper import run_gpt_researcher

            try:
                print(f"Running GPT Researcher with config: {config_name}")
                print(f"Query: {query}")
                result = asyncio.run(run_gpt_researcher(query=query, config_name=config_name, verbose=verbose))
                print("\n" + "=" * 80)
                print("RESEARCH REPORT")
                print("=" * 80)
                print(result.report)
            except Exception as e:
                print(f"Error running GPT Researcher: {str(e)}")
                if verbose:
                    import traceback

                    traceback.print_exc()

    def unmaintained_commands(cli_app: typer.Typer) -> None:
        @cli_app.command()
        def browser_agent(
            task: Annotated[str, typer.Argument(help="The task for the browser agent to execute")],
            headless: Annotated[bool, typer.Option(help="Run browser in headless mode")] = False,
            llm: Annotated[str, Option("--llm", "-m", help="LLM identifier (ID or tag from config)")] = "default",
        ) -> None:
            """Launch a browser agent to complete a given task.

            Example:
                uv run cli browser-agent "recent news on Atos" --headless
            """
            from browser_use import Agent, BrowserSession

            from genai_tk.core.factories.llm_factory import get_llm
            from genai_tk.wip.browser_use_langchain import ChatLangchain

            print(f"Running browser agent with task: {task}")
            browser_session = BrowserSession(
                headless=headless,
                window_size={"width": 800, "height": 600},
            )

            llm_model = ChatLangchain(chat=get_llm(llm=llm))
            agent = Agent(task=task, llm=llm_model, browser_session=browser_session)
            history = asyncio.run(agent.run())
            print(history.final_result())
