"""CLI commands for AI Agent functionality.

Thin coordinator that registers all agent sub-commands by delegating
to per-agent-type modules:

- ``langchain`` — LangChain ReAct agent (``react`` command)
- ``smolagents`` — SmolAgents CodeAct (``smol`` command)
- ``deep`` — Deep planning agents (``deep`` command)
- ``deer_flow`` — Deer-flow agents (``deerflow`` command group)
"""

import typer

from genai_tk.main.cli import CliTopCommand


class AgentCommands(CliTopCommand):
    description: str = "Commands to create Autonomus Agents"

    def get_description(self) -> tuple[str, str]:
        return "agents", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        from genai_tk.agents.deep.commands import register as register_deep
        from genai_tk.agents.deer_flow.cli_commands import DeerFlowCommands
        from genai_tk.agents.langchain.commands import register as register_react
        from genai_tk.agents.smolagents.commands import register as register_smol

        register_react(cli_app)
        register_smol(cli_app)
        register_deep(cli_app)
        DeerFlowCommands().register(cli_app)
