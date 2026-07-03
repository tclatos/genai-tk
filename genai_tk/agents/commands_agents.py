"""CLI commands for AI Agent functionality.

Thin coordinator that registers all agent sub-commands by delegating
to per-agent-type modules:

- ``harness`` — Unified cross-harness commands (``run`` | ``list``)
- ``langchain`` — Unified LangChain agents (react | deep | custom — including DeepAgents)
- ``deer_flow`` — Deer-flow agents (``deerflow`` command group)
"""

import typer

from genai_tk.cli.base import CliTopCommand


class AgentCommands(CliTopCommand):
    description: str = "Commands to create Autonomous Agents"

    def get_description(self) -> tuple[str, str]:
        return "agents", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        from genai_tk.agents.deer_flow.cli_commands import DeerFlowCommands
        from genai_tk.agents.harness.commands import register as register_harness
        from genai_tk.agents.langchain.commands import register as register_langchain

        register_harness(cli_app)
        register_langchain(cli_app)
        DeerFlowCommands().register(cli_app)
