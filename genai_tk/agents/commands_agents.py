"""CLI commands for AI Agent functionality.

Thin coordinator that registers all agent sub-commands by delegating
to per-agent-type modules:

- ``langchain`` — Unified LangChain agents (react | deep | custom)
- ``smolagents`` — SmolAgents CodeAct (``smol`` command)
- ``deer_flow`` — Deer-flow agents (``deerflow`` command group)
- ``deepagent`` — deepagents-cli integration (``deepagent`` command group)
"""

import typer

from genai_tk.cli.base import CliTopCommand


class AgentCommands(CliTopCommand):
    description: str = "Commands to create Autonomous Agents"

    def get_description(self) -> tuple[str, str]:
        return "agents", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        from genai_tk.agents.deepagent.cli_commands import DeepagentCommands
        from genai_tk.agents.deer_flow.cli_commands import DeerFlowCommands
        from genai_tk.agents.langchain.commands import register as register_langchain
        from genai_tk.agents.smolagents.commands import register as register_smol

        register_langchain(cli_app)
        register_smol(cli_app)
        DeerFlowCommands().register(cli_app)
        DeepagentCommands().register(cli_app)
