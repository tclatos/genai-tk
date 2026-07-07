"""CLI commands for AI Agent functionality.

Thin coordinator that registers the unified agent sub-commands:

- ``run`` — Run any LangChain or DeerFlow agent profile through the harness layer
- ``list`` — List all agent profiles across every harness

Framework-specific behaviour is exposed via cross-harness flags on ``run``
(``--chat``, ``--mode``, ``--sandbox``, ``--mcp``) rather than separate
``langchain``/``deerflow`` subcommands.
"""

import typer

from genai_tk.cli.base import CliTopCommand


class AgentCommands(CliTopCommand):
    description: str = "Commands to create Autonomous Agents"

    def get_description(self) -> tuple[str, str]:
        return "agents", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        from genai_tk.agents.harness.commands import register as register_harness

        register_harness(cli_app)
