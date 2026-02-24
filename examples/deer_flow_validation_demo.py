"""Demonstration of deer-flow validation and error handling.

This script shows how the validation functions provide clear, helpful error messages
when invalid configurations are provided.
"""

from rich.console import Console

from genai_tk.agents.deer_flow.profile import (
    DeerFlowProfile as DeerFlowAgentConfig,
)
from genai_tk.agents.deer_flow.profile import (
    InvalidModeError,
    MCPServerNotFoundError,
    ProfileNotFoundError,
    load_deer_flow_profiles,
    validate_mcp_servers,
    validate_mode,
    validate_profile_name,
)

console = Console()


def demo_profile_validation():
    """Demonstrate profile name validation."""
    console.rule("[cyan]Profile Validation Demo[/cyan]")

    # Create sample profiles
    profiles = [
        DeerFlowAgentConfig(name="Research Assistant", description="AI researcher"),
        DeerFlowAgentConfig(name="Coder", description="Coding helper"),
        DeerFlowAgentConfig(name="Web Browser", description="Web search agent"),
    ]

    # Valid profile (case insensitive)
    try:
        profile = validate_profile_name("research assistant", profiles)
        console.print(f"✅ Found profile: [green]{profile.name}[/green]")
    except ProfileNotFoundError as e:
        console.print(f"❌ {e}")

    # Invalid profile
    try:
        profile = validate_profile_name("NonExistent", profiles)
        console.print(f"✅ Found profile: [green]{profile.name}[/green]")
    except ProfileNotFoundError as e:
        console.print(f"[red]❌ {e}[/red]")

    console.print()


def demo_mode_validation():
    """Demonstrate mode validation."""
    console.rule("[cyan]Mode Validation Demo[/cyan]")

    # Valid modes
    for mode in ["flash", "THINKING", "Pro", "ultra"]:
        try:
            validated = validate_mode(mode)
            console.print(f"✅ Mode '{mode}' -> [green]{validated}[/green]")
        except InvalidModeError as e:
            console.print(f"❌ {e}")

    # Invalid mode
    console.print()
    try:
        validated = validate_mode("turbo")
        console.print(f"✅ Mode validated: [green]{validated}[/green]")
    except InvalidModeError as e:
        console.print(f"[red]❌ {e}[/red]")

    console.print()


def demo_mcp_validation():
    """Demonstrate MCP server validation."""
    console.rule("[cyan]MCP Server Validation Demo[/cyan]")

    # Note: This will use actual config, so results depend on your mcp_servers.yaml

    # Empty list
    try:
        result = validate_mcp_servers([])
        console.print(f"✅ Empty MCP list validated: [green]{result}[/green]")
    except MCPServerNotFoundError as e:
        console.print(f"❌ {e}")

    # Invalid servers (assuming these don't exist in config)
    try:
        result = validate_mcp_servers(["invalid1", "invalid2"])
        console.print(f"✅ MCP servers validated: [green]{result}[/green]")
    except MCPServerNotFoundError as e:
        console.print(f"[red]❌ {e}[/red]")

    console.print()


def demo_config_loading():
    """Demonstrate config loading with error handling."""
    console.rule("[cyan]Config Loading Demo[/cyan]")

    try:
        profiles = load_deer_flow_profiles()
        console.print(f"✅ Loaded [green]{len(profiles)}[/green] profiles:")
        for p in profiles:
            console.print(f"  • {p.name} - {p.description}")
    except FileNotFoundError as e:
        console.print(f"[red]❌ File not found: {e}[/red]")
    except ValueError as e:
        console.print(f"[red]❌ Invalid config: {e}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}[/red]")

    console.print()


if __name__ == "__main__":
    console.print("\n[bold cyan]🦌 Deer-flow Validation & Error Handling Demo[/bold cyan]\n")

    demo_profile_validation()
    demo_mode_validation()
    demo_mcp_validation()
    demo_config_loading()

    console.print("[bold green]✨ Demo complete![/bold green]\n")
