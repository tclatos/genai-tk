from rich.panel import Panel
from rich.text import Text


def create_error_panel(title: str, message: str) -> Panel:
    """Create a Rich panel for error messages.

    Args:
        title: Panel title
        message: Error message to display

    Returns:
        Rich Panel with error styling
    """
    return Panel(
        Text(message, style="red"),
        title=f"[red]❌ {title}[/red]",
        border_style="red",
    )


def create_success_panel(title: str, message: str) -> Panel:
    """Create a Rich panel for success messages.

    Args:
        title: Panel title
        message: Success message to display

    Returns:
        Rich Panel with success styling
    """
    return Panel(
        Text(message, style="green"),
        title=f"[green]✅ {title}[/green]",
        border_style="green",
    )


def create_warning_panel(title: str, message: str) -> Panel:
    """Create a Rich panel for warning messages.

    Args:
        title: Panel title
        message: Warning message to display

    Returns:
        Rich Panel with warning styling
    """
    return Panel(
        Text(message, style="yellow"),
        title=f"[yellow]⚠️  {title}[/yellow]",
        border_style="yellow",
    )
