"""Base Textual App with coherent multi-level exit and command handling."""

from __future__ import annotations

from textual.app import App

from genai_tk.ui.textual.constants import STANDARD_QUIT_BINDINGS


class BaseTuiApp(App[None]):
    """Base Textual App enforcing coherent navigation and single-level quit semantics.

    - Quitting only closes one level at a time: if a child modal or sub-screen is
      open, it is dismissed without terminating the top application.
    - Only when at the root screen does quit return to the parent shell.
    """

    BINDINGS = [*STANDARD_QUIT_BINDINGS]

    def action_quit(self) -> None:
        """Quit one level at a time: dismiss top modal/screen if present, otherwise exit to shell."""
        if len(self.screen_stack) > 1:
            self.pop_screen()
        else:
            self.exit()
