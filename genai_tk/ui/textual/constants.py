"""Standard bindings and constants for GenAI Toolkit Textual TUIs."""

from __future__ import annotations

from textual.binding import Binding

# Universal quit bindings: 'q' is the primary standard key on top-level screens.
# 'ctrl+q' is available as a secondary alternative (without priority to avoid
# interfering with terminal multiplexers like Zellij/tmux or child modals).
STANDARD_QUIT_BINDINGS: list[Binding] = [
    Binding("q", "quit", "Quit", show=True),
    Binding("ctrl+q", "quit", "Quit", show=False),
]

# Standard chat-style slash commands for quitting text input prompts
QUIT_COMMANDS: tuple[str, ...] = ("/q", "/quit", "/exit", ":q", ":quit")
