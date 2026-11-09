"""Standard bindings and constants for GenAI Toolkit Textual TUIs."""

from __future__ import annotations

from textual.binding import Binding

# Universal quit bindings: ^q (Ctrl-Q) is always priority=True so it triggers
# regardless of whether an Input, DataTable, or TextArea is focused.
# 'q' is kept as a secondary non-priority quit key when not focused in text inputs.
STANDARD_QUIT_BINDINGS: list[Binding] = [
    Binding("ctrl+q", "quit", "Quit", show=True, priority=True),
    Binding("q", "quit", "Quit", show=False),
]
