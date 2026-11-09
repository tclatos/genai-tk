"""Generic modal popup screen for viewing Document and MarkdownSection contents."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Markdown, Static


class DocumentViewerModal(ModalScreen[None]):
    """Modal dialog displaying a full Document or MarkdownSection with scrollable preview."""

    DEFAULT_CSS = """
    DocumentViewerModal {
        align: center middle;
        background: rgba(15, 23, 42, 0.85);
    }

    #modal-container {
        width: 94%;
        height: 94%;
        background: #0f172a;
        border: thick #38bdf8;
        padding: 0 1;
    }

    #modal-header {
        height: auto;
        border-bottom: solid #334155;
        padding: 0 0 1 0;
        margin-bottom: 0;
    }

    #modal-title-row {
        height: auto;
        align-vertical: middle;
    }

    #modal-title {
        color: #38bdf8;
        text-style: bold;
        width: 1fr;
        height: auto;
    }

    #modal-subtitle {
        color: #94a3b8;
        height: auto;
        margin: 0;
    }

    #modal-close-btn {
        min-width: 9;
        height: 1;
        border: none;
        background: #dc2626;
        color: #ffffff;
    }

    #modal-scroll {
        height: 1fr;
        background: #090d16;
        padding: 0 1;
        border: none;
    }

    #modal-markdown {
        height: auto;
    }

    #modal-footer {
        height: 1;
        color: #64748b;
        content-align: right middle;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_modal", "Close", show=True),
        Binding("enter", "dismiss_modal", "Close", show=False),
        Binding("q", "dismiss_modal", "Close", show=False),
        Binding("ctrl+q", "quit_app", "Quit", show=True, priority=True),
    ]

    def __init__(
        self,
        title: str,
        content: str,
        subtitle: str | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.doc_title = title
        self.doc_content = content or "*(Empty document or section content)*"
        self.doc_subtitle = subtitle

    def compose(self) -> ComposeResult:
        """Compose modal dialog widgets."""
        with Vertical(id="modal-container"):
            with Vertical(id="modal-header"):
                with Horizontal(id="modal-title-row"):
                    yield Static(f"📄 {self.doc_title}", id="modal-title")
                    yield Button("✖ Close", id="modal-close-btn", variant="error")
                if self.doc_subtitle:
                    yield Static(f"[dim]{self.doc_subtitle}[/dim]", id="modal-subtitle")
            with VerticalScroll(id="modal-scroll"):
                yield Markdown(self.doc_content, id="modal-markdown")
            yield Static("[bold][Esc/Enter/q][/bold] Close  |  [bold][^Q][/bold] Quit", id="modal-footer")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Close button click."""
        if event.button.id == "modal-close-btn":
            self.dismiss()

    def action_dismiss_modal(self) -> None:
        """Dismiss the modal screen and return to caller."""
        self.dismiss()

    def action_quit_app(self) -> None:
        """Exit the application immediately."""
        self.app.exit()
