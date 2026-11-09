from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from genai_tk.ui.textual.base import BaseTuiApp
from genai_tk.ui.textual.constants import QUIT_COMMANDS, STANDARD_QUIT_BINDINGS
from genai_tk.ui.textual.screens.document_modal import DocumentViewerModal
from genai_tk.ui.textual.screens.document_resolver import (
    open_document_modal,
    resolve_document_or_section,
)


class DummyModalHostApp(BaseTuiApp):
    """Dummy app for testing modal screens and standard quit bindings."""

    def compose(self) -> ComposeResult:
        yield Static("Main Screen Content", id="main-content")


@pytest.mark.asyncio
async def test_standard_quit_bindings() -> None:
    """Ensure standard quit bindings include q as primary and ctrl+q as alternative."""
    keys = {b.key: b for b in STANDARD_QUIT_BINDINGS}
    assert "q" in keys
    assert keys["q"].action == "quit"
    assert "ctrl+q" in keys
    assert "/q" in QUIT_COMMANDS


@pytest.mark.asyncio
async def test_document_viewer_modal_push_and_dismiss() -> None:
    """Test pushing and dismissing DocumentViewerModal with ESC and q (one level at a time)."""
    app = DummyModalHostApp()
    async with app.run_test() as pilot:
        assert len(app.screen_stack) == 1

        modal = DocumentViewerModal(
            title="Test Document",
            content="# Heading\n\nSome test markdown content.",
            subtitle="file_1234::1",
        )
        app.push_screen(modal)
        await pilot.pause()

        assert len(app.screen_stack) == 2
        assert isinstance(app.screen, DocumentViewerModal)

        # Press escape to dismiss (pop_screen)
        await pilot.press("escape")
        await pilot.pause()

        assert len(app.screen_stack) == 1
        assert not isinstance(app.screen, DocumentViewerModal)
        assert not app._exit

        # Push again and test 'q' closes modal without exiting the app
        modal2 = DocumentViewerModal(
            title="Test Document 2",
            content="Content 2",
        )
        app.push_screen(modal2)
        await pilot.pause()
        assert len(app.screen_stack) == 2

        await pilot.press("q")
        await pilot.pause()
        assert len(app.screen_stack) == 1
        assert not app._exit


@pytest.mark.asyncio
async def test_quit_one_level_at_a_time() -> None:
    """Test action_quit dismisses top modal when open, and only exits app when at root."""
    app = DummyModalHostApp()
    async with app.run_test() as pilot:
        modal = DocumentViewerModal(title="Test", content="Content")
        app.push_screen(modal)
        await pilot.pause()
        assert len(app.screen_stack) == 2

        # action_quit on modal pops screen
        app.action_quit()
        await pilot.pause()
        assert len(app.screen_stack) == 1
        assert not app._exit

        # action_quit at root exits app
        app.action_quit()
        await pilot.pause()
        assert app._exit


@pytest.mark.asyncio
async def test_open_document_modal_helper() -> None:
    """Test the open_document_modal helper function."""
    app = DummyModalHostApp()
    async with app.run_test() as pilot:
        open_document_modal(
            app,
            "doc://dummy_document.md",
            fallback_content="### Fallback Content",
            title="Custom Title",
        )
        await pilot.pause()

        assert len(app.screen_stack) == 2
        modal = app.screen
        assert isinstance(modal, DocumentViewerModal)
        assert modal.doc_title == "Custom Title"
        assert "Fallback Content" in modal.doc_content

        # Close via button click
        btn = modal.query_one("#modal-close-btn")
        btn.press()
        await pilot.pause()

        assert len(app.screen_stack) == 1


def test_resolve_document_or_section_fallback() -> None:
    """Test resolving non-existent target falls back gracefully without raising."""
    title, content, subtitle = resolve_document_or_section("doc://non_existent_doc_123.md")
    assert "non_existent_doc_123.md" in title
    assert "Referenced Resource" in content
    assert subtitle == "Resource Reference (Unresolved)"
