from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from genai_tk.ui.textual.constants import STANDARD_QUIT_BINDINGS
from genai_tk.ui.textual.screens.document_modal import DocumentViewerModal
from genai_tk.ui.textual.screens.document_resolver import (
    open_document_modal,
    resolve_document_or_section,
)


class DummyModalHostApp(App[None]):
    """Dummy app for testing modal screens and standard quit bindings."""

    BINDINGS = [*STANDARD_QUIT_BINDINGS]

    def compose(self) -> ComposeResult:
        yield Static("Main Screen Content", id="main-content")


@pytest.mark.asyncio
async def test_standard_quit_bindings() -> None:
    """Ensure standard quit bindings include ctrl+q with priority."""
    keys = {b.key: b for b in STANDARD_QUIT_BINDINGS}
    assert "ctrl+q" in keys
    assert keys["ctrl+q"].priority is True
    assert keys["ctrl+q"].action == "quit"
    assert "q" in keys


@pytest.mark.asyncio
async def test_document_viewer_modal_push_and_dismiss() -> None:
    """Test pushing and dismissing DocumentViewerModal."""
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
