from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Tree

from genai_tk.cli.trajectory_tui import TrajectoryTuiApp
from genai_tk.extra.monitoring.trajectory_store import TrajectoryStore
from tests.unit_tests.utils.test_trajectory_store import (
    _ROOT,
    _RUN_ID_PARENT,
    _sample_meta,
    _sub_agent_events,
    _write_run,
)


def _tree_labels(tree: Tree[str]) -> list[str]:
    labels: list[str] = []

    def walk(node) -> None:
        for child in node.children:
            labels.append(str(child.label))
            walk(child)

    walk(tree.root)
    return labels


@pytest.mark.asyncio
async def test_tui_timeline_marks_sub_agent_turns(tmp_path: Path) -> None:
    _write_run(tmp_path, _ROOT, _sub_agent_events(_RUN_ID_PARENT), _sample_meta())
    store = TrajectoryStore(root=tmp_path)
    app = TrajectoryTuiApp(initial_run_id=_ROOT, store=store)

    async with app.run_test() as pilot:
        await pilot.pause()
        tree = app.query_one("#timeline-tree", Tree)
        labels = _tree_labels(tree)

    assert any("[sub-agent: pricing]" in label for label in labels)
    assert not any("[sub-agent: None]" in label for label in labels)
