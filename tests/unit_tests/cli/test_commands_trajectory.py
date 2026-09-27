"""Unit tests for genai_tk.cli.commands_trajectory (CliRunner)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from genai_tk.cli.commands_trajectory import TrajectoryCommands
from genai_tk.utils.trajectory_store import TrajectoryStore


@pytest.fixture
def trajectory_app() -> typer.Typer:
    app = typer.Typer()
    TrajectoryCommands().register(app)
    return app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def populated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TrajectoryStore:
    """Create a TrajectoryStore with a sample recorded run."""
    store_dir = tmp_path / "trajectories"
    store_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(TrajectoryStore, "_resolve_root", classmethod(lambda cls: store_dir))

    store = TrajectoryStore(root=store_dir)

    run_id = "test-run-123"
    run_dir = store_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_id,
        "profile": "react-agent",
        "started_at": "2026-09-27T10:00:00Z",
        "ended_at": "2026-09-27T10:01:00Z",
        "status": "ok",
        "n_llm_calls": 2,
        "n_tool_calls": 1,
        "total_prompt_tokens": 150,
        "total_completion_tokens": 50,
        "tools": ["add_numbers"],
        "skills_loaded": ["math"],
    }
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    events = [
        {
            "kind": "scope",
            "scope_category": "start",
            "category": "agent",
            "name": "Agent",
            "timestamp": "2026-09-27T10:00:00Z",
        },
        {"kind": "mark", "name": "skill.load", "timestamp": "2026-09-27T10:00:05Z", "data": {"skill": "math"}},
        {
            "kind": "scope",
            "scope_category": "start",
            "category": "llm",
            "name": "LLM Call",
            "timestamp": "2026-09-27T10:00:10Z",
        },
        {
            "kind": "scope",
            "scope_category": "end",
            "category": "llm",
            "name": "LLM Call",
            "timestamp": "2026-09-27T10:00:20Z",
        },
        {
            "kind": "scope",
            "scope_category": "end",
            "category": "agent",
            "name": "Agent",
            "timestamp": "2026-09-27T10:01:00Z",
        },
    ]
    events_file = run_dir / "events.jsonl"
    events_file.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    index_line = {
        "run_id": run_id,
        "profile": "react-agent",
        "started_at": "2026-09-27T10:00:00Z",
        "ended_at": "2026-09-27T10:01:00Z",
        "status": "ok",
        "n_llm_calls": 2,
        "n_tool_calls": 1,
        "total_prompt_tokens": 150,
        "total_completion_tokens": 50,
        "tools": ["add_numbers"],
        "skills_loaded": ["math"],
    }
    (store_dir / "index.jsonl").write_text(json.dumps(index_line) + "\n", encoding="utf-8")

    return store


class TestTrajectoryHelp:
    def test_help_exits_zero(self, trajectory_app, runner) -> None:
        result = runner.invoke(trajectory_app, ["trajectory", "--help"])
        assert result.exit_code == 0
        assert "list" in result.stdout
        assert "show" in result.stdout
        assert "stats" in result.stdout


class TestTrajectoryList:
    def test_list_empty(self, trajectory_app, runner, tmp_path, monkeypatch) -> None:
        empty_dir = tmp_path / "empty_trajectories"
        empty_dir.mkdir()
        monkeypatch.setattr(TrajectoryStore, "_resolve_root", classmethod(lambda cls: empty_dir))

        result = runner.invoke(trajectory_app, ["trajectory", "list"])
        assert result.exit_code == 0
        assert "No recorded runs found" in result.stdout

    def test_list_populated(self, trajectory_app, runner, populated_store) -> None:
        result = runner.invoke(trajectory_app, ["trajectory", "list"])
        assert result.exit_code == 0
        assert "test-run-123" in result.stdout
        assert "react-agent" in result.stdout


class TestTrajectoryShow:
    def test_show_nonexistent_run(self, trajectory_app, runner, populated_store) -> None:
        result = runner.invoke(trajectory_app, ["trajectory", "show", "missing-run-id"])
        assert "not found" in result.stdout.lower()

    def test_show_tree_format(self, trajectory_app, runner, populated_store) -> None:
        result = runner.invoke(trajectory_app, ["trajectory", "show", "test-run-123", "--format", "tree"])
        assert result.exit_code == 0

    def test_show_json_format(self, trajectory_app, runner, populated_store) -> None:
        result = runner.invoke(trajectory_app, ["trajectory", "show", "test-run-123", "--format", "json"])
        assert result.exit_code == 0


class TestTrajectorySkillsAndStats:
    def test_skills_command(self, trajectory_app, runner, populated_store) -> None:
        result = runner.invoke(trajectory_app, ["trajectory", "skills", "test-run-123"])
        assert result.exit_code == 0

    def test_stats_command(self, trajectory_app, runner, populated_store) -> None:
        result = runner.invoke(trajectory_app, ["trajectory", "stats"])
        assert result.exit_code == 0

    def test_diff_missing_run(self, trajectory_app, runner, populated_store) -> None:
        result = runner.invoke(trajectory_app, ["trajectory", "diff", "test-run-123", "nonexistent-456"])
        assert result.exit_code == 1

    def test_prune_requires_args(self, trajectory_app, runner, populated_store) -> None:
        result = runner.invoke(trajectory_app, ["trajectory", "prune"])
        assert result.exit_code == 1
