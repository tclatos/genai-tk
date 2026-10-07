"""Unit tests for git_repo_flow."""

from __future__ import annotations

import subprocess
from pathlib import Path

from genai_tk.workflow.prefect.flows.git_repo_flow import git_repo_flow


def test_git_repo_flow_e2e(tmp_path: Path) -> None:
    # Set up a local git repository
    origin_dir = tmp_path / "test_repo"
    origin_dir.mkdir()
    subprocess.run(["git", "init", str(origin_dir)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(origin_dir), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(origin_dir), "config", "user.email", "test@example.com"], check=True)

    (origin_dir / "README.md").write_text("# Test Repository\nContent here", encoding="utf-8")
    docs_dir = origin_dir / "docs"
    docs_dir.mkdir()
    (docs_dir / "architecture.md").write_text("# Architecture\nDetails here", encoding="utf-8")

    subprocess.run(["git", "-C", str(origin_dir), "add", "."], check=True)
    subprocess.run(["git", "-C", str(origin_dir), "commit", "-m", "Initial commit"], check=True)

    md_out = tmp_path / "markdown_out" / "test_repo"

    result = git_repo_flow(
        sources=[f"file://{origin_dir}"],
        md_output_dir=str(md_out),
    )

    assert result["failed"] == 0
    assert result["processed"] >= 2
    assert (md_out / "README.md").exists()
    assert (md_out / "docs" / "architecture.md").exists()
    assert "Test Repository" in (md_out / "README.md").read_text(encoding="utf-8")
    assert "Architecture" in (md_out / "docs" / "architecture.md").read_text(encoding="utf-8")
