"""Unit tests for sources.py (git repo discovery, URL parsing, and resolution)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from genai_tk.workflow.sources import (
    clone_git_repo,
    git_repo_slug,
    is_git_url,
    parse_git_url,
    resolve_sources,
)


def test_is_git_url() -> None:
    assert is_git_url("https://github.com/tclatos/prefect-yaml")
    assert is_git_url("https://github.com/tclatos/prefect-yaml.git")
    assert is_git_url("https://github.com/tclatos/prefect-yaml/")
    assert is_git_url("git@github.com:tclatos/prefect-yaml.git")
    assert is_git_url("https://gitlab.com/group/repo")
    assert is_git_url("https://bitbucket.org/group/repo")
    assert is_git_url("https://example.com/repo.git")
    assert is_git_url("git://example.com/repo")
    assert is_git_url("ssh://git@example.com/repo")

    assert not is_git_url("https://example.com/page.html")
    assert not is_git_url("https://google.com")
    assert not is_git_url("/data/docs/file.pdf")
    assert not is_git_url("relative/path/file.md")


def test_parse_git_url() -> None:
    repo, ref, sub = parse_git_url("https://github.com/owner/repo")
    assert repo == "https://github.com/owner/repo"
    assert ref is None
    assert sub is None

    repo, ref, sub = parse_git_url("https://github.com/owner/repo/tree/main/docs")
    assert repo == "https://github.com/owner/repo"
    assert ref == "main"
    assert sub == "docs"

    repo, ref, sub = parse_git_url("https://github.com/owner/repo/tree/feature-branch")
    assert repo == "https://github.com/owner/repo"
    assert ref == "feature-branch"
    assert sub is None

    repo, ref, sub = parse_git_url("git@github.com:owner/repo.git")
    assert repo == "git@github.com:owner/repo.git"
    assert ref is None
    assert sub is None


def test_git_repo_slug() -> None:
    assert git_repo_slug("https://github.com/owner/my-repo") == "my-repo"
    assert git_repo_slug("https://github.com/owner/my-repo.git") == "my-repo"
    assert git_repo_slug("https://github.com/owner/my-repo/") == "my-repo"
    assert git_repo_slug("git@github.com:owner/my-repo.git") == "my-repo"


def test_clone_and_resolve_git_repo(tmp_path: Path) -> None:
    # Initialize a local git repository as a test fixture
    repo_dir = tmp_path / "local_origin"
    repo_dir.mkdir()
    subprocess.run(["git", "init", str(repo_dir)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.email", "test@example.com"], check=True)

    # Add a markdown file and a docs subfolder
    (repo_dir / "README.md").write_text("# Test Repo", encoding="utf-8")
    docs_dir = repo_dir / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text("# Guide", encoding="utf-8")

    subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-m", "Initial commit"], check=True)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Test cloning via file:// URL
    file_url = f"file://{repo_dir}"
    cloned_path = clone_git_repo(file_url, cache_dir)
    assert cloned_path.exists()
    assert (cloned_path / "README.md").exists()
    assert (cloned_path / "docs" / "guide.md").exists()

    # Test resolve_sources with git repo
    resolved = resolve_sources([file_url], cache_dir=cache_dir)
    filenames = {rf.path.name for rf in resolved}
    assert "README.md" in filenames
    assert "guide.md" in filenames
    # Ensure .git internals are NOT resolved
    assert not any(".git" in str(rf.path) for rf in resolved)
