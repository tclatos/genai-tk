"""Tests for the project scaffolder (cli init)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    """Create a minimal project directory with config/ that mimics cli init config copy."""
    config = tmp_path / "config"
    config.mkdir()
    # Write a minimal app_conf.yaml
    (config / "app_conf.yaml").write_text(
        "cli:\n  commands:\n    - genai_tk.core.commands_core.CoreCommands\n    - genai_tk.main.cli.register_commands\n"
    )
    # Write a minimal webapp.yaml
    (config / "webapp.yaml").write_text(
        "ui:\n"
        "  app_name: Test Project\n"
        "  logo: null\n"
        "  # pages_dir: example/webapp/pages\n"
        "  # navigation:\n"
        "  #   demos:\n"
        "  #     - demos/example.py\n"
    )
    return tmp_path


class TestSanitizePackageName:
    def test_simple(self):
        from genai_tk.main.scaffolder import _sanitize_package_name

        assert _sanitize_package_name("My Project") == "my_project"

    def test_hyphens(self):
        from genai_tk.main.scaffolder import _sanitize_package_name

        assert _sanitize_package_name("my-cool-app") == "my_cool_app"

    def test_leading_digit(self):
        from genai_tk.main.scaffolder import _sanitize_package_name

        assert _sanitize_package_name("123app") == "my_project"

    def test_special_chars(self):
        from genai_tk.main.scaffolder import _sanitize_package_name

        assert _sanitize_package_name("Hello World! #1") == "hello_world_1"


class TestProjectScaffolder:
    def test_scaffold_creates_expected_files(self, project_dir: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        scaffolder = ProjectScaffolder(project_dir, "Test Project")
        written = scaffolder.scaffold()

        assert written > 0
        pkg = project_dir / "test_project"
        assert pkg.is_dir()
        assert (pkg / "__init__.py").exists()
        assert (pkg / "commands" / "example_commands.py").exists()
        assert (pkg / "chains" / "joke_chain.py").exists()
        assert (pkg / "main" / "streamlit.py").exists()
        assert (pkg / "webapp" / "pages" / "demos" / "hello_agent.py").exists()
        assert (project_dir / "pyproject.toml").exists()
        assert (project_dir / "README.md").exists()
        assert (project_dir / "AGENTS.md").exists()
        assert (project_dir / ".github" / "copilot-instructions.md").exists()

    def test_scaffold_uses_package_name_in_content(self, project_dir: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        scaffolder = ProjectScaffolder(project_dir, "Test Project")
        scaffolder.scaffold()

        pyproject = (project_dir / "pyproject.toml").read_text()
        assert 'name = "test_project"' in pyproject

        agents_md = (project_dir / "AGENTS.md").read_text()
        assert "test_project" in agents_md
        assert "Test Project" in agents_md

    def test_scaffold_does_not_overwrite_without_force(self, project_dir: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        # First run
        scaffolder1 = ProjectScaffolder(project_dir, "Test Project")
        written1 = scaffolder1.scaffold()
        assert written1 > 0

        # Second run without force
        scaffolder2 = ProjectScaffolder(project_dir, "Test Project")
        written2 = scaffolder2.scaffold()
        assert written2 == 0

    def test_scaffold_overwrites_with_force(self, project_dir: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        scaffolder1 = ProjectScaffolder(project_dir, "Test Project")
        scaffolder1.scaffold()

        scaffolder2 = ProjectScaffolder(project_dir, "Test Project", force=True)
        written2 = scaffolder2.scaffold()
        assert written2 > 0

    def test_scaffold_patches_app_conf(self, project_dir: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        scaffolder = ProjectScaffolder(project_dir, "Test Project")
        scaffolder.scaffold()

        app_conf = (project_dir / "config" / "app_conf.yaml").read_text()
        assert "test_project.commands.example_commands.ExampleCommands" in app_conf

    def test_scaffold_patches_webapp_yaml(self, project_dir: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        scaffolder = ProjectScaffolder(project_dir, "Test Project")
        scaffolder.scaffold()

        webapp = (project_dir / "config" / "webapp.yaml").read_text()
        assert "pages_dir: ${paths.project}/test_project/webapp/pages" in webapp
        assert "demos/hello_agent.py" in webapp

    def test_scaffold_patches_makefile_streamlit_entry(self, project_dir: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        # Create a Makefile that mimics the generated one
        (project_dir / "Makefile").write_text("STREAMLIT_ENTRY ?=\n")
        scaffolder = ProjectScaffolder(project_dir, "Test Project")
        scaffolder.scaffold()

        makefile = (project_dir / "Makefile").read_text()
        assert "STREAMLIT_ENTRY ?= test_project/main/streamlit.py" in makefile

    def test_scaffold_ensures_package_mode_in_pyproject(self, project_dir: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        # Simulate a uv-init-style pyproject.toml without package=true
        (project_dir / "pyproject.toml").write_text('[project]\nname = "test_project"\nversion = "0.1.0"\n')
        scaffolder = ProjectScaffolder(project_dir, "Test Project", force=True)
        scaffolder.scaffold()

        pyproject = (project_dir / "pyproject.toml").read_text()
        assert "package = true" in pyproject
        # Must scope package discovery to avoid picking up config/
        assert "[tool.setuptools.packages.find]" in pyproject
        assert 'include = ["test_project*"]' in pyproject
