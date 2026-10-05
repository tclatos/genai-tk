"""Tests for the project scaffolder (cli init)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def mock_subprocess_run(monkeypatch):
    """Prevent uv sync from creating real virtualenvs in tmp_path during unit tests."""
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, stdout="", stderr=""))


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    """Create a minimal project directory with config/ that mimics cli init config copy."""
    config = tmp_path / "config"
    config.mkdir()
    # Write a minimal app_conf.yaml
    (config / "app_conf.yaml").write_text(
        "cli:\n  commands:\n    - genai_tk.cli.commands_core.CoreCommands\n    - genai_tk.main.cli.register_commands\n"
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
        assert (pkg / "commands" / "agent_commands.py").exists()
        assert (pkg / "tools" / "example_tool.py").exists()
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
        assert "test_project.commands.agent_commands.AgentCommands" in app_conf

    def test_scaffold_patches_webapp_yaml(self, project_dir: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        scaffolder = ProjectScaffolder(project_dir, "Test Project")
        scaffolder.scaffold()

        webapp = (project_dir / "config" / "webapp.yaml").read_text()
        assert "pages_dir: ${paths.project}/test_project/webapp/pages" in webapp
        assert "demos/hello_agent.py" in webapp

    def test_scaffold_ensures_package_mode_in_pyproject(self, project_dir: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        # Simulate a uv-init-style pyproject.toml without package=true
        (project_dir / "pyproject.toml").write_text('[project]\nname = "test_project"\nversion = "0.1.0"\n')
        scaffolder = ProjectScaffolder(project_dir, "Test Project", force=True)
        scaffolder.scaffold()

        pyproject = (project_dir / "pyproject.toml").read_text()
        assert "package = true" in pyproject
        # Must scope package discovery to avoid picking up config/
        # Template uses hatchling: [tool.hatch.build.targets.wheel] with packages = ["<pkg>"]
        assert "[tool.hatch.build.targets.wheel]" in pyproject
        assert 'packages = ["test_project"]' in pyproject

    def test_scaffolded_pyproject_carries_override_warning(self, project_dir: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        scaffolder = ProjectScaffolder(project_dir, "Test Project")
        scaffolder.scaffold()

        pyproject = (project_dir / "pyproject.toml").read_text()
        assert "override-dependencies" in pyproject
        assert 'requires-python = ">=3.12,<3.13"' in pyproject

    def test_scaffolded_gitignore_excludes_runtime_data(self, project_dir: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        scaffolder = ProjectScaffolder(project_dir, "Test Project")
        scaffolder.scaffold()

        gitignore = (project_dir / ".gitignore").read_text()
        assert "data/*" in gitignore
        assert "!data/sources/" in gitignore


class TestScaffolderPatches:
    """Idempotent pyproject/.gitignore patches applied by `cli init`."""

    def test_tightens_loose_requires_python(self, project_dir: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        (project_dir / "pyproject.toml").write_text('[project]\nname = "test_project"\nrequires-python = ">=3.12"\n')
        ProjectScaffolder(project_dir, "Test Project")._ensure_package_installed()

        assert 'requires-python = ">=3.12,<3.13"' in (project_dir / "pyproject.toml").read_text()

    def test_inserts_requires_python_when_missing(self, project_dir: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        (project_dir / "pyproject.toml").write_text('[project]\nname = "test_project"\n')
        ProjectScaffolder(project_dir, "Test Project")._ensure_package_installed()

        assert 'requires-python = ">=3.12,<3.13"' in (project_dir / "pyproject.toml").read_text()

    def test_keeps_existing_upper_bound_pin(self, project_dir: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        (project_dir / "pyproject.toml").write_text(
            '[project]\nname = "test_project"\nrequires-python = ">=3.12,<3.13"\n'
        )
        ProjectScaffolder(project_dir, "Test Project")._ensure_package_installed()

        pyproject = (project_dir / "pyproject.toml").read_text()
        assert pyproject.count("requires-python") == 1
        assert 'requires-python = ">=3.12,<3.13"' in pyproject

    def test_warns_on_override_dependencies_without_crash(self, project_dir: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        pyproject_path = project_dir / "pyproject.toml"
        pyproject_path.write_text(
            '[project]\nname = "test_project"\nrequires-python = ">=3.12,<3.13"\n'
            '[tool.uv]\noverride-dependencies = ["genai-tk @ file:///x"]\n'
        )
        # Must print the warning and leave the override entry in place.
        ProjectScaffolder(project_dir, "Test Project")._ensure_package_installed()

        assert 'override-dependencies = ["genai-tk @ file:///x"]' in pyproject_path.read_text()

    def test_gitignore_merge_appends_runtime_block(self, project_dir: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        (project_dir / ".gitignore").write_text("# Python-generated files\n__pycache__/\n.venv\n")
        ProjectScaffolder(project_dir, "Test Project")._patch_gitignore()

        text = (project_dir / ".gitignore").read_text()
        assert "__pycache__/" in text  # original content kept
        assert "data/*" in text and "!data/sources/" in text

    def test_gitignore_merge_is_idempotent(self, project_dir: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        (project_dir / ".gitignore").write_text("data/*\n!data/sources/\n")
        scaffolder = ProjectScaffolder(project_dir, "Test Project")
        scaffolder._patch_gitignore()

        assert (project_dir / ".gitignore").read_text() == "data/*\n!data/sources/\n"
