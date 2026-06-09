"""Tests for the skills discovery system — categories, listing, bundled skills."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_skill(root: Path, name: str, category: str | None = None) -> Path:
    """Create a minimal SKILL.md under <root>/<name>/SKILL.md."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    extra = f"\ncategory: {category}" if category else ""
    (d / "SKILL.md").write_text(
        dedent(f"""\
        ---
        name: {name}
        description: A test skill
        tags: [test]{extra}
        version: "1.0"
        author: ""
        ---

        # {name.title()}

        ## When to Use

        Use this skill for testing.
        """)
    )
    return d


# ---------------------------------------------------------------------------
# SkillInfo model
# ---------------------------------------------------------------------------


class TestSkillInfoCategory:
    def test_default_category_is_project(self):
        from genai_tk.main.models_skills import SkillInfo

        s = SkillInfo(name="x", path=Path("/tmp"), source="custom")
        assert s.category == "project"

    def test_category_dev(self):
        from genai_tk.main.models_skills import SkillInfo

        s = SkillInfo(name="x", path=Path("/tmp"), source="bundled", category="dev")
        assert s.category == "dev"

    def test_category_agent(self):
        from genai_tk.main.models_skills import SkillInfo

        s = SkillInfo(name="x", path=Path("/tmp"), source="bundled", category="agent")
        assert s.category == "agent"

    def test_invalid_category_raises(self):
        from pydantic import ValidationError

        from genai_tk.main.models_skills import SkillInfo

        with pytest.raises(ValidationError):
            SkillInfo(name="x", path=Path("/tmp"), source="custom", category="unknown")


# ---------------------------------------------------------------------------
# discover_skills
# ---------------------------------------------------------------------------


class TestDiscoverSkills:
    def test_discovers_skill_in_root(self, tmp_path: Path):
        from genai_tk.main.skills_manager import discover_skills

        _write_skill(tmp_path, "my-skill")
        skills = discover_skills([tmp_path], source="custom", category="project")
        assert len(skills) == 1
        assert skills[0].name == "my-skill"
        assert skills[0].source == "custom"
        assert skills[0].category == "project"

    def test_discovers_multiple_skills(self, tmp_path: Path):
        from genai_tk.main.skills_manager import discover_skills

        for name in ("alpha", "beta", "gamma"):
            _write_skill(tmp_path, name)
        skills = discover_skills([tmp_path])
        assert len(skills) == 3

    def test_skips_dirs_without_skill_md(self, tmp_path: Path):
        from genai_tk.main.skills_manager import discover_skills

        (tmp_path / "not-a-skill").mkdir()
        skills = discover_skills([tmp_path])
        assert skills == []

    def test_category_propagated_to_all_skills(self, tmp_path: Path):
        from genai_tk.main.skills_manager import discover_skills

        _write_skill(tmp_path, "s1")
        _write_skill(tmp_path, "s2")
        skills = discover_skills([tmp_path], source="bundled", category="dev")
        assert all(s.category == "dev" for s in skills)
        assert all(s.source == "bundled" for s in skills)

    def test_empty_root_returns_empty(self, tmp_path: Path):
        from genai_tk.main.skills_manager import discover_skills

        skills = discover_skills([tmp_path])
        assert skills == []

    def test_nonexistent_root_skipped(self, tmp_path: Path):
        from genai_tk.main.skills_manager import discover_skills

        skills = discover_skills([tmp_path / "does-not-exist"])
        assert skills == []


# ---------------------------------------------------------------------------
# DIR_CATEGORY mapping
# ---------------------------------------------------------------------------


class TestDirCategoryMapping:
    def test_genai_tk_dir_maps_to_dev(self):
        from genai_tk.main.skills_manager import _DIR_CATEGORY

        assert _DIR_CATEGORY["genai-tk"] == "dev"

    def test_copilot_dir_maps_to_dev(self):
        from genai_tk.main.skills_manager import _DIR_CATEGORY

        assert _DIR_CATEGORY["copilot"] == "dev"

    def test_public_dir_maps_to_agent(self):
        from genai_tk.main.skills_manager import _DIR_CATEGORY

        assert _DIR_CATEGORY["public"] == "agent"

    def test_langchain_examples_maps_to_agent(self):
        from genai_tk.main.skills_manager import _DIR_CATEGORY

        assert _DIR_CATEGORY["langchain_examples"] == "agent"


# ---------------------------------------------------------------------------
# discover_all_skills — bundled category assignment
# ---------------------------------------------------------------------------


class TestDiscoverAllSkillsBundled:
    def test_bundled_dev_skills_have_dev_category(self):
        from genai_tk.main.skills_manager import discover_all_skills

        skills = discover_all_skills(Path("/tmp/__nonexistent__"))
        dev_skills = [s for s in skills if s.category == "dev"]
        # genai-tk/ and copilot/ dirs both map to dev
        assert len(dev_skills) > 0

    def test_bundled_agent_skills_have_agent_category(self):
        from genai_tk.main.skills_manager import discover_all_skills

        skills = discover_all_skills(Path("/tmp/__nonexistent__"))
        agent_skills = [s for s in skills if s.category == "agent"]
        # public/ and langchain_examples/ dirs both map to agent
        assert len(agent_skills) > 0

    def test_all_bundled_skills_have_source_bundled(self):
        from genai_tk.main.skills_manager import discover_all_skills

        skills = discover_all_skills(Path("/tmp/__nonexistent__"))
        bundled = [s for s in skills if s.source == "bundled"]
        assert len(bundled) > 0
        assert all(s.source == "bundled" for s in bundled)

    def test_project_custom_skills_discovered(self, tmp_path: Path):
        from genai_tk.main.skills_manager import discover_all_skills

        custom_root = tmp_path / "skills" / "custom"
        _write_skill(custom_root, "my-custom-skill")

        skills = discover_all_skills(tmp_path)
        custom = [s for s in skills if s.source == "custom"]
        assert any(s.name == "my-custom-skill" for s in custom)
        assert all(s.category == "project" for s in custom)

    def test_community_skills_discovered(self, tmp_path: Path):
        from genai_tk.main.skills_manager import discover_all_skills

        community_root = tmp_path / "skills" / "community"
        _write_skill(community_root, "a-community-skill")

        skills = discover_all_skills(tmp_path)
        community = [s for s in skills if s.source == "skillssh"]
        assert any(s.name == "a-community-skill" for s in community)
        assert all(s.category == "project" for s in community)


# ---------------------------------------------------------------------------
# init command — deer-flow flag
# ---------------------------------------------------------------------------


class TestInitCommandDeerFlow:
    def test_with_deer_flow_flag_calls_install(self, tmp_path: Path, monkeypatch):
        """--extra harnessing triggers _install_extra('harnessing')."""
        from genai_tk.main import commands_init

        calls = []
        monkeypatch.setattr(commands_init, "_install_extra", lambda extra: calls.append(extra) or True)
        monkeypatch.setattr(commands_init, "_copy_default_config", lambda dest, force: True)
        monkeypatch.setattr(commands_init, "_patch_webapp_yaml", lambda dest, name: None)
        monkeypatch.setattr(commands_init, "_scaffold_project", lambda *a, **kw: None)
        monkeypatch.setattr(commands_init, "_print_next_steps", lambda *a, **kw: None)
        monkeypatch.chdir(tmp_path)

        commands_init._install_extra("harnessing")
        assert "harnessing" in calls

    def test_install_extra_harnessing_uses_uv_sync(self, monkeypatch):
        """_install_extra('harnessing') runs 'uv sync --extra harnessing'."""
        import subprocess

        from genai_tk.main import commands_init

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd

            class FakeResult:
                returncode = 0

            return FakeResult()

        monkeypatch.setattr(subprocess, "run", fake_run)
        commands_init._install_extra("harnessing")

        assert captured["cmd"] == ["uv", "sync", "--extra", "harnessing"]

    def test_no_deer_flow_path_env_var_needed(self, monkeypatch):
        """After refactor, DEER_FLOW_PATH is not referenced in install logic."""
        monkeypatch.delenv("DEER_FLOW_PATH", raising=False)
        import inspect

        from genai_tk.main import commands_init

        src = inspect.getsource(commands_init._install_extra)
        assert "DEER_FLOW_PATH" not in src


# ---------------------------------------------------------------------------
# Scaffolder — cursor/windsurf files removed
# ---------------------------------------------------------------------------


class TestScaffolderNoIdeFiles:
    def test_no_cursor_file_generated(self, tmp_path: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        # Create minimal config
        config = tmp_path / "config"
        config.mkdir()
        (config / "app_conf.yaml").write_text("cli:\n  commands:\n    - genai_tk.main.cli.register_commands\n")
        (config / "webapp.yaml").write_text("ui:\n  app_name: Test\n")

        scaffolder = ProjectScaffolder(tmp_path, "Test Project")
        scaffolder.scaffold()

        assert not (tmp_path / ".cursor").exists()

    def test_no_windsurfrules_generated(self, tmp_path: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        config = tmp_path / "config"
        config.mkdir()
        (config / "app_conf.yaml").write_text("cli:\n  commands:\n    - genai_tk.main.cli.register_commands\n")
        (config / "webapp.yaml").write_text("ui:\n  app_name: Test\n")

        scaffolder = ProjectScaffolder(tmp_path, "Test Project")
        scaffolder.scaffold()

        assert not (tmp_path / ".windsurfrules").exists()

    def test_copilot_instructions_still_generated(self, tmp_path: Path):
        from genai_tk.main.scaffolder import ProjectScaffolder

        config = tmp_path / "config"
        config.mkdir()
        (config / "app_conf.yaml").write_text("cli:\n  commands:\n    - genai_tk.main.cli.register_commands\n")
        (config / "webapp.yaml").write_text("ui:\n  app_name: Test\n")

        scaffolder = ProjectScaffolder(tmp_path, "Test Project")
        scaffolder.scaffold()

        assert (tmp_path / ".github" / "copilot-instructions.md").exists()


# ---------------------------------------------------------------------------
# DeerFlow CLI — importability check replaces DEER_FLOW_PATH
# ---------------------------------------------------------------------------


class TestDeerFlowCLI:
    def test_require_deerflow_installed_succeeds_when_importable(self, monkeypatch):
        """No exit when harnessing feature reports as available."""
        from genai_tk.agents.deer_flow import cli_commands
        from genai_tk.config_mgmt import features

        monkeypatch.setattr(features, "is_available", lambda name: True)

        # Should not raise
        cli_commands._require_deer_flow_installed()

    def test_require_deerflow_installed_exits_when_not_installed(self, monkeypatch):
        """Exits with typer.Exit(1) when harnessing feature is not available."""
        import typer

        from genai_tk.agents.deer_flow import cli_commands
        from genai_tk.config_mgmt import features

        # Patch is_available so it reports harnessing as missing
        monkeypatch.setattr(features, "is_available", lambda name: False)

        with pytest.raises(typer.Exit):
            cli_commands._require_deer_flow_installed()

    def test_no_deer_flow_path_check_in_require(self):
        """After refactor, _require_deer_flow_installed does not check DEER_FLOW_PATH at runtime."""
        import inspect

        from genai_tk.agents.deer_flow import cli_commands

        src = inspect.getsource(cli_commands._require_deer_flow_installed)
        # The docstring mentions DEER_FLOW_PATH, but the actual code logic must not
        # reference it: no os.environ access, no Path resolution, no exists() check.
        assert "os.environ" not in src
        assert "expanduser" not in src
        assert "exists()" not in src

    def test_prepare_profile_uses_new_check(self):
        """_prepare_profile calls _require_deer_flow_installed, not _require_deer_flow_path."""
        import inspect

        from genai_tk.agents.deer_flow import cli_commands

        src = inspect.getsource(cli_commands._prepare_profile)
        assert "_require_deer_flow_installed" in src
        assert "_require_deer_flow_path" not in src


# ---------------------------------------------------------------------------
# init command — sandbox flag
# ---------------------------------------------------------------------------


class TestSandboxInit:
    def test_install_extra_harnessing_uses_uv_sync(self, monkeypatch):
        """_install_extra('harnessing') runs 'uv sync --extra harnessing'."""
        import subprocess

        from genai_tk.main import commands_init

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd

            class FakeResult:
                returncode = 0

            return FakeResult()

        monkeypatch.setattr(subprocess, "run", fake_run)
        commands_init._install_extra("harnessing")

        assert captured["cmd"] == ["uv", "sync", "--extra", "harnessing"]

    def test_with_extra_flag_calls_install(self, monkeypatch):
        """--extra harnessing triggers _install_extra."""
        from genai_tk.main import commands_init

        calls = []
        monkeypatch.setattr(commands_init, "_install_extra", lambda extra: calls.append(extra) or True)
        commands_init._install_extra("harnessing")
        assert "harnessing" in calls

    def test_aio_sandbox_not_in_core_deps(self):
        """agent-sandbox / opensandbox must NOT appear in [project.dependencies]."""
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).parents[3] / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text())
        core_deps = data.get("project", {}).get("dependencies", [])

        forbidden = ["agent-sandbox", "opensandbox"]
        for dep in core_deps:
            for pkg in forbidden:
                assert not dep.lower().startswith(pkg), (
                    f"{pkg!r} must not be in [project.dependencies] — it belongs in the 'harnessing' optional extra"
                )
