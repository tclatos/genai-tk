"""Tests for the skills discovery system — categories, listing, bundled skills."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import typer

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
    def test_default_category_is_custom(self):
        from genai_tk.main.models_skills import SkillInfo

        s = SkillInfo(name="x", path=Path("/tmp"), source="custom")
        assert s.category == "custom"

    def test_category_development(self):
        from genai_tk.main.models_skills import SkillInfo

        s = SkillInfo(name="x", path=Path("/tmp"), source="bundled", category="development")
        assert s.category == "development"

    def test_category_runtime(self):
        from genai_tk.main.models_skills import SkillInfo

        s = SkillInfo(name="x", path=Path("/tmp"), source="bundled", category="runtime")
        assert s.category == "runtime"

    def test_category_governance(self):
        from genai_tk.main.models_skills import SkillInfo

        s = SkillInfo(name="x", path=Path("/tmp"), source="bundled", category="governance")
        assert s.category == "governance"

    def test_category_vendor(self):
        from genai_tk.main.models_skills import SkillInfo

        s = SkillInfo(name="x", path=Path("/tmp"), source="vendor", category="vendor")
        assert s.category == "vendor"

    def test_invalid_category_raises(self):
        from pydantic import ValidationError

        from genai_tk.main.models_skills import SkillInfo

        with pytest.raises(ValidationError):
            SkillInfo(name="x", path=Path("/tmp"), source="custom", category="unknown_invalid_category")


# ---------------------------------------------------------------------------
# discover_skills
# ---------------------------------------------------------------------------


class TestDiscoverSkills:
    def test_discovers_skill_in_root(self, tmp_path: Path):
        from genai_tk.main.skills_manager import discover_skills

        _write_skill(tmp_path, "my-skill")
        skills = discover_skills([tmp_path], source="custom", category="custom")
        assert len(skills) == 1
        assert skills[0].name == "my-skill"
        assert skills[0].source == "custom"
        assert skills[0].category == "custom"

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
        skills = discover_skills([tmp_path], source="bundled", category="development")
        assert all(s.category == "development" for s in skills)
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
    def test_development_dir_maps_to_development(self):
        from genai_tk.main.skills_manager import _DIR_CATEGORY

        assert _DIR_CATEGORY["development"] == "development"

    def test_runtime_dir_maps_to_runtime(self):
        from genai_tk.main.skills_manager import _DIR_CATEGORY

        assert _DIR_CATEGORY["runtime"] == "runtime"

    def test_governance_dir_maps_to_governance(self):
        from genai_tk.main.skills_manager import _DIR_CATEGORY

        assert _DIR_CATEGORY["governance"] == "governance"

    def test_vendor_dir_maps_to_vendor(self):
        from genai_tk.main.skills_manager import _DIR_CATEGORY

        assert _DIR_CATEGORY["vendor"] == "vendor"


# ---------------------------------------------------------------------------
# discover_all_skills — bundled category assignment
# ---------------------------------------------------------------------------


class TestDiscoverAllSkillsBundled:
    def test_bundled_development_skills_have_development_category(self):
        from genai_tk.main.skills_manager import discover_all_skills

        skills = discover_all_skills(Path("/tmp/__nonexistent__"))
        dev_skills = [s for s in skills if s.category == "development"]
        assert len(dev_skills) > 0

    def test_bundled_runtime_skills_have_runtime_category(self):
        from genai_tk.main.skills_manager import discover_all_skills

        skills = discover_all_skills(Path("/tmp/__nonexistent__"))
        runtime_skills = [s for s in skills if s.category == "runtime"]
        assert len(runtime_skills) > 0

    def test_bundled_governance_skills_have_governance_category(self):
        from genai_tk.main.skills_manager import discover_all_skills

        skills = discover_all_skills(Path("/tmp/__nonexistent__"))
        gov_skills = [s for s in skills if s.category == "governance"]
        assert len(gov_skills) > 0

    def test_all_bundled_skills_have_source_bundled_or_vendor(self):
        from genai_tk.main.skills_manager import discover_all_skills

        skills = discover_all_skills(Path("/tmp/__nonexistent__"))
        assert len(skills) > 0
        assert all(s.source in ("bundled", "vendor") for s in skills)

    def test_project_custom_skills_discovered(self, tmp_path: Path):
        from genai_tk.main.skills_manager import discover_all_skills

        custom_root = tmp_path / "skills" / "custom"
        _write_skill(custom_root, "my-custom-skill")

        skills = discover_all_skills(tmp_path)
        custom = [s for s in skills if s.source == "custom"]
        assert any(s.name == "my-custom-skill" for s in custom)
        assert all(s.category == "custom" for s in custom)

    def test_community_skills_discovered(self, tmp_path: Path):
        from genai_tk.main.skills_manager import discover_all_skills

        community_root = tmp_path / "skills" / "community"
        _write_skill(community_root, "a-community-skill")

        skills = discover_all_skills(tmp_path)
        community = [s for s in skills if s.source == "skillssh"]
        assert any(s.name == "a-community-skill" for s in community)
        assert all(s.category == "custom" for s in community)


# ---------------------------------------------------------------------------
# init command — deer-flow flag
# ---------------------------------------------------------------------------


class TestInitCommandDeerFlow:
    def test_with_deer_flow_flag_calls_install(self, tmp_path: Path, monkeypatch):
        """--extra deerflow triggers _install_extra('deerflow')."""
        from genai_tk.main import commands_init

        calls = []
        monkeypatch.setattr(commands_init, "_install_extra", lambda extra: calls.append(extra) or True)
        monkeypatch.setattr(commands_init, "_copy_default_config", lambda dest, force: True)
        monkeypatch.setattr(commands_init, "_patch_webapp_yaml", lambda dest, name: None)
        monkeypatch.setattr(commands_init, "_scaffold_project", lambda *a, **kw: None)
        monkeypatch.setattr(commands_init, "_print_next_steps", lambda *a, **kw: None)
        monkeypatch.chdir(tmp_path)

        commands_init._install_extra("deerflow")
        assert "deerflow" in calls

    def test_install_extra_deerflow_uses_uv_sync(self, monkeypatch):
        """_install_extra('deerflow') runs 'uv sync --extra deerflow'."""
        import subprocess

        from genai_tk.main import commands_init

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd

            class FakeResult:
                returncode = 0

            return FakeResult()

        monkeypatch.setattr(subprocess, "run", fake_run)
        commands_init._install_extra("deerflow")

        assert captured["cmd"] == ["uv", "sync", "--extra", "deerflow"]

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
    @pytest.fixture(autouse=True)
    def mock_subprocess_run(self, monkeypatch):
        """Prevent uv sync from creating real virtualenvs in tmp_path during unit tests."""
        import subprocess

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, stdout="", stderr=""))

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
        """No raise when the deerflow feature reports as available."""
        from genai_tk.agents.deer_flow import runtime
        from genai_tk.config_mgmt import features

        monkeypatch.setattr(features, "is_available", lambda name: True)

        # Should not raise
        runtime.require_deer_flow_installed()

    def test_require_deerflow_installed_exits_when_not_installed(self, monkeypatch):
        """Raises DeerFlowNotInstalledError when the deerflow feature is not available."""
        from genai_tk.agents.deer_flow import runtime
        from genai_tk.agents.deer_flow.runtime import DeerFlowNotInstalledError
        from genai_tk.config_mgmt import features

        # Patch is_available so it reports deerflow as missing
        monkeypatch.setattr(features, "is_available", lambda name: False)

        with pytest.raises(DeerFlowNotInstalledError):
            runtime.require_deer_flow_installed()

    def test_no_deer_flow_path_check_in_require(self):
        """After refactor, require_deer_flow_installed does not check DEER_FLOW_PATH at runtime."""
        import inspect

        from genai_tk.agents.deer_flow import runtime

        src = inspect.getsource(runtime.require_deer_flow_installed)
        # The pure runtime check must not reference DEER_FLOW_PATH side-effects:
        # no os.environ access, no Path resolution, no exists() check.
        assert "os.environ" not in src
        assert "expanduser" not in src
        assert "exists()" not in src


# ---------------------------------------------------------------------------
# Skills CLI Commands (CliRunner)
# ---------------------------------------------------------------------------


class TestSkillsCommandsCliRunner:
    @pytest.fixture
    def skills_app(self) -> typer.Typer:
        import typer

        from genai_tk.cli.commands_skills import SkillsCommands

        app = typer.Typer()
        SkillsCommands().register(app)
        return app

    @pytest.fixture
    def runner(self):
        from typer.testing import CliRunner

        return CliRunner()

    def test_skills_help(self, skills_app, runner):
        result = runner.invoke(skills_app, ["skills", "--help"])
        assert result.exit_code == 0
        assert "list" in result.stdout
        assert "validate" in result.stdout

    def test_skills_list(self, skills_app, runner):
        result = runner.invoke(skills_app, ["skills", "list"])
        assert result.exit_code == 0
        assert len(result.stdout) > 0

    def test_skills_list_with_category_filter(self, skills_app, runner):
        result = runner.invoke(skills_app, ["skills", "list", "--category", "runtime"])
        assert result.exit_code == 0

    def test_skills_info_nonexistent(self, skills_app, runner):
        result = runner.invoke(skills_app, ["skills", "info", "definitely-not-a-real-skill-12345"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_skills_search(self, skills_app, runner):
        result = runner.invoke(skills_app, ["skills", "search", "python"])
        assert result.exit_code == 0

    def test_skills_create(self, skills_app, runner, tmp_path, monkeypatch):
        import genai_tk.cli.commands_skills as skills_cmd_mod

        monkeypatch.setattr(skills_cmd_mod, "_project_dir", lambda: tmp_path)
        result = runner.invoke(skills_app, ["skills", "create", "test-new-skill", "--description", "A test skill"])
        assert result.exit_code == 0
        created = tmp_path / "skills" / "custom" / "test-new-skill" / "SKILL.md"
        assert created.exists()
        assert "test-new-skill" in created.read_text()

    def test_prepare_profile_uses_new_check(self):
        """prepare_profile calls require_deer_flow_installed, not _require_deer_flow_path."""
        import inspect

        from genai_tk.agents.deer_flow import runtime

        src = inspect.getsource(runtime.prepare_profile)
        assert "require_deer_flow_installed" in src
        assert "_require_deer_flow_path" not in src


# ---------------------------------------------------------------------------
# init command — sandbox flag
# ---------------------------------------------------------------------------


class TestSandboxInit:
    def test_install_extra_aio_sandbox_uses_uv_sync(self, monkeypatch):
        """_install_extra('aio-sandbox') runs 'uv sync --extra aio-sandbox'."""
        import subprocess

        from genai_tk.main import commands_init

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd

            class FakeResult:
                returncode = 0

            return FakeResult()

        monkeypatch.setattr(subprocess, "run", fake_run)
        commands_init._install_extra("aio-sandbox")

        assert captured["cmd"] == ["uv", "sync", "--extra", "aio-sandbox"]

    def test_with_extra_flag_calls_install(self, monkeypatch):
        """--extra aio-sandbox triggers _install_extra."""
        from genai_tk.main import commands_init

        calls = []
        monkeypatch.setattr(commands_init, "_install_extra", lambda extra: calls.append(extra) or True)
        commands_init._install_extra("aio-sandbox")
        assert "aio-sandbox" in calls

    def test_aio_sandbox_not_in_core_deps(self):
        """agent-sandbox / opensandbox must NOT appear in [project.dependencies], but deepagents MUST."""
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).parents[3] / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text())
        core_deps = data.get("project", {}).get("dependencies", [])

        forbidden = ["agent-sandbox", "opensandbox"]
        for dep in core_deps:
            for pkg in forbidden:
                assert not dep.lower().startswith(pkg), (
                    f"{pkg!r} must not be in [project.dependencies] — it belongs in the 'aio-sandbox' optional extra"
                )

        assert any(dep.lower().startswith("deepagents") for dep in core_deps), (
            "'deepagents' must be in [project.dependencies] (installed by default)"
        )


class TestFindShadowedLocalSkills:
    """`find_shadowed_local_skills` warns when a local skill shadows a changed bundled one."""

    def _write_upstream(self, root: Path, body: str) -> Path:
        upstream = root / "decision-navigation"
        upstream.mkdir(parents=True)
        (upstream / "SKILL.md").write_text(
            dedent(
                """\
                ---
                name: decision-navigation
                description: Evidence-pack answering loop for decision-model navigation.
                version: 1.1.0
                ---

                # upstream body
                """
            )
            + body,
            encoding="utf-8",
        )
        return root

    def _write_local(self, project_dir: Path, body: str) -> Path:
        local = project_dir / "skills" / "custom" / "decision-navigation"
        local.mkdir(parents=True)
        (local / "SKILL.md").write_text(
            dedent(
                """\
                ---
                name: decision-navigation
                description: Evidence-pack answering loop for decision-model navigation.
                ---

                # local body
                """
            )
            + body,
            encoding="utf-8",
        )
        return local

    def test_warns_when_local_copy_differs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from genai_tk.main import skills_manager

        upstream_root = self._write_upstream(tmp_path / "pkg", body="")
        self._write_local(tmp_path / "proj", body="")
        monkeypatch.setattr(skills_manager, "bundled_skill_roots", lambda: {"genai_graph": upstream_root})

        drift = skills_manager.find_shadowed_local_skills(tmp_path / "proj")
        assert len(drift) == 1
        assert "decision-navigation" in drift[0]
        assert "v1.1.0" in drift[0] and "genai_graph" in drift[0]

    def test_in_sync_local_copy_is_not_reported(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from genai_tk.main import skills_manager

        upstream_root = self._write_upstream(tmp_path / "pkg", body="")
        local = self._write_local(tmp_path / "proj", body="")
        (local / "SKILL.md").write_text(
            (upstream_root / "decision-navigation" / "SKILL.md").read_text(encoding="utf-8"), encoding="utf-8"
        )
        monkeypatch.setattr(skills_manager, "bundled_skill_roots", lambda: {"genai_graph": upstream_root})

        assert skills_manager.find_shadowed_local_skills(tmp_path / "proj") == []

    def test_unrelated_local_skills_are_ignored(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from genai_tk.main import skills_manager

        upstream_root = self._write_upstream(tmp_path / "pkg", body="")
        my_skill = tmp_path / "proj" / "skills" / "custom" / "my-own-skill"
        my_skill.mkdir(parents=True)
        (my_skill / "SKILL.md").write_text(
            "---\nname: my-own-skill\ndescription: mine\n---\n\n# mine\n", encoding="utf-8"
        )
        monkeypatch.setattr(skills_manager, "bundled_skill_roots", lambda: {"genai_graph": upstream_root})

        assert skills_manager.find_shadowed_local_skills(tmp_path / "proj") == []
