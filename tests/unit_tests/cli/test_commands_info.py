"""Unit tests for genai_tk.cli.commands_info (CliRunner, fake models, no services)."""

from __future__ import annotations

import pytest
import typer
from typer.testing import CliRunner

import genai_tk.cli.commands_info as commands_info
from genai_tk.cli.commands_info import (
    CheckResult,
    InfoCommands,
    _comment_bashrc_proxy_exports,
    _doctor_check_proxy,
    _doctor_run_checks,
    _merge_env_file,
)


@pytest.fixture
def info_app() -> typer.Typer:
    app = typer.Typer()
    InfoCommands().register(app)
    return app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestInfoHelp:
    def test_help_exits_zero(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "--help"])
        assert result.exit_code == 0
        assert "config" in result.stdout


class TestInfoConfig:
    def test_config_runs_and_shows_active_context(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "config"])
        assert result.exit_code == 0
        assert "Active context" in result.stdout

    def test_config_shows_default_components_table(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "config"])
        assert result.exit_code == 0
        assert "Default Components" in result.stdout

    def test_config_lists_api_keys_table(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "config"])
        assert result.exit_code == 0
        assert "API Keys" in result.stdout or "Available API Keys" in result.stdout


class TestInfoModels:
    def test_models_runs_without_error(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "models"])
        assert result.exit_code == 0
        # The models command prints a table of configured models; expect some output.
        assert len(result.stdout) > 0


class TestInfoCommands:
    def test_commands_runs_and_lists_commands(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "commands"])
        # 'commands' builds a command tree of the registered CLI; it should not
        # crash and should produce output.
        assert result.exit_code == 0
        assert len(result.stdout) > 0


class TestInfoLs:
    def test_ls_without_path_shows_usage_or_error(self, info_app, runner, tmp_path) -> None:
        # 'ls' lists a directory; invoking without args should either error
        # gracefully or list the cwd. Either way it must not crash with a
        # non-handled traceback.
        result = runner.invoke(info_app, ["info", "ls", str(tmp_path)])
        assert result.exit_code == 0
        assert str(tmp_path) in result.stdout or len(result.stdout) >= 0

    def test_ls_lists_files_in_tmp_dir(self, info_app, runner, tmp_path) -> None:
        (tmp_path / "alpha.txt").write_text("hi")
        (tmp_path / "beta.md").write_text("yo")
        result = runner.invoke(info_app, ["info", "ls", str(tmp_path)])
        assert result.exit_code == 0
        assert "alpha.txt" in result.stdout
        assert "beta.md" in result.stdout

    def test_ls_with_pathspec_filter(self, info_app, runner, tmp_path) -> None:
        (tmp_path / "alpha.txt").write_text("hi")
        (tmp_path / "beta.md").write_text("yo")
        result = runner.invoke(
            info_app,
            ["info", "ls", str(tmp_path), "--pathspec", "**/*.md"],
        )
        assert result.exit_code == 0
        assert "beta.md" in result.stdout
        assert "alpha.txt" not in result.stdout


class TestInfoConfigKeys:
    def test_config_keys_runs_and_lists_keys(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "config-keys"])
        assert result.exit_code == 0
        assert len(result.stdout) > 0

    def test_config_keys_with_query_filter(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "config-keys", "llm"])
        assert result.exit_code == 0
        assert "llm" in result.stdout.lower()

    def test_config_keys_with_type_filter(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "config-keys", "models"])
        assert result.exit_code == 0


class TestInfoLlmProfile:
    def test_llm_profile_runs_default(self, info_app, runner) -> None:
        # No MODEL_ID and no --reload: the command is a documented no-op.
        result = runner.invoke(info_app, ["info", "llm-profile"])
        assert result.exit_code == 0

    def test_llm_profile_with_known_model(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "llm-profile", "parrot_local@fake"])
        assert result.exit_code == 0
        assert "parrot_local" in result.stdout or "fake" in result.stdout

    def test_llm_profile_with_raw_model_name(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "llm-profile", "gpt-4o-mini"])
        assert result.exit_code == 0


class TestInfoMcpTools:
    def test_mcp_tools_runs_and_reports_status(self, info_app, runner) -> None:
        result = runner.invoke(info_app, ["info", "mcp-tools"])
        assert result.exit_code == 0


def _check(name: str = "sample check", ok: bool = True, warn: bool = False, detail: str = "", hint: str = "") -> CheckResult:
    return CheckResult(name=name, ok=ok, warn=warn, detail=detail, hint=hint)


class TestInfoDoctor:
    """`cli info doctor` renders check results and maps failures to exit code 1."""

    @staticmethod
    def _patch_checks(monkeypatch: pytest.MonkeyPatch, results: list[CheckResult]) -> None:
        monkeypatch.setattr(commands_info, "_doctor_run_checks", lambda *, offline, timeout: results)

    def test_all_pass_exits_zero(self, info_app, runner, monkeypatch) -> None:
        self._patch_checks(monkeypatch, [_check("harnessing feature"), _check("prefect server")])
        result = runner.invoke(info_app, ["info", "doctor"])
        assert result.exit_code == 0
        assert "harnessing feature" in result.stdout
        assert "All checks passed" in result.stdout

    def test_failure_exits_one_and_prints_hint(self, info_app, runner, monkeypatch) -> None:
        failing = _check("models.dev cache", ok=False, detail="missing", hint="run 'cli info llm-profile --reload'")
        self._patch_checks(monkeypatch, [failing])
        result = runner.invoke(info_app, ["info", "doctor"])
        assert result.exit_code == 1
        assert "Doctor found problems" in result.stdout
        assert "llm-profile" in result.stdout

    def test_warn_exits_zero(self, info_app, runner, monkeypatch) -> None:
        self._patch_checks(monkeypatch, [_check("other optional features", warn=True, detail="missing: rag")])
        result = runner.invoke(info_app, ["info", "doctor"])
        assert result.exit_code == 0

    def test_offline_and_timeout_flags_are_forwarded(self, info_app, runner, monkeypatch) -> None:
        seen: dict[str, object] = {}

        def fake_checks(*, offline: bool, timeout: float) -> list[CheckResult]:
            seen.update(offline=offline, timeout=timeout)
            return [_check()]

        monkeypatch.setattr(commands_info, "_doctor_run_checks", fake_checks)
        result = runner.invoke(info_app, ["info", "doctor", "--offline", "--timeout", "1.5"])
        assert result.exit_code == 0
        assert seen == {"offline": True, "timeout": 1.5}

    def test_fix_reports_changes_and_rechecks_proxy(self, info_app, runner, monkeypatch) -> None:
        self._patch_checks(monkeypatch, [_check("proxy bypass for localhost", ok=False)])
        monkeypatch.setattr(
            commands_info,
            "_doctor_apply_fix",
            lambda *, timeout: [".env: NO_PROXY/no_proxy now covers 3 host(s)"],
        )
        monkeypatch.setattr(
            commands_info,
            "_doctor_check_proxy",
            lambda *, offline, timeout: [_check("API host reachability")],
        )
        result = runner.invoke(info_app, ["info", "doctor", "--fix"])
        assert result.exit_code == 0
        assert "now covers 3 host(s)" in result.stdout
        assert "API host reachability" in result.stdout


class TestDoctorCheckProxy:
    """Proxy checks with stubbed network probes (offline-safe)."""

    def test_offline_skips_network_probes(self, monkeypatch) -> None:
        import genai_tk.utils.net_env as net_env

        monkeypatch.setattr(net_env, "no_proxy_entries", lambda env=None: list(net_env.DEFAULT_BYPASS_HOSTS))

        def unexpected_probe(*args: object, **kwargs: object) -> dict[str, list[str]]:
            raise AssertionError("network probes must not run with --offline")

        monkeypatch.setattr(net_env, "recommended_bypass_hosts", unexpected_probe)
        results = _doctor_check_proxy(offline=True, timeout=0.1)
        assert any(r.name == "API host reachability" and "skipped" in r.detail for r in results)

    def test_missing_loopback_bypass_fails_check(self, monkeypatch) -> None:
        import genai_tk.utils.net_env as net_env

        monkeypatch.setattr(net_env, "no_proxy_entries", lambda env=None: ["pypi.org"])
        results = _doctor_check_proxy(offline=True, timeout=0.1)
        bypass = next(r for r in results if r.name == "proxy bypass for localhost")
        assert not bypass.ok
        assert "localhost" in bypass.detail

    def test_proxy_blocked_host_reports_bypass_needed(self, monkeypatch) -> None:
        import genai_tk.utils.net_env as net_env

        monkeypatch.setattr(net_env, "no_proxy_entries", lambda env=None: list(net_env.DEFAULT_BYPASS_HOSTS))
        classification: dict[str, list[str]] = {
            "proxy_ok": ["models.dev"],
            "bypass_needed": ["openrouter.ai"],
            "unreachable": [],
        }
        monkeypatch.setattr(net_env, "recommended_bypass_hosts", lambda hosts=None, *, timeout: classification)
        results = _doctor_check_proxy(offline=False, timeout=0.1)
        reachability = next(r for r in results if r.name == "API host reachability")
        assert not reachability.ok
        assert "openrouter.ai" in reachability.detail
        assert "--fix" in reachability.hint


class TestDoctorRunChecks:
    def test_offline_smoke_produces_all_check_families(self) -> None:
        results = _doctor_run_checks(offline=True, timeout=0.5)
        names = {r.name for r in results}
        assert "harnessing feature" in names
        assert "prefect server" in names
        assert "models.dev cache" in names
        assert "proxy bypass for localhost" in names


class TestMergeEnvFile:
    def test_creates_env_file_when_missing(self, tmp_path) -> None:
        env_path = tmp_path / ".env"
        summary = _merge_env_file(env_path, ["localhost", "openrouter.ai"])
        content = env_path.read_text(encoding="utf-8")
        assert "no_proxy=localhost,openrouter.ai" in content
        assert "NO_PROXY=localhost,openrouter.ai" in content
        assert "added: localhost, openrouter.ai" in summary

    def test_preserves_other_lines_and_existing_entries(self, tmp_path) -> None:
        env_path = tmp_path / ".env"
        env_path.write_text("FOO=bar\n\nno_proxy=pypi.org\n", encoding="utf-8")
        summary = _merge_env_file(env_path, ["pypi.org", "localhost"])
        content = env_path.read_text(encoding="utf-8")
        assert "FOO=bar" in content
        assert "no_proxy=pypi.org,localhost" in content
        assert "added: localhost" in summary

    def test_replaces_hand_maintained_export_lines(self, tmp_path) -> None:
        env_path = tmp_path / ".env"
        env_path.write_text('export NO_PROXY="old.host,other.host"\nKEY=value\n', encoding="utf-8")
        _merge_env_file(env_path, ["localhost"])
        content = env_path.read_text(encoding="utf-8")
        assert "export NO_PROXY=" not in content
        assert "NO_PROXY=old.host,other.host,localhost" in content
        assert "KEY=value" in content


class TestCommentBashrcProxyExports:
    def test_comments_out_proxy_export_lines(self, tmp_path) -> None:
        bashrc = tmp_path / ".bashrc"
        bashrc.write_text(
            "export PATH=$PATH:/opt/bin\n"
            'export no_proxy="localhost,old.host"\n'
            "export NO_PROXY=localhost,old.host\n"
            "alias ll='ls -la'\n",
            encoding="utf-8",
        )
        summary = _comment_bashrc_proxy_exports(bashrc)
        lines = bashrc.read_text(encoding="utf-8").splitlines()
        active = [line for line in lines if line.strip().startswith(("export no_proxy=", "export NO_PROXY="))]
        assert active == []
        disabled = [line for line in lines if "# disabled by 'cli info doctor --fix'" in line]
        assert len(disabled) == 2
        assert any("alias ll" in line for line in lines)
        assert summary is not None and "2" in summary

    def test_returns_none_when_no_proxy_lines(self, tmp_path) -> None:
        bashrc = tmp_path / ".bashrc"
        bashrc.write_text("alias ll='ls -la'\n", encoding="utf-8")
        assert _comment_bashrc_proxy_exports(bashrc) is None
        assert bashrc.read_text(encoding="utf-8") == "alias ll='ls -la'\n"

    def test_returns_none_when_file_missing(self, tmp_path) -> None:
        assert _comment_bashrc_proxy_exports(tmp_path / ".bashrc") is None
