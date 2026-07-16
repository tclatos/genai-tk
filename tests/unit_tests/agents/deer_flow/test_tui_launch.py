"""Unit tests for the DeerFlow TUI launcher (genai_tk.agents.deer_flow.tui).

No real model, no live Textual run. ``DeerFlowClient`` and the ``deerflow.tui``
surface are replaced with in-process fakes via ``sys.modules`` injection (same
pattern as test_client.py), and ``run_deerflow_tui``'s runtime dependencies
(prepare_profile, the checkpointer, the feature gate) are monkeypatched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from genai_tk.agents.commands_agents import AgentCommands
from genai_tk.agents.deer_flow.profile import DeerFlowProfile
from genai_tk.agents.deer_flow.tui import build_tui_client, run_deerflow_tui

# ---------------------------------------------------------------------------
# Fake DeerFlowClient classes (with explicit __init__ signatures so
# inspect.signature() in build_tui_client sees exactly the params we want)
# ---------------------------------------------------------------------------


class _FakeDeerFlowClient:
    """Full-signature fake matching the modern DeerFlowClient constructor."""

    def __init__(
        self,
        config_path: str | None = None,
        checkpointer: Any = None,
        *,
        model_name: str | None = None,
        thinking_enabled: bool = True,
        subagent_enabled: bool = False,
        plan_mode: bool = False,
        agent_name: str | None = None,
        available_skills: set[str] | None = None,
        middlewares: list | None = None,
        environment: str | None = None,
    ) -> None:
        self.config_path = config_path
        self.checkpointer = checkpointer
        self.model_name = model_name
        self.thinking_enabled = thinking_enabled
        self.subagent_enabled = subagent_enabled
        self.plan_mode = plan_mode
        self.agent_name = agent_name
        self.available_skills = available_skills
        self.middlewares = middlewares
        self.environment = environment


class _MinimalDeerFlowClient:
    """Older-signature fake lacking ``middlewares`` / ``available_skills``."""

    def __init__(
        self,
        config_path: str | None = None,
        checkpointer: Any = None,
        *,
        model_name: str | None = None,
        thinking_enabled: bool = True,
        subagent_enabled: bool = False,
        plan_mode: bool = False,
    ) -> None:
        self.config_path = config_path
        self.checkpointer = checkpointer
        self.model_name = model_name
        self.thinking_enabled = thinking_enabled
        self.subagent_enabled = subagent_enabled
        self.plan_mode = plan_mode


# ---------------------------------------------------------------------------
# Fake TUI surface (Session / LaunchPlan / DeerFlowTUI)
# ---------------------------------------------------------------------------


class _FakeSession:
    def __init__(self, client: Any = None, writer: Any = None, _loop: Any = None) -> None:
        self.client = client
        self.writer = writer
        self._loop = _loop
        self.close_called = False

    def close(self) -> None:
        self.close_called = True


class _FakeLaunchPlan:
    def __init__(
        self,
        mode: str | None = None,
        message: str | None = None,
        read_stdin: bool = False,
        thread_id: str | None = None,
        continue_recent: bool = False,
        forced_tui: bool = False,
        reason: str = "",
    ) -> None:
        self.mode = mode
        self.message = message
        self.thread_id = thread_id
        self.continue_recent = continue_recent


class _FakeTUIApp:
    last: _FakeTUIApp | None = None

    def __init__(self, session: _FakeSession, plan: _FakeLaunchPlan) -> None:
        self.session = session
        self.plan = plan
        self.run_called = False
        _FakeTUIApp.last = self

    def run(self) -> None:
        self.run_called = True


# ---------------------------------------------------------------------------
# sys.modules injection helpers
# ---------------------------------------------------------------------------


_TUI_MODULES = (
    "deerflow",
    "deerflow.client",
    "deerflow.tui",
    "deerflow.tui.app",
    "deerflow.tui.session",
    "deerflow.tui.cli",
)


def _install_fake_tui(client_cls: type) -> dict[str, ModuleType | None]:
    """Inject fake deerflow.client + deerflow.tui.* modules into sys.modules."""
    fake_df = ModuleType("deerflow")
    fake_df_client = ModuleType("deerflow.client")
    fake_df_client.DeerFlowClient = client_cls  # type: ignore[attr-defined]
    fake_tui = ModuleType("deerflow.tui")
    fake_tui_app = ModuleType("deerflow.tui.app")
    fake_tui_app.DeerFlowTUI = _FakeTUIApp  # type: ignore[attr-defined]
    fake_tui_session = ModuleType("deerflow.tui.session")
    fake_tui_session.Session = _FakeSession  # type: ignore[attr-defined]
    fake_tui_cli = ModuleType("deerflow.tui.cli")
    fake_tui_cli.LaunchPlan = _FakeLaunchPlan  # type: ignore[attr-defined]

    prev = {k: sys.modules.get(k) for k in _TUI_MODULES}
    sys.modules["deerflow"] = fake_df
    sys.modules["deerflow.client"] = fake_df_client
    sys.modules["deerflow.tui"] = fake_tui
    sys.modules["deerflow.tui.app"] = fake_tui_app
    sys.modules["deerflow.tui.session"] = fake_tui_session
    sys.modules["deerflow.tui.cli"] = fake_tui_cli
    return prev


def _restore_tui(prev: dict[str, ModuleType | None]) -> None:
    for key, mod in prev.items():
        if mod is not None:
            sys.modules[key] = mod
        else:
            sys.modules.pop(key, None)


# ---------------------------------------------------------------------------
# build_tui_client
# ---------------------------------------------------------------------------


def test_build_tui_client_bakes_mode_flags() -> None:
    """Profile mode/plan/subagent are baked into the DeerFlowClient constructor."""
    prev = _install_fake_tui(_FakeDeerFlowClient)
    try:
        profile = DeerFlowProfile(name="p", mode="pro", plan_mode=True, subagent_enabled=False)
        client = build_tui_client(
            profile,
            "model@openai",
            Path("/c.yaml"),
            checkpointer="ckpt",
            middlewares=["mw"],
            available_skills={"s1"},
        )
        assert isinstance(client, _FakeDeerFlowClient)
        # pro mode -> thinking_enabled True
        assert client.thinking_enabled is True
        assert client.plan_mode is True
        assert client.subagent_enabled is False
        assert client.model_name == "model@openai"
        assert client.config_path == "/c.yaml"
        assert client.checkpointer == "ckpt"
        assert client.middlewares == ["mw"]
        assert client.available_skills == {"s1"}
    finally:
        _restore_tui(prev)


def test_build_tui_client_passes_only_supported_kwargs() -> None:
    """middlewares/available_skills are dropped when the constructor lacks them."""
    prev = _install_fake_tui(_MinimalDeerFlowClient)
    try:
        profile = DeerFlowProfile(name="p", mode="flash")
        # _MinimalDeerFlowClient.__init__ has no middlewares/available_skills
        # params — build_tui_client must drop them rather than raise TypeError.
        client = build_tui_client(
            profile,
            None,
            Path("/c.yaml"),
            middlewares=["mw"],
            available_skills={"s"},
        )
        assert isinstance(client, _MinimalDeerFlowClient)
        assert client.thinking_enabled is False  # flash -> thinking disabled
        assert client.plan_mode is False
    finally:
        _restore_tui(prev)


# ---------------------------------------------------------------------------
# run_deerflow_tui
# ---------------------------------------------------------------------------


def test_run_deerflow_tui_launches_app(monkeypatch, tmp_path) -> None:
    """run_deerflow_tui prepares the profile, wraps the client in a Session, and runs the TUI."""
    # Bypass the harnessing feature gate and runtime prep so no real deerflow
    # config/model/Textual run is needed.
    monkeypatch.setattr("genai_tk.agents.deer_flow.tui.require_feature", lambda *a, **k: None)

    async def fake_prepare(profile_name: str, **_kwargs: Any):
        profile = DeerFlowProfile(name=profile_name, mode="pro", plan_mode=True)
        return profile, "model@openai", tmp_path / "config.yaml", None

    monkeypatch.setattr("genai_tk.agents.deer_flow.runtime.prepare_profile", fake_prepare)
    monkeypatch.setattr("genai_tk.agents.deer_flow.runtime.build_cli_middlewares", lambda _mws: ["mw"])
    monkeypatch.setattr("genai_tk.agents.deer_flow.tui._build_checkpointer", lambda: "ckpt")

    _FakeTUIApp.last = None
    prev = _install_fake_tui(_FakeDeerFlowClient)
    try:
        rc = run_deerflow_tui(
            "simple-deerflow",
            message="hi",
            thread_id="t1",
            continue_recent=True,
        )
        assert rc == 0

        app = _FakeTUIApp.last
        assert app is not None
        assert app.run_called is True

        # Session wraps the built client; no threads_meta persistence writer.
        assert app.session.writer is None
        assert isinstance(app.session.client, _FakeDeerFlowClient)
        # Mode baked into the client constructor.
        assert app.session.client.thinking_enabled is True  # pro
        assert app.session.client.plan_mode is True
        assert app.session.client.checkpointer == "ckpt"
        assert app.session.client.middlewares == ["mw"]
        # Session closed on shutdown.
        assert app.session.close_called is True

        # Plan carries the launch options.
        assert app.plan.mode == "tui"
        assert app.plan.message == "hi"
        assert app.plan.thread_id == "t1"
        assert app.plan.continue_recent is True
    finally:
        _restore_tui(prev)


# ---------------------------------------------------------------------------
# cli agents tui — command dispatch via CliRunner
# ---------------------------------------------------------------------------


@pytest.fixture
def agents_app() -> typer.Typer:
    app = typer.Typer()
    AgentCommands().register(app)
    return app


def test_tui_command_dispatches_deerflow(monkeypatch, agents_app: typer.Typer) -> None:
    """A DeerFlow profile invokes run_deerflow_tui with the resolved name + flags."""
    captured: dict[str, Any] = {}

    def fake_run(profile_name: str, **kwargs: Any) -> int:
        captured["profile_name"] = profile_name
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr("genai_tk.agents.deer_flow.tui.run_deerflow_tui", fake_run)

    result = CliRunner().invoke(agents_app, ["agents", "tui", "simple-deerflow", "--mode", "ultra"])
    assert result.exit_code == 0, result.output
    assert captured["profile_name"] == "simple-deerflow"
    assert captured["kwargs"]["mode_override"] == "ultra"


def test_tui_command_langchain_not_supported(agents_app: typer.Typer) -> None:
    """A LangChain profile exits with a 'not yet supported' message."""
    result = CliRunner().invoke(agents_app, ["agents", "tui", "simple"])
    assert result.exit_code != 0
    assert "not yet supported" in result.output
    assert "langchain" in result.output


def test_tui_command_unknown_profile(agents_app: typer.Typer) -> None:
    """An unknown profile exits with a 'not found' error."""
    result = CliRunner().invoke(agents_app, ["agents", "tui", "no-such-profile-xyz"])
    assert result.exit_code != 0
    assert "not found" in result.output
