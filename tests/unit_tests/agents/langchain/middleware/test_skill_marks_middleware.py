"""Unit tests for SkillLoadMarksMiddleware (nemo_relay mocked)."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from genai_tk.agents.langchain.middleware.skill_marks_middleware import SkillLoadMarksMiddleware

pytestmark = pytest.mark.unit


def _install_fake_relay(monkeypatch: pytest.MonkeyPatch, events: list) -> None:
    """Replace ``nemo_relay`` in sys.modules with a recording fake."""
    fake_scope = SimpleNamespace(event=lambda name, **kwargs: events.append((name, kwargs)))
    monkeypatch.setitem(sys.modules, "nemo_relay", SimpleNamespace(scope=fake_scope))


def test_emits_one_skill_load_mark_per_skill(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict]] = []
    _install_fake_relay(monkeypatch, events)

    SkillLoadMarksMiddleware(["rfq-extraction", "ru-classification"]).before_agent({}, None)

    assert [name for name, _ in events] == ["skill.load", "skill.load"]
    assert events[0][1]["data"] == {"skill_name": "rfq-extraction"}
    assert events[0][1]["metadata"] == {"skill_load_source": "configured"}


def test_async_before_agent_emits_marks(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict]] = []
    _install_fake_relay(monkeypatch, events)

    async def run() -> None:
        await SkillLoadMarksMiddleware(["alpha"]).abefore_agent({}, None)

    asyncio.run(run())
    assert [name for name, _ in events] == ["skill.load"]


def test_emission_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no active scope")

    monkeypatch.setitem(sys.modules, "nemo_relay", SimpleNamespace(scope=SimpleNamespace(event=boom)))

    middleware = SkillLoadMarksMiddleware(["alpha"])
    middleware.before_agent({}, None)  # must not raise

    async def run() -> None:
        await middleware.abefore_agent({}, None)

    asyncio.run(run())
