"""Middleware emitting one ``skill.load`` ATOF mark per configured skill.

The deepagents NeMo Relay integration only records the skill *source
directories* in its aggregated 'Skills Configured' mark. This middleware
records the individual skill names (discovered by the factory from
``SKILL.md`` files) as ``skill.load`` marks, emitted from the ``before_agent``
hook so they land inside the run's root scope.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from loguru import logger


class SkillLoadMarksMiddleware(AgentMiddleware[Any, Any, Any]):
    """Emit a ``skill.load`` NeMo Relay mark for each configured skill at run start."""

    def __init__(self, skill_names: list[str]) -> None:
        self._skill_names = skill_names

    def before_agent(self, state: Any, runtime: Any) -> None:
        self._emit_marks()

    async def abefore_agent(self, state: Any, runtime: Any) -> None:
        self._emit_marks()

    def _emit_marks(self) -> None:
        try:
            import nemo_relay  # noqa: PLC0415

            for skill_name in self._skill_names:
                nemo_relay.scope.event(
                    "skill.load",
                    data={"skill_name": skill_name},
                    metadata={"skill_load_source": "configured"},
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"skill.load mark emission failed: {exc}")
