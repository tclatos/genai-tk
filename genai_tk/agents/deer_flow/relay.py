"""NeMo Relay trajectory observability integration for DeerFlow harness.

Provides :class:`NemoRelayDeerFlowMiddleware`, which intercepts model calls and
tool executions via NeMo Relay and emits DeerFlow-specific semantic marks
(harness configuration, mode, plan/todo transitions, skills).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

from genai_tk.utils.nemo_relay_setup import is_nemo_relay_available


def json_safe(value: Any) -> Any:
    """Return a conservative JSON-compatible value for ATOF metadata."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [json_safe(item) for item in value]
    if isinstance(value, bytes | bytearray):
        return f"<{type(value).__name__}: {len(value)} bytes>"
    return repr(value)


try:
    import nemo_relay  # noqa: F401
    from nemo_relay.integrations.langchain.middleware import NemoRelayMiddleware
except ImportError:
    NemoRelayMiddleware = object  # type: ignore[misc,assignment]


class NemoRelayDeerFlowMiddleware(NemoRelayMiddleware):
    """Route DeerFlow model/tool calls through NeMo Relay and emit semantic marks.

    Inherits LangChain model and tool wrapping from
    :class:`nemo_relay.integrations.langchain.middleware.NemoRelayMiddleware`,
    then emits DeerFlow agent configuration and mode marks at turn start.
    """

    def __init__(
        self,
        *,
        name: str = "NemoRelayDeerFlowMiddleware",
        agent_name: str | None = None,
        mode: str | None = None,
        skills: Sequence[str] | None = None,
        subagents: Sequence[Mapping[str, Any]] | None = None,
        sandbox: str | None = None,
    ) -> None:
        if not is_nemo_relay_available():
            raise RuntimeError("nemo_relay is required to initialize NemoRelayDeerFlowMiddleware")
        super().__init__(name=name)
        self._agent_name = agent_name
        self._mode = mode
        self._skills = list(skills) if skills is not None else None
        self._subagents = list(subagents) if subagents is not None else None
        self._sandbox = sandbox

    def before_agent(self, state: Any, runtime: Any) -> None:
        """Emit run configuration metadata for sync DeerFlow runs."""
        self._emit_agent_configuration()

    async def abefore_agent(self, state: Any, runtime: Any) -> None:
        """Emit run configuration metadata for async DeerFlow runs."""
        self._emit_agent_configuration()

    def _emit_agent_configuration(self) -> None:
        """Emit a DeerFlow configured mark event to the active NeMo Relay scope."""
        data: dict[str, Any] = {
            "harness": "deerflow",
        }
        if self._agent_name is not None:
            data["agent_name"] = self._agent_name
        if self._mode is not None:
            data["mode"] = self._mode
        if self._skills is not None:
            data["skills"] = list(self._skills)
        if self._subagents is not None:
            data["subagents"] = list(self._subagents)
        if self._sandbox is not None:
            data["sandbox"] = self._sandbox

        event_metadata: dict[str, Any] = {
            "integration": "deerflow",
            "deerflow_kind": "harness",
            "phase": "configured",
        }
        if self._agent_name is not None:
            event_metadata["agent_name"] = self._agent_name

        try:
            import nemo_relay

            nemo_relay.scope.event(
                "DeerFlow Configured",
                data=json_safe(data),
                metadata=json_safe(event_metadata),
            )
        except Exception:
            logger.debug("NeMo Relay: DeerFlow mark emission failed")


def add_deerflow_nemo_relay_integration(
    middlewares: list[Any] | None = None,
    *,
    agent_name: str | None = None,
    mode: str | None = None,
    skills: Sequence[str] | None = None,
    subagents: Sequence[Mapping[str, Any]] | None = None,
    sandbox: str | None = None,
) -> list[Any]:
    """Append :class:`NemoRelayDeerFlowMiddleware` to middleware list if available.

    Args:
        middlewares: Existing list of middleware objects.
        agent_name: Name of the DeerFlow agent.
        mode: DeerFlow reasoning mode (flash | thinking | pro | ultra).
        skills: Discoverable skill names.
        subagents: Subagent descriptions if enabled.
        sandbox: Sandbox type (local | docker).

    Returns:
        List of middlewares containing the NeMo Relay middleware if available.
    """
    result = list(middlewares or [])
    if not is_nemo_relay_available():
        return result

    # Check if already present
    for mw in result:
        if isinstance(mw, NemoRelayDeerFlowMiddleware):
            return result

    try:
        relay_mw = NemoRelayDeerFlowMiddleware(
            agent_name=agent_name,
            mode=mode,
            skills=skills,
            subagents=subagents,
            sandbox=sandbox,
        )
        result.append(relay_mw)
    except Exception as exc:
        logger.debug(f"Could not initialize NemoRelayDeerFlowMiddleware: {exc}")

    return result
