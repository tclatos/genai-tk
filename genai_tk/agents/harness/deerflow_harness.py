"""DeerFlow harness adapter — wraps :class:`EmbeddedDeerFlowClient` behind the
shared :class:`BaseHarness` interface.

``EmbeddedDeerFlowClient.stream_message`` already yields the canonical harness
event model (:mod:`genai_tk.agents.harness.events`) directly, so this adapter
does not translate — it forwards events as-is and appends the final
:class:`EndEvent`.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

from genai_tk.agents.deer_flow.profile import DeerFlowProfile
from genai_tk.agents.harness.base import BaseHarness
from genai_tk.agents.harness.events import (
    EndEvent,
    ErrorEvent,
    HarnessModel,
    HarnessSkill,
    StreamEvent,
)


class DeerFlowHarness(BaseHarness):
    """Harness session backed by the embedded DeerFlow client.

    Args:
        profile_name: Name of a DeerFlow profile in the unified ``agents:`` config.
        llm_override: LLM identifier that takes precedence over ``profile.llm``.
        mode_override: Reasoning mode override (``flash`` | ``thinking`` | ``pro``
            | ``ultra``); ``None`` keeps the profile's configured mode.
        sandbox_override: Sandbox override (``local`` | ``docker``); ``None`` keeps
            the profile's configured sandbox.
        extra_mcp: Additional MCP server names appended to the profile's servers.
    """

    name = "deerflow"

    def __init__(
        self,
        profile_name: str,
        *,
        llm_override: str | None = None,
        mode_override: str | None = None,
        sandbox_override: str | None = None,
        extra_mcp: list[str] | None = None,
    ) -> None:
        self._profile_name = profile_name
        self._llm_override = llm_override
        self._mode_override = mode_override
        self._sandbox_override = sandbox_override
        self._extra_mcp = list(extra_mcp or [])
        self._client: Any = None
        self._profile: DeerFlowProfile | None = None
        self._model_name: str | None = None

    @property
    def profile(self) -> DeerFlowProfile | None:
        """Resolved profile, populated once :meth:`ensure_ready` or :meth:`astream` has run."""
        return self._profile

    @property
    def model_name(self) -> str | None:
        """Resolved model name, populated once :meth:`ensure_ready` or :meth:`astream` has run."""
        return self._model_name

    async def ensure_ready(self) -> None:
        """Eagerly prepare the DeerFlow config and embedded client.

        Useful for callers (e.g. the Streamlit workbench) that need
        ``profile``/``model_name`` before the first :meth:`astream` call.
        """
        await self._ensure_client()

    async def _ensure_client(self) -> Any:
        if self._client is None:
            from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient
            from genai_tk.agents.deer_flow.runtime import build_cli_middlewares, prepare_profile
            from genai_tk.utils.tracing import HarnessTraceMetadata, apply_harness_trace_metadata

            profile, model_name, config_path, _warnings = await prepare_profile(
                profile_name=self._profile_name,
                llm_override=self._llm_override,
                extra_mcp=self._extra_mcp,
                mode_override=self._mode_override,
                verbose=False,
                sandbox_override=self._sandbox_override,
            )
            apply_harness_trace_metadata(
                HarnessTraceMetadata(
                    harness=self.name,
                    profile_name=profile.name,
                    model_name=model_name,
                    environment=os.environ.get("GENAI_TK_ENV"),
                )
            )
            middlewares = build_cli_middlewares(profile.middlewares)
            available_skills = set(profile.available_skills) if profile.available_skills is not None else None
            self._client = EmbeddedDeerFlowClient(
                config_path=config_path,
                model_name=model_name,
                middlewares=middlewares,
                available_skills=available_skills,
            )
            self._profile = profile
            self._model_name = model_name
        return self._client

    async def astream(self, message: str, *, thread_id: str | None = None) -> AsyncIterator[StreamEvent]:
        client = await self._ensure_client()
        profile = self._profile
        assert profile is not None
        tid = thread_id or "harness-default"
        try:
            async for event in client.stream_message(
                tid,
                message,
                model_name=self._model_name,
                mode=profile.mode,
                subagent_enabled=profile.subagent_enabled,
                plan_mode=profile.plan_mode,
            ):
                yield event
        except Exception as exc:
            logger.opt(exception=True).warning(f"DeerFlowHarness stream error: {exc}")
            yield ErrorEvent(message=str(exc))
        yield EndEvent()

    async def list_models(self) -> list[HarnessModel]:
        client = await self._ensure_client()
        return [HarnessModel(name=m.get("name", ""), provider=m.get("provider", "")) for m in client.list_models()]

    async def list_skills(self) -> list[HarnessSkill]:
        client = await self._ensure_client()
        return [
            HarnessSkill(name=s.get("name", ""), enabled=bool(s.get("enabled", True))) for s in client.list_skills()
        ]

    async def get_graph(self) -> Any:
        """Return the compiled LangGraph graph for introspection (best-effort).

        See :meth:`EmbeddedDeerFlowClient.get_graph` — this bypasses the
        tracing/Langfuse/authorization wiring that ``astream`` performs via
        ``DeerFlowClient.stream()``. Use for ``isinstance`` checks or
        inspection only, not to drive a production turn.
        """
        client = await self._ensure_client()
        profile = self._profile
        assert profile is not None
        return client.get_graph(
            model_name=self._model_name,
            subagent_enabled=profile.subagent_enabled,
            is_plan_mode=profile.plan_mode,
        )

    async def get_checkpointer(self) -> Any:
        client = await self._ensure_client()
        return client.get_checkpointer()
