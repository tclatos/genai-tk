"""NeMo Relay to AUIP Bridge.

Translates NeMo Relay scope transitions (agent, llm, tool) and semantic marks
(skills, plans, hitl) into normalized Agent-User Interaction Protocol (AUIP) events.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Any

from loguru import logger

from genai_tk.protocols.auip.models import (
    AuipArtifactEvent,
    AuipEndEvent,
    AuipEvent,
    AuipHitlRequestEvent,
    AuipPhaseEvent,
    AuipToolCallEvent,
    AuipToolResultEvent,
    AuipUsageTelemetryEvent,
)


class RelayAuipBridge:
    """Bridges NeMo Relay lifecycle events into an AUIP stream or callback pipeline.

    Can be used in two modes:
    1. Callback mode: Calls ``on_auip_event(event)`` synchronously or schedules it.
    2. Async Queue mode: Pushes translated AUIP events into an ``asyncio.Queue``
       which can be consumed as an async iterator.
    """

    def __init__(
        self,
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
        on_event: Callable[[AuipEvent], None] | None = None,
        queue: asyncio.Queue[AuipEvent] | None = None,
    ) -> None:
        self.session_id = session_id
        self.turn_id = turn_id
        self._on_event = on_event
        self._queue = queue
        self._tool_starts: dict[str, dict[str, Any]] = {}
        self._attached = False

    def attach(self) -> RelayAuipBridge:
        """Register this bridge to receive live NeMo Relay events."""
        if not self._attached:
            try:
                from genai_tk.extra.monitoring.nemo_relay_setup import add_relay_event_listener

                add_relay_event_listener(self.handle_relay_event)
                self._attached = True
            except Exception as exc:
                logger.debug(f"Could not attach RelayAuipBridge: {exc}")
        return self

    def detach(self) -> None:
        """Deregister this bridge from NeMo Relay events."""
        if self._attached:
            try:
                from genai_tk.extra.monitoring.nemo_relay_setup import remove_relay_event_listener

                remove_relay_event_listener(self.handle_relay_event)
                self._attached = False
            except Exception as exc:
                logger.debug(f"Could not detach RelayAuipBridge: {exc}")

    def __enter__(self) -> RelayAuipBridge:
        return self.attach()

    def __exit__(self, *args: Any) -> None:
        self.detach()

    async def __aenter__(self) -> RelayAuipBridge:
        return self.attach()

    async def __aexit__(self, *args: Any) -> None:
        self.detach()

    def emit(self, event: AuipEvent) -> None:
        """Deliver an AUIP event to listeners and queue."""
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception as exc:
                logger.warning(f"Error in AUIP event callback: {exc}")

        if self._queue is not None:
            try:
                self._queue.put_nowait(event)
            except Exception as exc:
                logger.warning(f"Error pushing to AUIP queue: {exc}")

    def handle_relay_event(self, event: dict[str, Any]) -> AuipEvent | None:
        """Parse one raw NeMo Relay JSON event dictionary into an AUIP event."""
        kind = event.get("kind")
        sc = event.get("scope_category")
        cat = event.get("category")
        name = str(event.get("name") or "")
        uuid = str(event.get("uuid") or "")
        ts = str(event.get("timestamp") or "")

        auip_event: AuipEvent | None = None

        if kind == "scope":
            if sc == "start":
                if cat == "tool":
                    args = self._extract_tool_args(event)
                    self._tool_starts[uuid] = {
                        "name": name,
                        "args": args,
                        "ts": ts,
                    }
                    auip_event = AuipToolCallEvent(
                        session_id=self.session_id,
                        turn_id=self.turn_id,
                        tool_name=name,
                        call_id=uuid,
                        args=args,
                    )
                elif cat in ("agent", "step"):
                    auip_event = AuipPhaseEvent(
                        session_id=self.session_id,
                        turn_id=self.turn_id,
                        phase=name or "agent",
                    )

            elif sc == "end":
                if cat == "tool":
                    cached = self._tool_starts.pop(uuid, {})
                    tool_name = name or cached.get("name", "unknown")
                    result_text, status = self._extract_tool_result(event)
                    duration_ms = self._calc_duration_ms(cached.get("ts"), ts)
                    auip_event = AuipToolResultEvent(
                        session_id=self.session_id,
                        turn_id=self.turn_id,
                        tool_name=tool_name,
                        call_id=uuid,
                        content=result_text,
                        duration_ms=duration_ms,
                        status=status,
                    )
                elif cat == "llm":
                    usage = self._extract_llm_usage(event)
                    if usage:
                        auip_event = AuipUsageTelemetryEvent(
                            session_id=self.session_id,
                            turn_id=self.turn_id,
                            input_tokens=usage.get("prompt_tokens", 0),
                            output_tokens=usage.get("completion_tokens", 0),
                            model_name=usage.get("model", ""),
                        )
                elif cat == "agent" and not event.get("parent_uuid"):
                    # Root agent scope ended
                    auip_event = AuipEndEvent(
                        session_id=self.session_id,
                        turn_id=self.turn_id,
                        status="success",
                    )

        elif kind == "mark":
            mark_name = event.get("name")
            data = event.get("data") if isinstance(event.get("data"), dict) else {}

            if mark_name == "skill.load":
                skill_name = str(data.get("skill_name") or "skill")
                skill_path = str(data.get("path") or "")
                auip_event = AuipArtifactEvent(
                    session_id=self.session_id,
                    turn_id=self.turn_id,
                    artifact_type="skill",
                    title=f"Skill: {skill_name}",
                    content=f"Loaded skill from {skill_path}",
                    metadata={"skill_name": skill_name, "path": skill_path},
                )
            elif mark_name == "plan.update":
                todos = data.get("todos", [])
                auip_event = AuipPhaseEvent(
                    session_id=self.session_id,
                    turn_id=self.turn_id,
                    phase="planner",
                    state={"todos": todos},
                )
            elif mark_name in ("hitl.request", "clarification"):
                question = str(data.get("question") or "")
                c_type = str(data.get("type") or "missing_info")
                auip_event = AuipHitlRequestEvent(
                    session_id=self.session_id,
                    turn_id=self.turn_id,
                    question=question,
                    clarification_type=c_type,
                    context=str(data.get("context") or ""),
                )

        if auip_event is not None:
            self.emit(auip_event)

        return auip_event

    @staticmethod
    def _extract_tool_args(event: dict[str, Any]) -> dict[str, Any]:
        cp = event.get("category_profile") or {}
        req = cp.get("tool_request") or {}
        if isinstance(req, dict):
            args = req.get("args") or req.get("arguments") or req.get("tool_input")
            if isinstance(args, dict):
                return args
        meta = event.get("metadata") or {}
        if isinstance(meta.get("tool_args"), dict):
            return meta["tool_args"]
        return {}

    @staticmethod
    def _extract_tool_result(event: dict[str, Any]) -> tuple[str, Any]:
        cp = event.get("category_profile") or {}
        res = cp.get("tool_response") or {}
        content = ""
        if isinstance(res, dict):
            content = str(res.get("content") or res.get("output") or res.get("result") or "")
        meta = event.get("metadata") or {}
        status = "error" if meta.get("otel.status_code") == "ERROR" else "success"
        return content, status

    @staticmethod
    def _extract_llm_usage(event: dict[str, Any]) -> dict[str, Any] | None:
        cp = event.get("category_profile") or {}
        ar = cp.get("annotated_response") or {}
        usage = ar.get("usage")
        if isinstance(usage, dict):
            return {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "model": str(cp.get("model") or ""),
            }
        return None

    @staticmethod
    def _calc_duration_ms(start_ts: str | None, end_ts: str | None) -> float:
        if not start_ts or not end_ts:
            return 0.0
        try:
            t0 = datetime.fromisoformat(start_ts)
            t1 = datetime.fromisoformat(end_ts)
            return round((t1 - t0).total_seconds() * 1000, 1)
        except Exception:
            return 0.0
