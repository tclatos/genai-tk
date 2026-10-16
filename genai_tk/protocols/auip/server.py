"""FastAPI server exposing AUIP and official AG-UI standard streaming endpoints.

Provides:
- ``POST /agent``: Standard AG-UI protocol endpoint consumed by ``@assistant-ui/react-ag-ui`` / ``@ag-ui/client``.
- ``POST /api/v1/auip/stream``: Canonical AUIP Server-Sent Events endpoint.
- ``GET /profiles``: Enumerate discoverable agent profiles for UI selectors.
- ``GET /health``: Liveness check.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ag_ui.core import RunAgentInput
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from genai_tk.agents.harness import create_harness
from genai_tk.agents.harness.registry import list_harness_profiles
from genai_tk.protocols.auip.ag_ui_bridge import harness_events_to_ag_ui
from genai_tk.protocols.auip.config import AuipServerSettings, auip_config


class AuipChatRequest(BaseModel):
    """Payload for the canonical AUIP SSE endpoint."""

    message: str = Field(..., description="User query message")
    profile: str | None = Field(default=None, description="Target agent profile key")
    thread_id: str | None = Field(default=None, description="Conversation thread identifier")


class ProfileSummary(BaseModel):
    """Summary of an available agent profile."""

    key: str
    name: str
    kind: str = ""
    description: str = ""
    harness: str = ""


def _extract_query_text(input_data: RunAgentInput) -> str:
    """Extract plain text message from incoming AG-UI RunAgentInput."""
    if not input_data.messages:
        return ""
    last_msg = input_data.messages[-1]
    content = getattr(last_msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if hasattr(part, "text"):
                parts.append(str(part.text))
            elif isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
        return " ".join(parts)
    return str(content)


def create_app(settings: AuipServerSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI AUIP / AG-UI application."""
    active_settings = settings or auip_config().server

    app = FastAPI(
        title=active_settings.title,
        version="0.1.0",
        description="GenAI Toolkit Agent-User Interaction Protocol (AUIP) and AG-UI Gateway",
    )

    # CORS configuration for web clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "protocol": "auip/1.0",
            "default_profile": active_settings.default_profile,
        }

    @app.get("/profiles", response_model=list[ProfileSummary])
    async def get_profiles() -> list[ProfileSummary]:
        """List all available agent profiles across both harnesses."""
        profiles = list_harness_profiles()
        return [
            ProfileSummary(
                key=p.key,
                name=p.name,
                kind=p.kind,
                description=p.description,
                harness=p.harness,
            )
            for p in profiles
        ]

    @app.post(active_settings.endpoint)
    async def stream_ag_ui(
        input_data: RunAgentInput,
        request: Request,
        profile: str | None = Query(default=None, description="Agent profile override"),
    ) -> StreamingResponse:
        """Standard AG-UI protocol endpoint for assistant-ui and @ag-ui/client."""
        # 1. Resolve agent profile from query param, forwarded props, header, or default
        resolved_profile = profile
        if not resolved_profile and input_data.forwarded_props:
            resolved_profile = input_data.forwarded_props.get("profile")
        if not resolved_profile:
            resolved_profile = request.headers.get("x-agent-profile")
        if not resolved_profile:
            resolved_profile = active_settings.default_profile

        thread_id = input_data.thread_id or f"thread_{uuid.uuid4().hex[:10]}"
        run_id = input_data.run_id or f"run_{uuid.uuid4().hex[:10]}"
        query_text = _extract_query_text(input_data)

        logger.info(
            "Serving AG-UI run {} on thread {} using profile '{}' with query: {}",
            run_id,
            thread_id,
            resolved_profile,
            query_text[:80],
        )

        encoder = EventEncoder()

        async def event_generator() -> AsyncIterator[str]:
            try:
                harness = create_harness(resolved_profile)
            except Exception as exc:
                logger.error("Failed to initialize harness profile '{}': {}", resolved_profile, exc)
                from ag_ui.core import EventType, RunErrorEvent, RunFinishedEvent

                yield encoder.encode(
                    RunErrorEvent(
                        type=EventType.RUN_ERROR,
                        message=f"Agent profile '{resolved_profile}' initialization failed: {exc}",
                    )
                )
                yield encoder.encode(
                    RunFinishedEvent(
                        type=EventType.RUN_FINISHED,
                        thread_id=thread_id,
                        run_id=run_id,
                    )
                )
                return

            try:
                event_stream = harness.astream(query_text, thread_id=thread_id)
                async for ag_ui_event in harness_events_to_ag_ui(
                    event_stream,
                    thread_id=thread_id,
                    run_id=run_id,
                ):
                    yield encoder.encode(ag_ui_event)
            finally:
                await harness.aclose()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/v1/auip/stream")
    async def stream_auip(req: AuipChatRequest) -> StreamingResponse:
        """Canonical AUIP SSE endpoint emitting strongly-typed AUIP JSON events."""
        resolved_profile = req.profile or active_settings.default_profile
        thread_id = req.thread_id or f"thread_{uuid.uuid4().hex[:10]}"

        logger.info("Serving AUIP SSE stream for profile '{}'", resolved_profile)

        async def auip_sse_generator() -> AsyncIterator[str]:
            try:
                harness = create_harness(resolved_profile)
            except Exception as exc:
                logger.error("Failed to create harness for profile '{}': {}", resolved_profile, exc)
                error_payload = json.dumps({"type": "turn.error", "message": str(exc)})
                yield f"event: turn.error\ndata: {error_payload}\n\n"
                return

            try:
                async for auip_event in harness.astream_auip(req.message, thread_id=thread_id):
                    json_data = auip_event.model_dump_json()
                    yield f"event: {auip_event.type}\ndata: {json_data}\n\n"
            finally:
                await harness.aclose()

        return StreamingResponse(
            auip_sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def serve(
    *,
    host: str | None = None,
    port: int | None = None,
    profile: str | None = None,
    reload: bool = False,
) -> None:
    """Run the AUIP / AG-UI server via uvicorn.

    Args:
        host: Listen host IP or domain.
        port: Listen TCP port.
        profile: Default agent profile key override.
        reload: Enable uvicorn auto-reload for development.
    """
    import uvicorn

    cfg = auip_config().server
    target_host = host or cfg.host
    target_port = port or cfg.port
    if profile:
        cfg.default_profile = profile

    app = create_app(cfg)
    logger.info(
        "Starting AUIP / AG-UI server at http://{}:{} (default profile: {})",
        target_host,
        target_port,
        cfg.default_profile,
    )
    uvicorn.run(app, host=target_host, port=target_port, reload=reload)
