"""Unit tests for AUIP / AG-UI FastAPI server endpoints."""

from __future__ import annotations

import httpx
import pytest

from genai_tk.protocols.auip.config import AuipServerSettings
from genai_tk.protocols.auip.server import create_app


@pytest.fixture
def test_app():
    settings = AuipServerSettings(
        host="127.0.0.1",
        port=8000,
        default_profile="fake",
        allowed_profiles=["*"],
        cors_origins=["*"],
    )
    return create_app(settings)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_endpoint(test_app) -> None:
    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["protocol"] == "auip/1.0"
        assert data["default_profile"] == "fake"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_profiles_endpoint(test_app) -> None:
    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/profiles")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        keys = [p["key"] for p in data]
        assert "research" in keys or "browser_agent" in keys or "fake" in keys


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ag_ui_stream_endpoint(test_app, monkeypatch) -> None:
    from collections.abc import AsyncIterator
    from unittest.mock import MagicMock

    from genai_tk.agents.harness.events import EndEvent, TokenEvent

    class FakeMockHarness:
        async def astream(self, message: str, **kwargs) -> AsyncIterator:
            yield TokenEvent(text=f"Echo: {message}")
            yield EndEvent()

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(
        "genai_tk.protocols.auip.server.create_harness",
        lambda profile: FakeMockHarness(),
    )

    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "threadId": "test_th_1",
            "runId": "test_run_1",
            "messages": [{"id": "msg_user", "role": "user", "content": "What is 2+2?"}],
        }
        resp = await client.post("/agent", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        body = resp.text
        assert "RUN_STARTED" in body
        assert "TEXT_MESSAGE_CONTENT" in body
        assert "Echo: What is 2+2?" in body
        assert "RUN_FINISHED" in body


@pytest.mark.unit
@pytest.mark.asyncio
async def test_auip_stream_endpoint(test_app, monkeypatch) -> None:
    from collections.abc import AsyncIterator

    from genai_tk.protocols.auip.models import AuipEndEvent, AuipTokenEvent

    class FakeMockHarness:
        async def astream_auip(self, message: str, **kwargs) -> AsyncIterator:
            yield AuipTokenEvent(text=f"AUIP: {message}")
            yield AuipEndEvent()

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(
        "genai_tk.protocols.auip.server.create_harness",
        lambda profile: FakeMockHarness(),
    )

    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "message": "Hello AUIP",
            "profile": "fake",
            "thread_id": "th_auip_1",
        }
        resp = await client.post("/api/v1/auip/stream", json=payload)
        assert resp.status_code == 200
        body = resp.text
        assert "event: stream.token" in body
        assert "AUIP: Hello AUIP" in body
        assert "event: turn.end" in body
