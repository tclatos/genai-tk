"""Resource policy and cancellation tests against the external sandbox SDK boundary."""

from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from genai_tk.config_mgmt.features import is_available

if not is_available("harnessing"):
    pytest.skip("Requires harnessing extra", allow_module_level=True)

from genai_tk.agents.sandbox.aio_backend import AioSandboxBackend  # noqa: E402
from genai_tk.agents.sandbox.config import write_server_config  # noqa: E402
from genai_tk.agents.sandbox.models import DockerAioSettings  # noqa: E402


@pytest.mark.unit
@pytest.mark.parametrize("values", [{"cpu_limit": 0}, {"cpu_limit": float("inf")}, {"memory_limit_mb": 0}])
def test_invalid_resource_limits_are_rejected(values: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        DockerAioSettings.model_validate(values)


@pytest.mark.integration
def test_generated_server_limits_processes(tmp_path: Path) -> None:
    path = write_server_config(8080, path=tmp_path / "sandbox.toml", allowed_host_paths=[str(tmp_path)])
    config = tomllib.loads(path.read_text())
    assert config["docker"]["pids_limit"] == 256 and config["docker"]["no_new_privileges"] is True


@pytest.mark.unit
async def test_container_resource_limits_reach_sdk() -> None:
    backend = AioSandboxBackend(config=DockerAioSettings(cpu_limit=0.5, memory_limit_mb=512))
    with (
        patch("httpx.AsyncClient") as http_client,
        patch("opensandbox.Sandbox.create", new_callable=AsyncMock) as create,
    ):
        http_client.return_value.__aenter__.return_value.get = AsyncMock()
        await backend.start()
    assert create.call_args.kwargs["resource"] == {"cpu": "0.5", "memory": "512Mi"}


@pytest.mark.unit
async def test_command_deadline_kills_remote_sandbox() -> None:
    async def never_finishes(*args: object) -> None:
        await asyncio.Event().wait()

    sdk = MagicMock()
    sdk.commands.run = AsyncMock(side_effect=never_finishes)
    sdk.kill = AsyncMock()
    sdk.close = AsyncMock()
    backend = AioSandboxBackend()
    backend._sandbox = sdk
    result = await backend.aexecute("sleep 10", timeout=1)
    assert result.exit_code == 124 and sdk.kill.await_count == 1 and backend._sandbox is None


@pytest.mark.unit
async def test_failed_termination_is_not_reported_as_stopped() -> None:
    sdk = MagicMock()
    sdk.kill = AsyncMock(side_effect=ConnectionError("control plane unavailable"))
    backend = AioSandboxBackend()
    backend._sandbox = sdk
    with pytest.raises(ConnectionError, match="control plane unavailable"):
        await backend.stop()
    assert backend._sandbox is sdk


@pytest.mark.unit
async def test_closing_one_container_preserves_shared_server_and_peer_container() -> None:
    server_process = MagicMock()
    server_process._transport._closed = False
    first_sdk, second_sdk = MagicMock(), MagicMock()
    first_sdk.kill, first_sdk.close = AsyncMock(), AsyncMock()
    second_sdk.kill, second_sdk.close = AsyncMock(), AsyncMock()
    first = AioSandboxBackend(stop_server_on_close=False)
    second = AioSandboxBackend(stop_server_on_close=False)
    first._sandbox, first._server_proc = first_sdk, server_process
    second._sandbox = second_sdk

    await first.stop()

    assert first_sdk.kill.await_count == 1
    assert second._sandbox is second_sdk and second_sdk.kill.await_count == 0
    assert server_process.terminate.call_count == 0 and server_process.kill.call_count == 0
    assert server_process._transport._closed is True and first._server_proc is None
    await second.stop()
