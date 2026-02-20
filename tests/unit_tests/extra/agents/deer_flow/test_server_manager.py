"""Unit tests for DeerFlowServerManager."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genai_tk.extra.agents.deer_flow.server_manager import DeerFlowServerManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _manager(tmp_path: Path, deer_flow_path: str | None = None) -> DeerFlowServerManager:
    """Return a manager pointing at tmp_path/backend as its deer-flow location."""
    p = deer_flow_path or str(tmp_path)
    # Create backend subdir so _resolve_backend_path doesn't raise
    (tmp_path / "backend").mkdir(parents=True, exist_ok=True)
    return DeerFlowServerManager(deer_flow_path=p)


# ---------------------------------------------------------------------------
# is_running
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_running_when_both_up(tmp_path: Path) -> None:
    """is_running returns True when both health endpoints respond."""
    mgr = _manager(tmp_path)
    with patch.object(mgr, "_check_url", new_callable=AsyncMock, return_value=True):
        assert await mgr.is_running() is True


@pytest.mark.asyncio
async def test_is_running_when_lg_down(tmp_path: Path) -> None:
    """is_running returns False when LangGraph is unreachable."""
    mgr = _manager(tmp_path)
    call_results = [False, True]  # LG down, GW up

    async def _check(url: str) -> bool:
        return call_results.pop(0)

    with patch.object(mgr, "_check_url", side_effect=_check):
        assert await mgr.is_running() is False


@pytest.mark.asyncio
async def test_is_running_when_gw_down(tmp_path: Path) -> None:
    """is_running returns False when Gateway is unreachable."""
    mgr = _manager(tmp_path)
    call_results = [True, False]  # LG up, GW down

    async def _check(url: str) -> bool:
        return call_results.pop(0)

    with patch.object(mgr, "_check_url", side_effect=_check):
        assert await mgr.is_running() is False


# ---------------------------------------------------------------------------
# start — already running → no subprocess
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_skips_popen_when_already_running(tmp_path: Path) -> None:
    """start() does not launch subprocesses when both servers are already up."""
    mgr = _manager(tmp_path)
    with (
        patch.object(mgr, "_check_url", new_callable=AsyncMock, return_value=True),
        patch("genai_tk.extra.agents.deer_flow.server_manager.subprocess.Popen") as mock_popen,
    ):
        await mgr.start()
        mock_popen.assert_not_called()


# ---------------------------------------------------------------------------
# start — not running → launches subprocesses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_launches_subprocesses(tmp_path: Path) -> None:
    """start() calls Popen for both LangGraph and Gateway when servers are down."""
    mgr = _manager(tmp_path)
    mock_proc = MagicMock()
    mock_proc.pid = 1234
    mock_proc.poll = MagicMock(return_value=None)  # still running

    with (
        patch.object(mgr, "_check_url", new_callable=AsyncMock, return_value=False),
        patch("genai_tk.extra.agents.deer_flow.server_manager.subprocess.Popen", return_value=mock_proc) as mock_popen,
        patch.object(mgr, "_wait_for_ready", new_callable=AsyncMock),
    ):
        await mgr.start()
        assert mock_popen.call_count == 2  # LangGraph + Gateway
        assert mgr._owns_servers is True


@pytest.mark.asyncio
async def test_start_langgraph_command_contains_dev(tmp_path: Path) -> None:
    """LangGraph subprocess is started with 'langgraph dev' command."""
    mgr = _manager(tmp_path)
    mock_proc = MagicMock()
    mock_proc.pid = 999
    mock_proc.poll = MagicMock(return_value=None)

    captured_calls = []

    def _popen(cmd, **kwargs):
        captured_calls.append(cmd)
        return mock_proc

    with (
        patch.object(mgr, "_check_url", new_callable=AsyncMock, return_value=False),
        patch("genai_tk.extra.agents.deer_flow.server_manager.subprocess.Popen", side_effect=_popen),
        patch.object(mgr, "_wait_for_ready", new_callable=AsyncMock),
    ):
        await mgr.start()

    lg_cmd = captured_calls[0]
    assert "langgraph" in lg_cmd
    assert "dev" in lg_cmd


@pytest.mark.asyncio
async def test_start_gateway_command_contains_uvicorn(tmp_path: Path) -> None:
    """Gateway subprocess is started with 'uvicorn' command."""
    mgr = _manager(tmp_path)
    mock_proc = MagicMock()
    mock_proc.pid = 998
    mock_proc.poll = MagicMock(return_value=None)

    captured_calls = []

    def _popen(cmd, **kwargs):
        captured_calls.append(cmd)
        return mock_proc

    with (
        patch.object(mgr, "_check_url", new_callable=AsyncMock, return_value=False),
        patch("genai_tk.extra.agents.deer_flow.server_manager.subprocess.Popen", side_effect=_popen),
        patch.object(mgr, "_wait_for_ready", new_callable=AsyncMock),
    ):
        await mgr.start()

    gw_cmd = captured_calls[1]
    assert "uvicorn" in gw_cmd


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_sends_sigterm_when_owns_servers(tmp_path: Path) -> None:
    """stop() sends SIGTERM to process groups when we own the servers."""
    mgr = _manager(tmp_path)
    mgr._owns_servers = True

    mock_proc = MagicMock()
    mock_proc.pid = 1234
    mock_proc.poll = MagicMock(return_value=None)  # still running
    mock_proc.wait = MagicMock()

    mgr._lg_proc = mock_proc
    mgr._gw_proc = mock_proc

    with (
        patch("genai_tk.extra.agents.deer_flow.server_manager.os.killpg") as mock_killpg,
        patch("genai_tk.extra.agents.deer_flow.server_manager.os.getpgid", return_value=1234),
    ):
        await mgr.stop()
        assert mock_killpg.call_count == 2

    assert mgr._owns_servers is False
    assert mgr._lg_proc is None
    assert mgr._gw_proc is None


@pytest.mark.asyncio
async def test_stop_noop_when_not_owned(tmp_path: Path) -> None:
    """stop() does nothing when _owns_servers is False."""
    mgr = _manager(tmp_path)
    mgr._owns_servers = False

    with patch("genai_tk.extra.agents.deer_flow.server_manager.os.killpg") as mock_killpg:
        await mgr.stop()
        mock_killpg.assert_not_called()


# ---------------------------------------------------------------------------
# _resolve_backend_path
# ---------------------------------------------------------------------------


def test_resolve_backend_path_returns_correct_path(tmp_path: Path) -> None:
    """_resolve_backend_path returns <deer_flow_path>/backend."""
    backend_dir = tmp_path / "backend"
    backend_dir.mkdir()
    mgr = DeerFlowServerManager(deer_flow_path=str(tmp_path))
    assert mgr._resolve_backend_path() == backend_dir.resolve()


def test_resolve_backend_path_raises_when_no_path() -> None:
    """_resolve_backend_path raises RuntimeError when no path is configured."""
    # Create the manager *inside* the patch so the env override is active at construction time
    with patch.dict(os.environ, {"DEER_FLOW_PATH": ""}, clear=False):
        mgr = DeerFlowServerManager(deer_flow_path="")
        with pytest.raises(RuntimeError, match="DEER_FLOW_PATH"):
            mgr._resolve_backend_path()


def test_resolve_backend_path_raises_when_backend_missing(tmp_path: Path) -> None:
    """_resolve_backend_path raises FileNotFoundError when backend/ dir is absent."""
    mgr = DeerFlowServerManager(deer_flow_path=str(tmp_path))
    # backend/ does NOT exist in tmp_path
    with pytest.raises(FileNotFoundError):
        mgr._resolve_backend_path()


# ---------------------------------------------------------------------------
# _wait_for_ready timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_for_ready_raises_timeout(tmp_path: Path) -> None:
    """_wait_for_ready raises TimeoutError if servers never become ready."""
    mgr = DeerFlowServerManager(deer_flow_path=str(tmp_path), start_timeout=0.05)
    mgr._lg_proc = MagicMock(poll=MagicMock(return_value=None))
    mgr._gw_proc = MagicMock(poll=MagicMock(return_value=None))

    with patch.object(mgr, "_check_url", new_callable=AsyncMock, return_value=False):
        with pytest.raises(TimeoutError):
            await mgr._wait_for_ready()


@pytest.mark.asyncio
async def test_wait_for_ready_raises_on_early_exit(tmp_path: Path) -> None:
    """_wait_for_ready raises RuntimeError if a subprocess exits early."""
    mgr = DeerFlowServerManager(deer_flow_path=str(tmp_path), start_timeout=5.0)

    dead_proc = MagicMock()
    dead_proc.poll = MagicMock(return_value=1)  # exited with error
    dead_proc.returncode = 1
    dead_proc.stdout = None

    mgr._lg_proc = dead_proc
    mgr._gw_proc = MagicMock(poll=MagicMock(return_value=None))

    with patch.object(mgr, "_check_url", new_callable=AsyncMock, return_value=False):
        with pytest.raises(RuntimeError, match="exited with code"):
            await mgr._wait_for_ready()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_context_manager_calls_start_stop(tmp_path: Path) -> None:
    """Async context manager calls start on enter and stop on exit."""
    mgr = _manager(tmp_path)

    with (
        patch.object(mgr, "start", new_callable=AsyncMock) as mock_start,
        patch.object(mgr, "stop", new_callable=AsyncMock) as mock_stop,
    ):
        async with mgr:
            pass

        mock_start.assert_awaited_once()
        mock_stop.assert_awaited_once()
