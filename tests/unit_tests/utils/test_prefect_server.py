"""Tests for the Prefect server lifecycle helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_tk.utils.prefect_server import PrefectConfig, PrefectServer


@pytest.fixture()
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PrefectServer:
    """A PrefectServer bound to a temporary PREFECT_HOME with default config."""
    home = tmp_path / "prefect-home"
    monkeypatch.setattr(
        "genai_tk.utils.prefect_server._load_prefect_config",
        lambda: PrefectConfig(pid_file=str(home / "prefect.pid")),
    )
    monkeypatch.setenv("PREFECT_HOME", str(tmp_path / "prefect-home"))
    monkeypatch.delenv("PREFECT_SQLITE_DB", raising=False)
    return PrefectServer()


def test_backup_and_delete_db_backs_up_and_removes_db_and_sidecars(
    server: PrefectServer,
) -> None:
    db = server._sqlite_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_bytes(b"stale db content")
    Path(f"{db}-wal").write_bytes(b"wal")
    Path(f"{db}-shm").write_bytes(b"shm")

    backup = server._backup_and_delete_db()

    assert backup is not None
    assert backup.exists()
    assert ".bak-" in backup.name
    assert backup.read_bytes() == b"stale db content"
    assert not db.exists()
    assert not Path(f"{db}-wal").exists()
    assert not Path(f"{db}-shm").exists()


def test_backup_and_delete_db_returns_none_without_db(server: PrefectServer) -> None:
    assert server._backup_and_delete_db() is None


def test_log_mentions_db_error_detects_alembic_errors(server: PrefectServer, tmp_path: Path) -> None:
    log = tmp_path / "prefect.log"
    log.write_text("raise ResolutionError(\nResolutionError: No such revision or branch 'a1b2c3'\n")
    assert server._log_mentions_db_error(log)

    other = tmp_path / "other.log"
    other.write_text("Error: address already in use\n")
    assert not server._log_mentions_db_error(other)


def test_start_resets_stale_db_and_retries_once(server: PrefectServer, monkeypatch: pytest.MonkeyPatch) -> None:
    db = server._sqlite_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_bytes(b"stale")

    calls: list[Path] = []

    def fake_spawn(cmd: list[str], env: dict[str, str], log_file: Path) -> None:
        calls.append(log_file)
        if len(calls) == 1:
            log_file.write_text("ResolutionError: No such revision or branch 'a1b2c3'")
            raise RuntimeError("Prefect server exited during startup (code 1)")

    monkeypatch.setattr(server, "is_running", lambda: False)
    monkeypatch.setattr(server, "_spawn_and_wait", fake_spawn)

    server.start()

    assert len(calls) == 2  # initial attempt + single retry
    assert not db.exists()
    backups = list(db.parent.glob("prefect.db.bak-*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"stale"


def test_start_raises_without_retry_on_non_db_errors(server: PrefectServer, monkeypatch: pytest.MonkeyPatch) -> None:
    db = server._sqlite_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_bytes(b"stale")

    calls: list[Path] = []

    def fake_spawn(cmd: list[str], env: dict[str, str], log_file: Path) -> None:
        calls.append(log_file)
        log_file.write_text("Error: address already in use")
        raise RuntimeError("Prefect server exited during startup (code 1)")

    monkeypatch.setattr(server, "is_running", lambda: False)
    monkeypatch.setattr(server, "_spawn_and_wait", fake_spawn)

    with pytest.raises(RuntimeError, match="exited during startup"):
        server.start()

    assert len(calls) == 1  # no retry
    assert db.exists()  # DB untouched
