"""Unit tests for genai_tk.tools.browser.session_manager."""

from __future__ import annotations

import json
import time
from pathlib import Path

from genai_tk.tools.browser.models import AuthConfig, SessionConfig
from genai_tk.tools.browser.session_manager import _EXPIRY_BUFFER_SECONDS, SessionManager


def _make_auth(path: str) -> AuthConfig:
    return AuthConfig(session=SessionConfig(storage_state_path=path, check_validity=True))


def _write_state(path: Path, cookies: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cookies": cookies, "origins": []}))


class TestHasValidSession:
    def test_returns_false_when_check_disabled(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        _write_state(state_path, [{"name": "tok", "expires": time.time() + 3600}])
        auth = AuthConfig(session=SessionConfig(storage_state_path=str(state_path), check_validity=False))
        assert SessionManager.has_valid_session(auth, "test") is False

    def test_returns_false_when_file_missing(self, tmp_path: Path) -> None:
        auth = _make_auth(str(tmp_path / "no_such_file.json"))
        assert SessionManager.has_valid_session(auth, "test") is False

    def test_returns_true_for_valid_future_cookies(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        future = time.time() + 3600  # 1 hour from now
        _write_state(state_path, [{"name": "session_token", "expires": future}])
        auth = _make_auth(str(state_path))
        assert SessionManager.has_valid_session(auth, "test") is True

    def test_returns_false_for_expired_cookie(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        past = time.time() - 10  # already expired
        _write_state(state_path, [{"name": "session_token", "expires": past}])
        auth = _make_auth(str(state_path))
        assert SessionManager.has_valid_session(auth, "test") is False

    def test_returns_false_for_soon_expiring_cookie(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        # Expires within the buffer window
        soon = time.time() + _EXPIRY_BUFFER_SECONDS - 5
        _write_state(state_path, [{"name": "tok", "expires": soon}])
        auth = _make_auth(str(state_path))
        assert SessionManager.has_valid_session(auth, "test") is False

    def test_session_cookie_no_expiry_treated_as_valid(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        # expires: -1 means session cookie (no expiry set)
        _write_state(state_path, [{"name": "session_tok", "expires": -1}])
        auth = _make_auth(str(state_path))
        assert SessionManager.has_valid_session(auth, "test") is True

    def test_returns_false_for_malformed_json(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        state_path.write_text("not valid json {{{")
        auth = _make_auth(str(state_path))
        assert SessionManager.has_valid_session(auth, "test") is False

    def test_returns_false_for_empty_cookies(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        _write_state(state_path, [])
        auth = _make_auth(str(state_path))
        assert SessionManager.has_valid_session(auth, "test") is False


class TestStatePath:
    def test_name_substitution(self, tmp_path: Path) -> None:
        auth = _make_auth("data/sessions/{name}_session.json")
        path = SessionManager.state_path(auth, "enedis_production")
        assert path == Path("data/sessions/enedis_production_session.json")

    def test_no_placeholder(self, tmp_path: Path) -> None:
        auth = _make_auth(str(tmp_path / "fixed.json"))
        path = SessionManager.state_path(auth, "any_name")
        assert path == tmp_path / "fixed.json"


class TestDeleteSession:
    def test_deletes_existing_file(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        _write_state(state_path, [])
        auth = _make_auth(str(state_path))
        SessionManager.delete_session(auth, "test")
        assert not state_path.exists()

    def test_no_error_when_file_absent(self, tmp_path: Path) -> None:
        auth = _make_auth(str(tmp_path / "absent.json"))
        # Should not raise
        SessionManager.delete_session(auth, "test")
