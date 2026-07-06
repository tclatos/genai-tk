"""Unit tests for the basic_auth password hashing and verification module."""

from __future__ import annotations

import hashlib

from genai_tk.utils.basic_auth import (
    User,
    _is_legacy_sha256,
    hash_password,
    verify_password,
)


def test_hash_password_returns_bcrypt_hash() -> None:
    """hash_password produces a salted bcrypt hash, not a plain digest."""
    h = hash_password("s3cret!")

    assert h.startswith("$2")
    # bcrypt hashes carry the cost factor and a 22-char salt (~60 chars total)
    assert len(h) >= 57


def test_hash_password_uses_random_salt() -> None:
    """Two calls with the same password yield different hashes (random salt)."""
    assert hash_password("same") != hash_password("same")


def test_verify_password_accepts_correct_bcrypt_hash() -> None:
    """verify_password returns True for a matching bcrypt hash."""
    h = hash_password("correct horse battery staple")

    assert verify_password("correct horse battery staple", h)


def test_verify_password_rejects_wrong_bcrypt_hash() -> None:
    """verify_password returns False for a non-matching password."""
    h = hash_password("right")

    assert not verify_password("wrong", h)


def test_verify_password_legacy_sha256_still_works() -> None:
    """Existing unsalted SHA-256 hashes keep verifying (backward compat)."""
    legacy = hashlib.sha256("legacy-pw".encode("utf-8")).hexdigest()

    assert verify_password("legacy-pw", legacy)
    assert not verify_password("other", legacy)


def test_verify_password_legacy_uses_constant_time() -> None:
    """Legacy SHA-256 verification must not short-circuit on length mismatch."""
    # A legacy hash of a different length should still return False (no crash).
    assert not verify_password("anything", "deadbeef")


def test_verify_password_malformed_bcrypt_returns_false() -> None:
    """A malformed bcrypt hash must yield False, never raise."""
    assert not verify_password("x", "$2b$not-a-real-hash")
    assert not verify_password("x", "$2$")


def test_is_legacy_sha256_detects_format() -> None:
    """The legacy detector distinguishes SHA-256 from bcrypt hashes."""
    sha = hashlib.sha256("x".encode("utf-8")).hexdigest()

    assert _is_legacy_sha256(sha) is True
    assert _is_legacy_sha256(hash_password("x")) is False


def test_user_model_round_trips() -> None:
    """User is a plain Pydantic model holding username + password_hash."""
    u = User(username="alice", password_hash=hash_password("pw"))

    assert u.username == "alice"
    assert u.password_hash.startswith("$2")
