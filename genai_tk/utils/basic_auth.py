"""Basic authentication module for the application.

Provides functionality for:
- Password hashing and verification
- User authentication against a YAML config file
- Session management for Streamlit

Passwords are hashed with bcrypt (salted, slow KDF). For backward
compatibility, legacy unsalted SHA-256 hashes (64 hex characters) are still
verified with a constant-time comparison, so existing ``password_hash`` entries
keep working until they are re-hashed via :func:`hash_password`.
"""

from __future__ import annotations

import hashlib
import secrets

import bcrypt
import yaml
from pydantic import BaseModel, ConfigDict

from genai_tk.config_mgmt.config_mngr import global_config

# Legacy SHA-256 hashes are 64 lowercase hex characters; bcrypt hashes start
# with "$2" (e.g. "$2b$..."). Used to pick the verifier in :func:`verify_password`.
_BCRYPT_PREFIX = "$2"


class User(BaseModel):
    """User model for authentication."""

    username: str
    password_hash: str


class AuthConfig(BaseModel):
    """Authentication configuration."""

    enabled: bool = False
    config_file: str | None = None
    users: list[User] = []
    model_config = ConfigDict(extra="ignore")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt with a random salt.

    Args:
        password: The plain text password to hash.

    Returns:
        The bcrypt hash as an ASCII string (e.g. ``$2b$12$...``).
    """
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def _is_legacy_sha256(hashed_password: str) -> bool:
    """Return True when *hashed_password* is a legacy unsalted SHA-256 digest."""
    return not hashed_password.startswith(_BCRYPT_PREFIX)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a stored hash in constant time.

    Bcrypt hashes are checked with :func:`bcrypt.checkpw` (constant-time).
    Legacy unsalted SHA-256 hashes are checked with
    :func:`secrets.compare_digest` to avoid a timing side-channel.

    Args:
        plain_password: The plain text password to verify.
        hashed_password: The stored hash to check against.

    Returns:
        True if the password matches, False otherwise (also on malformed hashes).
    """
    if _is_legacy_sha256(hashed_password):
        legacy = hashlib.sha256(plain_password.encode("utf-8")).hexdigest()
        return secrets.compare_digest(legacy, hashed_password)
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("ascii"))
    except (ValueError, TypeError):
        return False


def load_auth_config() -> AuthConfig:
    """Load authentication configuration from the config file.

    Returns:
        The authentication configuration
    """
    try:
        auth = global_config().section("auth", AuthConfig)
    except Exception:
        return AuthConfig()

    if not auth.enabled or not auth.config_file:
        return auth

    from pathlib import Path

    config_path = Path(auth.config_file)
    if not config_path.exists():
        return AuthConfig(enabled=False, users=[])

    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        if not config_data:
            return AuthConfig(enabled=False, users=[])
        config_data["enabled"] = auth.enabled
        return AuthConfig.model_validate(config_data)
    except Exception:
        return AuthConfig(enabled=False, users=[])


def authenticate(username: str, password: str) -> bool:
    """Authenticate a user against the config file.

    Args:
        username: The username to authenticate
        password: The plain text password to verify

    Returns:
        True if authentication is successful, False otherwise
    """
    auth_config = load_auth_config()

    # If authentication is disabled, always return True
    if not auth_config.enabled:
        return True

    # Find the user in the config
    user = next((u for u in auth_config.users if u.username == username), None)
    if not user:
        return False

    # Verify the password
    return verify_password(password, user.password_hash)


def is_authenticated() -> bool:
    """Check if the current session is authenticated.

    Returns:
        True if authenticated, False otherwise
    """
    import streamlit as st

    # If authentication is disabled, always return True
    auth_config = load_auth_config()
    if not auth_config.enabled:
        return True

    # Check if the user is authenticated in the session
    return st.session_state.get("authenticated", False)
