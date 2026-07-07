"""Runtime helpers for DeerFlow — pure logic with no CLI/console coupling.

This module holds the harness- and runtime-level DeerFlow helpers (config path
resolution, profile preparation, middleware instantiation, sandbox checks)
that are safe to call from non-CLI callers (the harness adapter, the registry,
tests). It is deliberately free of ``rich``/``typer`` so it does not invert the
dependency direction (CLI depends on runtime, never the reverse).

CLI presentation (console output, exit codes) lives in the unified harness
commands (``genai_tk.agents.harness.commands``), which call into this runtime
module and render any ``DeerFlowError`` / ``DeerFlowNotInstalledError`` to the
terminal.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, cast

from loguru import logger

from genai_tk.config_mgmt.config_mngr import global_config

if TYPE_CHECKING:
    from genai_tk.agents.deer_flow.profile import DeerFlowProfile, DeerFlowSandbox


class DeerFlowNotInstalledError(RuntimeError):
    """Raised when the 'harnessing' feature (deerflow-harness) is not installed."""


def require_deer_flow_installed() -> None:
    """Raise :class:`DeerFlowNotInstalledError` if deerflow-harness is not installed.

    deerflow-harness is a regular Python package installed via ``uv add`` (no
    ``DEER_FLOW_PATH`` clone). Callers (CLI) translate this into an exit code
    and an install hint.
    """
    from genai_tk.config_mgmt.features import is_available

    if not is_available("harnessing"):
        raise DeerFlowNotInstalledError(
            "Feature 'harnessing' is not installed. Install with: uv sync --extra harnessing"
        )


def get_default_profile_name() -> str | None:
    """Return the configured DeerFlow default profile, else the first one, or ``None``."""
    try:
        from genai_tk.agents.harness.profiles import load_deerflow_profiles

        # Prefer the configured ``deerflow.default_profile`` global setting.
        try:
            configured = global_config().get("deerflow.default_profile")
        except Exception:
            configured = None
        if configured:
            return configured
        profiles = load_deerflow_profiles()
        return profiles[0].name if profiles else None
    except Exception as e:
        logger.debug(f"Could not resolve default DeerFlow profile: {e}")
        return None


def resolve_model_name(llm_identifier: str) -> str:
    """Resolve a genai-tk LLM identifier to a deer-flow model_name string.

    Args:
        llm_identifier: LLM ID, tag, or compact alias (e.g. ``gpt_41mini@openai``).

    Returns:
        Resolved LLM ID string.

    Raises:
        ValueError: If the identifier cannot be resolved.
    """
    from genai_tk.core.factories.llm_factory import LlmFactory

    llm_id, error_msg = LlmFactory.resolve_llm_identifier_safe(llm_identifier)
    if error_msg or llm_id is None:
        raise ValueError(error_msg or "Could not resolve model identifier")
    return llm_id


def validate_and_normalize_sandbox(sandbox: str) -> DeerFlowSandbox:
    """Validate and normalize a sandbox provider string.

    Args:
        sandbox: Raw sandbox string from the profile.

    Returns:
        Normalized sandbox string (``"local"`` or ``"docker"``).

    Raises:
        ValueError: If the value is neither ``local`` nor ``docker``.
    """
    normalized = (sandbox or "").strip().lower() or "local"
    if normalized not in {"local", "docker"}:
        raise ValueError(f"Invalid sandbox value: '{sandbox}'. Expected 'local' or 'docker'.")
    return cast("DeerFlowSandbox", normalized)


def check_docker_available() -> bool:
    """Return True if the ``docker`` CLI is present and the daemon is reachable."""
    from shutil import which

    if which("docker") is None:
        return False
    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True, timeout=3, check=False)
        return result.returncode == 0
    except Exception:
        return False


def check_agent_sandbox_importable() -> bool:
    """Return True if the ``agent-sandbox`` package is importable."""
    try:
        import agent_sandbox  # noqa: F401

        return True
    except ImportError:
        return False


def validate_docker_sandbox() -> None:
    """Raise :class:`DockerSandboxError` if Docker sandbox prerequisites are unmet.

    Checks that the ``docker`` CLI is reachable and that the ``agent-sandbox``
    package (used by DeerFlow's ``AioSandbox``) is installed.
    """
    from genai_tk.agents.deer_flow.profile import DockerSandboxError

    reasons: list[str] = []
    if not check_docker_available():
        reasons.append("Docker is not available (docker CLI not found or daemon not running)")
    if not check_agent_sandbox_importable():
        reasons.append("'agent-sandbox' package is not installed — install with: uv add agent-sandbox")
    if reasons:
        raise DockerSandboxError(reasons)


def verify_written_sandbox(config_path: Path, expected_sandbox: str) -> str | None:
    """Verify the generated deer-flow config.yaml matches the profile sandbox.

    Returns a warning string if the written sandbox provider does not match the
    expected value, or ``None`` when it matches / cannot be checked. The CLI
    layer is responsible for rendering the warning to the user.
    """
    try:
        import yaml

        raw = yaml.safe_load(config_path.read_text()) or {}
        use_str = ((raw.get("sandbox") or {}).get("use") or "").strip()
    except Exception as e:
        logger.debug(f"Could not verify sandbox provider in {config_path}: {e}")
        return None

    expected = expected_sandbox.lower()
    if expected == "docker" and "aio_sandbox_provider" not in use_str:
        return (
            f"Profile sandbox is 'docker' but generated config does not look like a Docker sandbox. "
            f"config.yaml={config_path} sandbox.use={use_str!r}"
        )
    if expected == "local" and "LocalSandboxProvider" not in use_str:
        return (
            f"Profile sandbox is 'local' but generated config does not look like LocalSandboxProvider. "
            f"config.yaml={config_path} sandbox.use={use_str!r}"
        )
    return None


# ---------------------------------------------------------------------------
# Config-setup warnings returned to the caller (rendered by CLI / workbench)
# ---------------------------------------------------------------------------


class SetupWarnings:
    """Collected warnings from ``prepare_profile`` for the caller to render.

    A lightweight, plain-Python container (no Pydantic — these are internal
    advisory fields, not a system boundary model). Mirrors the shape returned
    by ``setup_deer_flow_config``.
    """

    def __init__(self) -> None:
        self.missing_skill_directories: list[str] = []
        self.external_symlinks: list[str] = []

    @property
    def has_warnings(self) -> bool:
        return bool(self.missing_skill_directories or self.external_symlinks)


async def prepare_profile(
    profile_name: str,
    llm_override: str | None,
    extra_mcp: list[str],
    mode_override: str | None,
    verbose: bool,
    *,
    sandbox_override: str | None = None,
) -> tuple[DeerFlowProfile, str | None, Path, SetupWarnings]:
    """Load, validate and prepare a profile, then write the deer-flow config.

    Pure runtime counterpart of the old CLI ``_prepare_profile``: returns
    warnings through a :class:`SetupWarnings` object and raises
    :class:`DeerFlowError` / :class:`ValueError` / :class:`DockerSandboxError`
    on failure instead of printing or calling ``typer.Exit``.

    Writes ``config.yaml`` + ``extensions_config.json`` ready for use by the
    embedded client.

    Args:
        profile_name: Profile name from deerflow.yaml.
        llm_override: LLM identifier override (ID or tag).
        extra_mcp: Additional MCP server names.
        mode_override: Mode override string.
        verbose: Enable DEBUG-level logging on stderr.
        sandbox_override: Override sandbox type (None = use profile setting).

    Returns:
        Tuple of (prepared DeerFlowProfile, resolved model_name or None,
        config_path, SetupWarnings).
    """
    import sys

    from genai_tk.agents.deer_flow.config_bridge import setup_deer_flow_config
    from genai_tk.agents.deer_flow.profile import (
        validate_mcp_servers,
        validate_mode,
        validate_profile_name,
    )
    from genai_tk.agents.harness.profiles import load_deerflow_profiles

    if verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        )

    require_deer_flow_installed()

    profiles = load_deerflow_profiles()
    profile = validate_profile_name(profile_name, profiles)

    # Initialise all active monitoring backends (idempotent). Per-profile trace
    # project naming is owned by the harness layer (see
    # ``genai_tk.utils.tracing.apply_harness_trace_metadata``) so this function
    # no longer sets LANGSMITH_PROJECT itself.
    from genai_tk.utils.tracing import setup_monitoring

    setup_monitoring()

    if mode_override:
        profile.mode = validate_mode(mode_override)
    if extra_mcp:
        validated = validate_mcp_servers(extra_mcp)
        profile.mcp_servers = list(set(profile.mcp_servers + validated))
    if sandbox_override:
        profile.sandbox = sandbox_override
    profile.sandbox = validate_and_normalize_sandbox(profile.sandbox)

    model_name: str | None = None
    if llm_override:
        model_name = resolve_model_name(llm_override)
    elif profile.llm:
        model_name = resolve_model_name(profile.llm)

    config_path, _ext_path, setup_warnings_obj = setup_deer_flow_config(
        mcp_server_names=profile.mcp_servers,
        skill_directories=profile.skill_directories,
        sandbox=profile.sandbox,
        selected_llm=model_name,
    )
    _sandbox_warn = verify_written_sandbox(config_path, profile.sandbox)

    warnings = SetupWarnings()
    warnings.missing_skill_directories = list(getattr(setup_warnings_obj, "missing_skill_directories", []))
    warnings.external_symlinks = list(getattr(setup_warnings_obj, "external_symlinks", []))
    if _sandbox_warn:
        warnings.external_symlinks.append(_sandbox_warn)

    if profile.sandbox == "docker":
        validate_docker_sandbox()

    return profile, model_name, config_path, warnings


def build_cli_middlewares(
    profile_middlewares: list,
    *,
    rich_console: object | None = None,
) -> list:
    """Instantiate profile middlewares and prepend RichToolCallMiddleware.

    Mirrors the LangChain agent default: every run gets Rich tool-call tracing
    automatically, regardless of what the profile config lists.

    Args:
        profile_middlewares: List of :class:`MiddlewareConfig` entries.
        rich_console: Optional rich console to attach to the
            :class:`RichToolCallMiddleware`. ``None`` uses a fresh console.

    Returns:
        List of instantiated middleware objects.
    """
    from rich.console import Console

    from genai_tk.agents.langchain.config import instantiate_middlewares
    from genai_tk.agents.langchain.middleware.rich_middleware import RichToolCallMiddleware

    user_mws = instantiate_middlewares(profile_middlewares, "custom")
    if not any(isinstance(m, RichToolCallMiddleware) for m in user_mws):
        console = rich_console if rich_console is not None else Console()
        user_mws.insert(0, RichToolCallMiddleware(console=console))
    return user_mws


def stable_thread_id() -> str:
    """Return a deterministic thread ID for sandbox container reuse."""
    import hashlib

    return hashlib.sha256(b"genai-tk-deerflow-single").hexdigest()[:16]
