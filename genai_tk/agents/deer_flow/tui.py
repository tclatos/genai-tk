"""Launch the DeerFlow terminal workbench (TUI) for a genai-tk DeerFlow profile.

Deer-flow 2.1.0 ships a Textual TUI (``deerflow.tui``) that runs an embedded
``DeerFlowClient``. This module bridges a genai-tk ``DeerFlowProfile`` to that
TUI: it prepares the profile (writes ``config.yaml``, resolves the model,
validates sandbox/MCP) using the same runtime helpers as ``cli agents run``,
then bakes the profile's reasoning mode into the ``DeerFlowClient``
constructor — the TUI drives ``client.stream()`` directly and only overrides
``model_name`` per turn, so ``thinking_enabled`` / ``plan_mode`` /
``subagent_enabled`` must be set at construction time rather than per stream.

Exposed through ``cli agents tui <profile>`` (see
:mod:`genai_tk.agents.harness.commands`). ``textual`` is a core genai-tk
dependency; only the ``harnessing`` extra (deerflow-harness) is optional, gated
by :func:`require_feature`.

Example:
    ```python
    from genai_tk.agents.deer_flow.tui import run_deerflow_tui

    run_deerflow_tui("simple-deerflow")
    ```
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from genai_tk.agents.deer_flow.embedded_client import (
    _build_checkpointer,
    _mode_flags,
    _prepare_deer_flow_environment,
)
from genai_tk.config_mgmt.features import require_feature

if TYPE_CHECKING:
    from genai_tk.agents.deer_flow.profile import DeerFlowProfile


def build_tui_client(
    profile: DeerFlowProfile,
    model_name: str | None,
    config_path: Path,
    *,
    checkpointer: Any = None,
    middlewares: list | None = None,
    available_skills: set[str] | None = None,
) -> Any:
    """Construct a ``DeerFlowClient`` for the TUI with the profile mode baked in.

    Unlike :class:`EmbeddedDeerFlowClient` (which forwards mode flags
    per-stream via ``stream_message``), the TUI calls ``client.stream()``
    directly and only overrides ``model_name`` per turn — so
    ``thinking_enabled`` / ``plan_mode`` / ``subagent_enabled`` must be set on
    the constructor. Only ``middlewares`` / ``available_skills`` are gated on
    the installed ``DeerFlowClient.__init__`` signature (they vary across
    deer-flow versions); the core + mode-flag params ship with the TUI itself.

    Args:
        profile: Resolved DeerFlow profile (mode/sandbox already applied).
        model_name: Resolved model ID, or ``None`` for the config default.
        config_path: Path to the deer-flow ``config.yaml`` written by
            :func:`setup_deer_flow_config`.
        checkpointer: LangGraph checkpointer for multi-turn memory (e.g. from
            :func:`_build_checkpointer`). ``None`` lets DeerFlow pick its own.
        middlewares: Instantiated middleware objects to inject.
        available_skills: Optional skill-name allowlist; ``None`` = all skills.

    Returns:
        A ``deerflow.client.DeerFlowClient`` instance.
    """
    _prepare_deer_flow_environment()

    from deerflow.client import DeerFlowClient  # type: ignore[import]

    flags = _mode_flags(profile.mode)
    supported = set(inspect.signature(DeerFlowClient.__init__).parameters)
    kwargs: dict[str, Any] = {
        "config_path": str(config_path),
        "checkpointer": checkpointer,
        "model_name": model_name,
        "thinking_enabled": flags["thinking_enabled"],
        "plan_mode": profile.plan_mode,
        "subagent_enabled": profile.subagent_enabled,
    }
    if "middlewares" in supported:
        kwargs["middlewares"] = middlewares or []
    elif middlewares:
        logger.warning(
            "This deer-flow version does not support the 'middlewares' parameter — ignoring. "
            "Update your deer-flow clone to enable middleware injection."
        )
    if "available_skills" in supported:
        kwargs["available_skills"] = available_skills
    elif available_skills is not None:
        logger.warning(
            "This deer-flow version does not support the 'available_skills' parameter — ignoring. "
            "Update your deer-flow clone to enable skill filtering."
        )
    return DeerFlowClient(**kwargs)


def run_deerflow_tui(
    profile_name: str,
    *,
    llm_override: str | None = None,
    mode_override: str | None = None,
    sandbox_override: str | None = None,
    extra_mcp: list[str] | None = None,
    message: str | None = None,
    thread_id: str | None = None,
    continue_recent: bool = False,
    verbose: bool = False,
) -> int:
    """Prepare a DeerFlow profile and launch its terminal workbench.

    Mirrors the first half of ``cli agents run`` (profile prep via
    :func:`prepare_profile`) then launches the DeerFlow Textual TUI with the
    profile's mode baked into the client. The TUI runs an embedded
    ``DeerFlowClient`` — no Gateway, frontend, or Docker services required.

    Args:
        profile_name: DeerFlow profile key/name to launch.
        llm_override: LLM identifier override (ID or tag).
        mode_override: Reasoning mode override (``flash`` | ``thinking`` |
            ``pro`` | ``ultra``); ``None`` keeps the profile's mode.
        sandbox_override: Sandbox override (``local`` | ``docker``).
        extra_mcp: Additional MCP server names appended to the profile.
        message: Optional initial prompt sent on mount (``None`` = empty
            composer).
        thread_id: Resume a thread by id/title (``--resume``).
        continue_recent: Resume the most recent thread (``--continue``).
        verbose: Enable DEBUG-level logging on stderr.

    Returns:
        Process exit code (``0`` on clean exit).
    """
    import asyncio

    require_feature("harnessing", context="cli agents tui")

    from genai_tk.agents.deer_flow.runtime import build_cli_middlewares, prepare_profile

    profile, model_name, config_path, _warnings = asyncio.run(
        prepare_profile(
            profile_name=profile_name,
            llm_override=llm_override,
            extra_mcp=list(extra_mcp or []),
            mode_override=mode_override,
            verbose=verbose,
            sandbox_override=sandbox_override,
        )
    )

    middlewares = build_cli_middlewares(profile.middlewares)
    available_skills = set(profile.available_skills) if profile.available_skills is not None else None
    checkpointer = _build_checkpointer()
    client = build_tui_client(
        profile,
        model_name,
        config_path,
        checkpointer=checkpointer,
        middlewares=middlewares,
        available_skills=available_skills,
    )

    from deerflow.tui.app import DeerFlowTUI  # type: ignore[import]
    from deerflow.tui.cli import LaunchPlan  # type: ignore[import]
    from deerflow.tui.session import Session  # type: ignore[import]

    # No threads_meta writer: genai-tk does not ship the DeerFlow Web UI, so the
    # shared-persistence writer is unnecessary. The SqliteSaver checkpointer
    # still gives multi-turn memory within this TUI session.
    session = Session(client=client)
    plan = LaunchPlan(
        mode="tui",
        message=message,
        thread_id=thread_id,
        continue_recent=continue_recent,
    )
    app = DeerFlowTUI(session, plan)
    try:
        app.run()
    finally:
        session.close()
    return 0
