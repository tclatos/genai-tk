"""Shared fixtures for Deer-flow evaluation tests.

All tests in this sub-package require:
  1. ``DEER_FLOW_PATH`` env var pointing to a cloned deer-flow repo.
  2. ``--include-real-models`` pytest flag (deer-flow always uses a real LLM).

Both conditions are enforced via session-scoped fixtures so a single clear
``pytest.skip`` message appears rather than N individual test skips.
"""

from __future__ import annotations

import os
import uuid

import pytest

# ---------------------------------------------------------------------------
# Session-level skip guards
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _require_deer_flow_path():
    """Skip the entire session if DEER_FLOW_PATH is not configured."""
    if not os.environ.get("DEER_FLOW_PATH"):
        pytest.skip(
            "DEER_FLOW_PATH is not set — deer-flow eval tests are skipped. "
            "Clone deer-flow and set DEER_FLOW_PATH to its root."
        )


@pytest.fixture(scope="session")
def _require_real_models(request):
    """Skip the entire session if --include-real-models is not passed."""
    if not request.config.getoption("--include-real-models", default=False):
        pytest.skip("Deer-flow eval tests require --include-real-models (uses a real LLM).")


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def deerflow_client(_require_deer_flow_path, _require_real_models):
    """Session-scoped ``DeerFlowEvalClient`` in ``flash`` mode.

    ``flash`` mode has no thinking or planning overhead and is the fastest
    option for evaluation tests.  Config is written once for the whole session.
    """
    from tests.eval_tests.deerflow.eval_client import DeerFlowEvalClient

    return DeerFlowEvalClient(mode="flash")


@pytest.fixture
def deerflow_agent_app(deerflow_client):
    """Sync callable wrapping ``deerflow_client`` for ``run_multiturn_simulation``.

    A stable ``thread_id`` is created per test so all turns in one simulation
    share the same deer-flow checkpointer state (multi-turn memory).
    """
    thread_id = str(uuid.uuid4())

    def _app(message: dict, **_) -> dict:
        content = message.get("content", "") if isinstance(message, dict) else str(message)
        response = deerflow_client.run_sync(content, thread_id=thread_id)
        return {"role": "assistant", "content": response}

    return _app


@pytest.fixture
def judge_llm(request):
    """Return the configured ``fast_model`` for use as an openevals judge.

    Skips automatically if ``--include-real-models`` is not set (though in
    this sub-package the session guard already handles that).
    """
    if not request.config.getoption("--include-real-models", default=False):
        pytest.skip("LLM-judged evals require --include-real-models")

    from genai_tk.core.llm_factory import get_llm

    return get_llm("fast_model")
