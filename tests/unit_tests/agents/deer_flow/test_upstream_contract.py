"""Contract test guarding the private upstream ``deerflow-harness`` surface.

``deerflow-harness`` is a third-party package (github.com/bytedance/deer-flow,
pinned in ``pyproject.toml`` under ``[tool.uv.sources]``) that genai-tk's
harness layer reaches into *beyond its public API* — specifically the private
``DeerFlowClient._agent`` / ``_ensure_agent`` / ``_agent_config_key`` used to
expose the compiled LangGraph graph via ``DeerFlowHarness.get_graph()``.

This test does not exercise behaviour; it asserts the *shape* we depend on
still exists after any bump of the pinned commit. If it fails, the private
surface changed upstream and ``EmbeddedDeerFlowClient``/``DeerFlowHarness``
graph-access code must be re-verified before moving the pin.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("deerflow", reason="deerflow-harness not installed")


@pytest.mark.unit
def test_deerflow_client_constructor_accepts_checkpointer_and_middlewares() -> None:
    """``EmbeddedDeerFlowClient`` builds its own checkpointer and passes it in —
    this must keep working so genai-tk keeps checkpointer ownership."""
    from deerflow.client import DeerFlowClient

    params = inspect.signature(DeerFlowClient.__init__).parameters
    assert "checkpointer" in params
    assert "middlewares" in params
    assert "available_skills" in params


@pytest.mark.unit
def test_deerflow_client_has_private_agent_surface() -> None:
    """The private surface ``DeerFlowHarness.get_graph()`` depends on.

    ``_ensure_agent(config, ...)`` must exist and accept a ``config`` mapping
    with a ``configurable`` key; the compiled graph is then read back from
    ``_agent``. If this method's name, arity, or the ``_agent`` attribute
    disappears, our graph accessor breaks silently.
    """
    from deerflow.client import DeerFlowClient

    assert hasattr(DeerFlowClient, "_ensure_agent")
    sig = inspect.signature(DeerFlowClient._ensure_agent)
    assert "config" in sig.parameters


@pytest.mark.unit
def test_deerflow_client_init_stores_model_name() -> None:
    """``EmbeddedDeerFlowClient.get_graph()`` reads ``self._client._model_name``
    as a fallback default — must stay the attribute name upstream uses."""
    import inspect as _inspect

    from deerflow.client import DeerFlowClient

    src = _inspect.getsource(DeerFlowClient.__init__)
    assert "self._model_name = model_name" in src


@pytest.mark.unit
def test_deerflow_client_stream_is_a_sync_generator_function() -> None:
    """``stream()`` stays a sync generator — confirms the thread/queue bridge
    in ``EmbeddedDeerFlowClient.stream_message`` is still necessary/correct."""
    from deerflow.client import DeerFlowClient

    assert inspect.isgeneratorfunction(DeerFlowClient.stream)


@pytest.mark.unit
def test_trace_context_helpers_are_public() -> None:
    """Public trace-correlation helpers we must reuse if we ever stream the
    compiled graph directly (bypassing ``DeerFlowClient.stream()``), so
    Langfuse trace correlation is not silently lost."""
    from deerflow import trace_context

    for name in (
        "get_current_trace_id",
        "generate_trace_id",
        "set_current_trace_id",
        "reset_current_trace_id",
        "is_trace_id_from_request_header",
    ):
        assert hasattr(trace_context, name), f"deerflow.trace_context.{name} missing"


@pytest.mark.unit
def test_create_deerflow_agent_signature_accepts_checkpointer() -> None:
    """SDK-level factory referenced by the design doc; not used for graph
    exposure (see docs/design), but guarded here in case a future phase adopts it."""
    from deerflow.agents import create_deerflow_agent

    sig = inspect.signature(create_deerflow_agent)
    assert "checkpointer" in sig.parameters
