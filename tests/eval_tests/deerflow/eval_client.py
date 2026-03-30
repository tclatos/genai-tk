"""Evaluation harness for Deer-flow agents.

Wraps ``EmbeddedDeerFlowClient`` with helpers that collect streaming events
into the formats expected by agentevals (OpenAI-format trajectory dicts) and
openevals (plain text responses).

Usage:
    ```python
    from tests.eval_tests.deerflow.eval_client import DeerFlowEvalClient

    client = DeerFlowEvalClient(mode="flash")

    # Plain text response
    answer = asyncio.run(client.arun("What is 2 + 2?"))

    # Full trajectory (OpenAI message-dict format)
    trajectory = asyncio.run(client.acollect_trajectory("Execute: sum(range(10))"))

    # Sync wrapper for run_multiturn_simulation
    response = client.run_sync("What is 2 + 2?", thread_id="my-thread")
    ```
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import uuid
from typing import Any


class DeerFlowEvalClient:
    """Thin evaluation harness wrapping ``EmbeddedDeerFlowClient``.

    Calls ``setup_deer_flow_config()`` once at construction time (writes
    ``config.yaml`` + ``extensions_config.json`` to ``DEER_FLOW_PATH/backend``),
    then exposes helpers that drive the embedded client and translate its
    streaming events into formats consumed by agentevals / openevals.
    """

    def __init__(
        self,
        mode: str = "flash",
        model_name: str | None = None,
        mcp_servers: list[str] | None = None,
    ) -> None:
        """Initialize the evaluation client.

        Args:
            mode: Deer-flow reasoning mode (flash | thinking | pro | ultra).
                Use ``flash`` for eval tests — fastest, no planner overhead.
            model_name: Resolved genai-tk LLM ID (e.g. ``"fast_model"``). When
                ``None``, resolves ``fast_model`` from the project config —
                the same model used as judge LLM, guaranteed to support
                tool-calling (required by deer-flow's agent loop).
            mcp_servers: MCP server names to enable. ``None`` / ``[]`` disables
                all MCP servers (avoids network deps in most eval tests).
        """
        from genai_tk.agents.deer_flow.config_bridge import setup_deer_flow_config
        from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient
        from genai_tk.core.llm_factory import LlmFactory

        if model_name is None:
            resolved, err = LlmFactory.resolve_llm_identifier_safe("fast_model")
            if err or resolved is None:
                raise RuntimeError(f"Could not resolve 'fast_model' for DeerFlowEvalClient: {err}")
            model_name = resolved

        config_path, _ = setup_deer_flow_config(
            mcp_server_names=mcp_servers or [],
            sandbox="local",
            selected_llm=model_name,
        )
        self._client = EmbeddedDeerFlowClient(
            config_path=config_path,
            model_name=model_name,
        )
        self._mode = mode

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def arun(self, query: str, *, thread_id: str | None = None) -> str:
        """Run a query and return the final text response.

        Args:
            query: User query string.
            thread_id: Thread ID for checkpointer-backed multi-turn context.
                Auto-generated (one-shot) when omitted.

        Returns:
            Final AI text response stripped of reasoning markers,
            or empty string if the agent emits no text.
        """
        from genai_tk.agents.deer_flow.embedded_client import ErrorEvent, TokenEvent, strip_reasoning_markers

        tid = thread_id or str(uuid.uuid4())
        last_text = ""
        async for event in self._client.stream_message(tid, query, mode=self._mode):
            if isinstance(event, TokenEvent) and event.data:
                last_text = event.data
            elif isinstance(event, ErrorEvent):
                raise RuntimeError(f"DeerFlow stream error (arun): {event.message!r}")
        return strip_reasoning_markers(last_text)

    async def acollect_trajectory(self, query: str, *, thread_id: str | None = None) -> list[dict[str, Any]]:
        """Run a query and collect the full trajectory in OpenAI message format.

        Translates stream events into trajectory dicts matching the format
        expected by ``agentevals.trajectory.match.create_trajectory_match_evaluator``:

        - ``ToolCallEvent``   → assistant message with ``"tool_calls"``
        - ``ToolResultEvent`` → tool message
        - ``TokenEvent``      → assistant text (accumulated, emitted on flush)

        Args:
            query: User query string.
            thread_id: Thread ID for multi-turn context. Auto-generated when omitted.

        Returns:
            List of OpenAI-format message dicts starting with the user message.
        """
        from genai_tk.agents.deer_flow.embedded_client import (
            ErrorEvent,
            TokenEvent,
            ToolCallEvent,
            ToolResultEvent,
        )

        tid = thread_id or str(uuid.uuid4())
        trajectory: list[dict[str, Any]] = [{"role": "user", "content": query}]
        text_parts: list[str] = []

        def _flush_text() -> None:
            if text_parts:
                trajectory.append({"role": "assistant", "content": "".join(text_parts)})
                text_parts.clear()

        async for event in self._client.stream_message(tid, query, mode=self._mode):
            if isinstance(event, ToolCallEvent):
                # Flush any preceding text (rare, but possible with some models)
                _flush_text()
                trajectory.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": event.tool_name,
                                    "arguments": json.dumps(event.args),
                                },
                                "id": event.call_id,
                            }
                        ],
                    }
                )
            elif isinstance(event, ToolResultEvent):
                trajectory.append(
                    {
                        "role": "tool",
                        "content": event.content,
                        "name": event.tool_name,
                    }
                )
            elif isinstance(event, TokenEvent) and event.data:
                text_parts.append(event.data)
            elif isinstance(event, ErrorEvent):
                raise RuntimeError(f"DeerFlow stream error (trajectory): {event.message!r}")

        _flush_text()
        return trajectory

    # ------------------------------------------------------------------
    # Sync API (for run_multiturn_simulation)
    # ------------------------------------------------------------------

    def run_sync(self, query: str, *, thread_id: str | None = None) -> str:
        """Synchronous wrapper around ``arun`` for use as a simulation ``app``.

        Runs the async agent in a dedicated ``ThreadPoolExecutor`` thread so it
        can be called from a synchronous context without interfering with any
        outer event loop.

        Args:
            query: User query string.
            thread_id: Thread ID for multi-turn context. The same ``thread_id``
                should be reused across turns so the checkpointer can maintain
                conversation state.

        Returns:
            Final AI text response.
        """

        def _run() -> str:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.arun(query, thread_id=thread_id))
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run).result(timeout=120)
