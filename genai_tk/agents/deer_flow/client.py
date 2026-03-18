"""HTTP client for the Deer-flow LangGraph server.

Thin wrapper around ``langgraph_sdk.get_client`` that exposes the same
typed event model as :mod:`genai_tk.agents.deer_flow.embedded_client`.

Usage:
    ```python
    from genai_tk.agents.deer_flow.client import DeerFlowClient, TokenEvent

    client = DeerFlowClient()
    thread_id = await client.create_thread()
    async for event in client.stream_run(thread_id, "Explain RAG in one sentence."):
        if isinstance(event, TokenEvent):
            print(event.data, end="", flush=True)
    ```
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from langgraph_sdk import get_client

from genai_tk.agents.deer_flow.embedded_client import TokenEvent

__all__ = ["DeerFlowClient", "TokenEvent"]

_ASSISTANT_ID = "lead_agent"


class DeerFlowClient:
    """Async HTTP client for the Deer-flow LangGraph server.

    Connects to the server at ``http://localhost:2024`` by default.
    """

    def __init__(self, url: str = "http://localhost:2024") -> None:
        self._client = get_client(url=url)

    async def health_check(self) -> bool:
        """Return True if the LangGraph server responds to the /info endpoint."""
        try:
            await self._client.http.get("/info")
            return True
        except Exception:
            return False

    async def create_thread(self) -> str:
        """Create a new conversation thread and return its ID."""
        thread = await self._client.threads.create()
        return thread["thread_id"]

    async def stream_run(
        self,
        thread_id: str,
        message: str,
    ) -> AsyncGenerator[Any, None]:
        """Stream a run on the given thread, yielding typed events.

        Args:
            thread_id: Thread ID returned by ``create_thread()``.
            message: User message to send.

        Yields:
            ``TokenEvent`` instances for AI text tokens.
        """
        input_payload = {"messages": [{"role": "user", "content": message}]}
        async for part in self._client.runs.stream(
            thread_id,
            _ASSISTANT_ID,
            input=input_payload,
            stream_mode="messages",
            context={"thread_id": thread_id},
        ):
            event_type = getattr(part, "event", None)
            data = getattr(part, "data", None)
            if event_type in ("messages/partial", "messages/complete") and data:
                for msg in data if isinstance(data, list) else [data]:
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("type") == "ai":
                        content = msg.get("content", "")
                        if content:
                            yield TokenEvent(data=content)
