"""Abstract base class for a running agent harness session.

A :class:`BaseHarness` wraps exactly one agent runtime (LangChain or DeerFlow)
behind a single streaming interface, so callers (CLI, Streamlit) do not need
to special-case the underlying framework.

Subclasses:
    - :class:`~genai_tk.agents.harness.langchain_harness.LangChainHarness`
    - :class:`~genai_tk.agents.harness.deerflow_harness.DeerFlowHarness`
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from genai_tk.agents.harness.events import (
    ErrorEvent,
    HarnessModel,
    HarnessSkill,
    HarnessThread,
    StreamEvent,
    TokenEvent,
)


class BaseHarness(ABC):
    """Abstract base for a single agent harness session.

    Args:
        name: Harness discriminator (``"langchain"`` or ``"deerflow"``).
    """

    name: str

    #: Default thread ID used by :meth:`astream` when the caller passes
    #: ``thread_id=None``. Subclasses may override to a harness-specific
    #: stable value (e.g. for sandbox container reuse); the base default is
    #: a plain ``"default"`` so the magic literal does not leak into adapters.
    default_thread_id: str = "default"

    @abstractmethod
    def astream(self, message: str, *, thread_id: str | None = None) -> AsyncIterator[StreamEvent]:
        """Stream one conversation turn as canonical :class:`StreamEvent` objects.

        Args:
            message: User message text.
            thread_id: Conversation thread ID; ``None`` uses a harness-default thread.

        Yields:
            Typed :class:`StreamEvent` instances, ending with an ``EndEvent``.
        """
        raise NotImplementedError
        yield  # pragma: no cover - makes this an async generator for type checkers

    async def arun(self, message: str, *, thread_id: str | None = None) -> str:
        """Consume the stream and return the concatenated response text.

        Args:
            message: User message text.
            thread_id: Conversation thread ID; ``None`` uses a harness-default thread.

        Returns:
            The concatenated text of all ``TokenEvent`` chunks.
        """
        chunks: list[str] = []
        async for event in self.astream(message, thread_id=thread_id):
            if isinstance(event, TokenEvent):
                chunks.append(event.text)
            elif isinstance(event, ErrorEvent):
                raise RuntimeError(event.message)
        return "".join(chunks)

    async def list_threads(self) -> list[HarnessThread]:
        """Return persisted conversation threads, if the harness supports it."""
        return []

    async def list_models(self) -> list[HarnessModel]:
        """Return models available to this harness, if it supports enumeration."""
        return []

    async def list_skills(self) -> list[HarnessSkill]:
        """Return discoverable skills available to this harness, if any."""
        return []

    async def aclose(self) -> None:
        """Release any resources held by this harness session (sandboxes, connections)."""
        return None
