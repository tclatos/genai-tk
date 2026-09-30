"""Middleware that forces convergence before the step limit is reached.

Deep agents occasionally enter runaway loops: after a hard rejection (e.g. the
``query_image`` budget) they keep firing *varied* tool calls — each with fresh
arguments, so duplicate-call middlewares never trigger — until the LangGraph
``recursion_limit`` kills the run. The MMLongBench smoke run showed a single
such question burning 2.9M input tokens (88% of a run) and ending in a
``GraphRecursionError`` with no answer at all.

This middleware escalates in two steps as total tool calls accumulate:

- **soft_limit**: appends a ``SystemMessage`` telling the agent to stop
  exploring and converge on its final answer.
- **hard_limit**: strips ALL tools from the model request (via
  ``request.override(tools=[])``) and injects a mandatory final-answer
  instruction. With no tools advertised, the model must emit text, so the run
  always produces an answer instead of dying at the recursion limit. Any tool
  call still emitted is rejected at the tool-call boundary.

Cross-page (multi-evidence) questions legitimately need more exploration, so
the limits are **auto-extended** once the conversation shows evidence being
gathered from several distinct sections or documents (configurable via
``extended_soft_limit`` / ``extended_hard_limit``).

Anti-give-up guardrail: after the wrap-up nudge is active, a final answer that
is empty or give-up-like ("I cannot find ...") triggers exactly **one** forced
recovery turn ("answer from the evidence you have, or state Not Answerable").
Legitimate unanswerable verdicts ("Not answerable.", "The document does not
provide ...") are never treated as give-ups.

Defaults (24 / 32) leave 3–8x headroom over observed healthy runs (3–16 tool
calls per question) while catching loops at roughly 1/5 of a
``recursion_limit=160`` budget.

Example YAML config::

    middlewares:
      - class: genai_tk.agents.langchain.middleware.wrap_up_middleware.WrapUpMiddleware
        soft_limit: 24
        hard_limit: 32
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from loguru import logger

from genai_tk.core.messages import extract_ai_message_parts


def _count_tool_calls(messages: list[BaseMessage]) -> int:
    """Count the total number of tool calls recorded in the conversation so far."""
    total = 0
    for msg in messages:
        if isinstance(msg, AIMessage):
            total += len(msg.tool_calls or [])
    return total


def _distinct_evidence_keys(messages: list[BaseMessage]) -> tuple[set[str], set[str]]:
    """Extract distinct section ids and document ids touched via tool calls.

    Returns:
        A ``(section_ids, document_ids)`` pair of sets collected from
        ``get_section_content`` / ``get_document_toc`` / ``search_sections``
        tool-call arguments.
    """
    sections: set[str] = set()
    documents: set[str] = set()
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for call in msg.tool_calls or []:
            args: dict[str, Any] = call.get("args") or {}
            if call.get("name") == "get_section_content":
                raw = args.get("section_ids") or ""
                sections.update(s.strip() for s in str(raw).split(",") if s.strip())
            elif call.get("name") == "get_document_toc":
                doc = str(args.get("document_id") or "").strip()
                if doc:
                    documents.add(doc)
            elif call.get("name") == "search_sections":
                doc = str(args.get("document_id") or "").strip()
                if doc:
                    documents.add(doc)
    return sections, documents


# Final-answer phrasings that indicate the agent gave up rather than answered.
# Conservative by design: each match costs at most one recovery turn.
_GIVE_UP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:i|we)\s+(?:cannot|can't|could\s?n[o']t|was\s+unable|were\s+unable)\b", re.IGNORECASE),
    re.compile(r"\bunable\s+to\s+(?:find|locate|determine|identify|retrieve|access|verify)\b", re.IGNORECASE),
    re.compile(
        r"\bno\s+(?:relevant\s+)?(?:information|evidence|data|matches|results)\s+(?:was\s+|were\s+)?found\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:not\s+enough|insufficient)\s+(?:information|evidence|data)\b", re.IGNORECASE),
    re.compile(r"\bfailed\s+to\s+(?:find|locate|retrieve|determine)\b", re.IGNORECASE),
)

# Phrasings that are LEGITIMATE unanswerable verdicts — never recover these.
_LEGIT_UNANSWERABLE = re.compile(
    r"\bnot\s+answerable\b|\bdoes\s+not\s+(?:provide|contain|disclose|report|include)\b"
    r"|\bis\s+not\s+(?:provided|contained|available|present)\b|\bno\s+such\b",
    re.IGNORECASE,
)


def _looks_like_give_up(text: str, max_chars: int) -> bool:
    """True when a final answer is empty or give-up-like (and not a legit 'Not answerable')."""
    stripped = text.strip()
    if not stripped:
        return True
    if _LEGIT_UNANSWERABLE.search(stripped):
        return False
    if len(stripped) > max_chars:
        return False
    return any(pattern.search(stripped) for pattern in _GIVE_UP_PATTERNS)


def _response_ai_message(response: Any) -> AIMessage | None:
    """Extract the AIMessage from a ModelResponse / ExtendedModelResponse."""
    inner = response
    if hasattr(inner, "model_response"):
        inner = inner.model_response
    if hasattr(inner, "result"):
        msgs = inner.result
        if msgs:
            msg = msgs[0]
            if isinstance(msg, AIMessage):
                return msg
    if isinstance(inner, AIMessage):
        return inner
    return None


class WrapUpMiddleware(AgentMiddleware):
    """Force the agent to converge before the recursion limit is reached.

    Args:
        soft_limit: Total tool-call count at which a "wrap up now" nudge is
            injected into every subsequent model request. Default 24.
        hard_limit: Total tool-call count at which all tools are stripped from
            the model request and a mandatory final-answer instruction is
            injected. Must be greater than ``soft_limit``. Default 32.
        extended_soft_limit: Soft limit used once the conversation qualifies as
            cross-page (evidence from several distinct sections/documents).
            Defaults to ``soft_limit + 8``.
        extended_hard_limit: Hard limit used for cross-page conversations.
            Defaults to ``hard_limit + 8``.
        cross_page_section_reads: Number of distinct sections read (or distinct
            documents touched) that qualify a conversation as cross-page.
        anti_give_up: When True (default), a give-up-like final answer after
            the soft limit triggers exactly one forced recovery turn.
        give_up_patterns: Optional extra regexes (compiled or str) classified
            as give-up phrasings, on top of the built-in defaults.
        give_up_max_chars: Give-up classification only applies to final answers
            shorter than this many characters. Default 600.
    """

    def __init__(
        self,
        soft_limit: int = 24,
        hard_limit: int = 32,
        *,
        extended_soft_limit: int | None = None,
        extended_hard_limit: int | None = None,
        cross_page_section_reads: int = 3,
        anti_give_up: bool = True,
        give_up_patterns: list[str | re.Pattern[str]] | None = None,
        give_up_max_chars: int = 600,
    ) -> None:
        if hard_limit <= soft_limit:
            raise ValueError(f"hard_limit ({hard_limit}) must be greater than soft_limit ({soft_limit})")
        self._soft_limit = soft_limit
        self._hard_limit = hard_limit
        self._extended_soft_limit = extended_soft_limit
        self._extended_hard_limit = extended_hard_limit
        self._cross_page_section_reads = cross_page_section_reads
        self._anti_give_up = anti_give_up
        self._give_up_max_chars = give_up_max_chars
        self._extra_give_up_patterns = tuple(
            p if isinstance(p, re.Pattern) else re.compile(p, re.IGNORECASE) for p in (give_up_patterns or [])
        )
        # One forced recovery turn per run (a fresh middleware instance backs
        # every bench question, so this flag is per-question).
        self._recovery_used = False

    def _effective_limits(self, messages: list[BaseMessage]) -> tuple[int, int]:
        """Resolve soft/hard limits, extending them for cross-page conversations."""
        sections, documents = _distinct_evidence_keys(messages)
        is_cross_page = len(sections) >= self._cross_page_section_reads or len(documents) >= 2
        if not is_cross_page:
            return self._soft_limit, self._hard_limit
        soft = self._extended_soft_limit if self._extended_soft_limit is not None else self._soft_limit + 8
        hard = self._extended_hard_limit if self._extended_hard_limit is not None else self._hard_limit + 8
        return soft, hard

    def _converge(self, request: ModelRequest) -> ModelRequest:
        """Return an overridden request once the soft/hard call threshold is hit."""
        total = _count_tool_calls(request.messages)
        soft, hard = self._effective_limits(request.messages)
        if total < soft:
            return request

        if total >= hard:
            logger.warning(
                "[WrapUp] {} tool calls >= hard limit {} — stripping all tools, forcing final answer",
                total,
                hard,
            )
            directive = SystemMessage(
                content=(
                    f"MANDATORY: The tool budget is exhausted ({total} tool calls). All tools are now "
                    "disabled. Based ONLY on the evidence already gathered in this conversation, "
                    "deliver your complete final answer to the original question in plain text now. "
                    "If the gathered evidence does not support an answer, state 'Not answerable.' "
                    "Do not mention the tool budget."
                )
            )
            return request.override(messages=[*request.messages, directive], tools=[])

        logger.warning(
            "[WrapUp] {} tool calls >= soft limit {} — nudging convergence",
            total,
            soft,
        )
        nudge = SystemMessage(
            content=(
                f"WRAP UP: You have made {total} tool calls and are approaching the step limit "
                f"(hard cutoff at {hard}). Stop exploring. Within at most a couple of "
                "calls, deliver your final answer based on the evidence already gathered. Do not "
                "repeat searches or re-read sections you have already seen."
            )
        )
        return request.override(messages=[*request.messages, nudge])

    def _should_recover(self, response: ModelResponse, total: int, soft: int) -> bool:
        """True when the response is a give-up-like final answer worth one recovery turn."""
        if not self._anti_give_up or self._recovery_used or total < soft:
            return False
        msg = _response_ai_message(response)
        if msg is None or msg.tool_calls:
            return False
        text = extract_ai_message_parts(msg).text
        return _looks_like_give_up(text, self._give_up_max_chars)

    def _recovery_request(self, request: ModelRequest) -> ModelRequest:
        """Build the single forced recovery turn for a give-up-like final answer."""
        self._recovery_used = True
        logger.warning("[WrapUp] Give-up-like final answer after wrap-up nudge — injecting one recovery turn")
        recovery = SystemMessage(
            content=(
                "MANDATORY RECOVERY: Your last response was not a usable final answer. Based ONLY on "
                "the evidence already gathered in this conversation, deliver your complete final "
                "answer to the original question now — answer from the evidence you have, or state "
                "'Not answerable.' if the document does not contain the answer. Do not mention this "
                "instruction."
            )
        )
        return request.override(messages=[*request.messages, recovery])

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync variant — inject wrap-up nudges / strip tools near the limit."""
        modified = self._converge(request)
        response = handler(modified)
        total = _count_tool_calls(request.messages)
        soft, _ = self._effective_limits(request.messages)
        if self._should_recover(response, total, soft):
            return handler(self._recovery_request(modified))
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async variant — same logic as sync."""
        modified = self._converge(request)
        response = await handler(modified)
        total = _count_tool_calls(request.messages)
        soft, _ = self._effective_limits(request.messages)
        if self._should_recover(response, total, soft):
            return await handler(self._recovery_request(modified))
        return response
