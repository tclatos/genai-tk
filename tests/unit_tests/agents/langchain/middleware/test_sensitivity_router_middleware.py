"""Unit tests for SensitivityRouterMiddleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, ToolMessage

from genai_tk.agents.langchain.middleware.presidio_detector import PresidioDetectorConfig
from genai_tk.agents.langchain.middleware.sensitivity_router_middleware import (
    SensitivityRouterConfig,
    SensitivityRouterMiddleware,
)
from genai_tk.agents.langchain.middleware.sensitivity_scorer import (
    DefaultScorerConfig,
    DefaultSensitivityScorer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_middleware(
    safe_llm: str = "safe_model@fake",
    sensitive_source_patterns: list[str] | None = None,
    threshold: float = 0.35,
) -> SensitivityRouterMiddleware:
    """Build a router middleware with a mocked scorer and lazy safe-LLM resolution."""
    config = SensitivityRouterConfig(
        safe_llm=safe_llm,
        sensitive_source_patterns=sensitive_source_patterns or [],
    )
    middleware = SensitivityRouterMiddleware(config)
    # Replace scorer with controlled mock
    middleware._scorer = _make_scorer(threshold=threshold)
    # Provide a fake safe model to avoid LlmFactory calls
    middleware._safe_model = MagicMock(name="safe_model")
    return middleware


def _make_scorer(threshold: float = 0.35) -> DefaultSensitivityScorer:
    return DefaultSensitivityScorer(
        DefaultScorerConfig(
            sensitivity_threshold=threshold,
            detector=PresidioDetectorConfig(enable_spacy=False),
        )
    )


def _make_model_request(text: str, thread_id: str = "t1") -> MagicMock:
    """Create a mock ModelRequest with human message and thread_id config."""
    request = MagicMock()
    request.messages = [HumanMessage(content=text)]
    request.config = {"configurable": {"thread_id": thread_id}}
    # override() should return a new request — track what model was set
    overridden = MagicMock()
    overridden.messages = request.messages
    overridden.config = request.config
    request.override = MagicMock(return_value=overridden)
    return request


def _make_tool_request(thread_id: str = "t1") -> MagicMock:
    req = MagicMock()
    req.config = {"configurable": {"thread_id": thread_id}}
    return req


# ---------------------------------------------------------------------------
# Tests: content-based routing
# ---------------------------------------------------------------------------


class TestContentBasedRouting:
    def test_clean_text_uses_default_model(self) -> None:
        middleware = _make_middleware()
        request = _make_model_request("The weather is nice today.", thread_id="clean-thread")
        handler = MagicMock(return_value=MagicMock())

        middleware.wrap_model_call(request, handler)

        # Safe model should NOT have been substituted (no override call)
        request.override.assert_not_called()

    def test_sensitive_text_routes_to_safe_model(self) -> None:
        middleware = _make_middleware(threshold=0.01)  # very low threshold → always sensitive for any PII
        request = _make_model_request("root password is abc123", thread_id="sens-thread")
        handler = MagicMock(return_value=MagicMock())

        middleware.wrap_model_call(request, handler)

        request.override.assert_called_once_with(model=middleware._safe_model)

    @pytest.mark.asyncio
    async def test_async_sensitive_text_routes_to_safe_model(self) -> None:
        middleware = _make_middleware(threshold=0.01)
        request = _make_model_request("root password is abc123", thread_id="async-sens-thread")
        handler = AsyncMock(return_value=MagicMock())

        await middleware.awrap_model_call(request, handler)

        request.override.assert_called_once_with(model=middleware._safe_model)

    @pytest.mark.asyncio
    async def test_async_clean_text_no_override(self) -> None:
        middleware = _make_middleware(threshold=0.99)  # extremely high → nothing is sensitive
        request = _make_model_request("Tell me about Python.", thread_id="clean-async")
        handler = AsyncMock(return_value=MagicMock())

        await middleware.awrap_model_call(request, handler)

        request.override.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: source-based routing via tool call inspection
# ---------------------------------------------------------------------------


class TestSourceBasedRouting:
    def test_sensitive_source_triggers_routing(self) -> None:
        """After a tool returns a Document with a sensitive source, model calls route to safe LLM."""
        middleware = _make_middleware(
            sensitive_source_patterns=["**/hr/**"],
            threshold=0.99,  # content scoring won't trigger
        )
        tool_req = _make_tool_request(thread_id="rag-thread")

        # Simulate a ToolMessage with a Document from a sensitive path
        doc = Document(page_content="Salary info", metadata={"source": "/data/hr/salaries.pdf"})
        tool_message = ToolMessage(content="Results", tool_call_id="call-1", artifact=[doc])
        handler = MagicMock(return_value=tool_message)

        middleware.wrap_tool_call(tool_req, handler)

        # Now a model call should be routed
        model_req = _make_model_request("What is the salary?", thread_id="rag-thread")
        model_handler = MagicMock(return_value=MagicMock())
        middleware.wrap_model_call(model_req, model_handler)

        model_req.override.assert_called_once_with(model=middleware._safe_model)

    def test_non_sensitive_source_no_routing(self) -> None:
        middleware = _make_middleware(
            sensitive_source_patterns=["**/hr/**"],
            threshold=0.99,
        )
        tool_req = _make_tool_request(thread_id="clean-rag")

        doc = Document(page_content="Public info", metadata={"source": "/data/public/faq.pdf"})
        tool_message = ToolMessage(content="Results", tool_call_id="call-1", artifact=[doc])
        handler = MagicMock(return_value=tool_message)

        middleware.wrap_tool_call(tool_req, handler)

        model_req = _make_model_request("What is in the FAQ?", thread_id="clean-rag")
        model_handler = MagicMock(return_value=MagicMock())
        middleware.wrap_model_call(model_req, model_handler)

        model_req.override.assert_not_called()

    def test_source_routing_is_sticky(self) -> None:
        """Once a sensitive source is encountered, all future calls for that thread are routed."""
        middleware = _make_middleware(
            sensitive_source_patterns=["**/confidential/**"],
            threshold=0.99,
        )
        # Trigger source detection
        doc = Document(page_content="Secret", metadata={"source": "/docs/confidential/report.pdf"})
        tool_msg = ToolMessage(content="ok", tool_call_id="c1", artifact=[doc])
        middleware.wrap_tool_call(_make_tool_request("sticky-thread"), MagicMock(return_value=tool_msg))

        # Multiple subsequent calls should all be routed
        for i in range(3):
            req = _make_model_request("Follow-up question", thread_id="sticky-thread")
            middleware.wrap_model_call(req, MagicMock(return_value=MagicMock()))
            req.override.assert_called_once_with(model=middleware._safe_model)

    @pytest.mark.asyncio
    async def test_async_source_routing(self) -> None:
        middleware = _make_middleware(
            sensitive_source_patterns=["**/hr/**"],
            threshold=0.99,
        )
        doc = Document(page_content="HR data", metadata={"source": "/hr/contracts/emp.pdf"})
        tool_msg = ToolMessage(content="ok", tool_call_id="c1", artifact=[doc])
        await middleware.awrap_tool_call(_make_tool_request("async-rag"), AsyncMock(return_value=tool_msg))

        model_req = _make_model_request("What are the terms?", thread_id="async-rag")
        await middleware.awrap_model_call(model_req, AsyncMock(return_value=MagicMock()))
        model_req.override.assert_called_once_with(model=middleware._safe_model)


# ---------------------------------------------------------------------------
# Tests: glob pattern matching
# ---------------------------------------------------------------------------


class TestGlobPatternMatching:
    @pytest.mark.parametrize(
        "source,patterns,should_match",
        [
            ("/data/hr/file.pdf", ["**/hr/**"], True),
            ("/data/hr/subdir/file.pdf", ["**/hr/**"], True),
            ("/data/public/file.pdf", ["**/hr/**"], False),
            ("/docs/confidential/x.txt", ["**/confidential/**"], True),
            ("/docs/public/x.txt", ["**/confidential/**"], False),
            ("C:/data/hr/file.txt", ["**/hr/**"], True),  # Windows-style paths
            ("/data/file.txt", ["*.txt"], True),
        ],
    )
    def test_glob_matching(self, source: str, patterns: list[str], should_match: bool) -> None:
        middleware = _make_middleware(sensitive_source_patterns=patterns)
        assert middleware._matches_sensitive_pattern(source) == should_match


# ---------------------------------------------------------------------------
# Tests: thread isolation
# ---------------------------------------------------------------------------


class TestThreadIsolation:
    def test_sensitive_source_only_affects_its_thread(self) -> None:
        middleware = _make_middleware(sensitive_source_patterns=["**/hr/**"], threshold=0.99)

        # Thread A gets a sensitive source
        doc = Document(page_content="HR", metadata={"source": "/data/hr/file.pdf"})
        tool_msg = ToolMessage(content="ok", tool_call_id="c1", artifact=[doc])
        middleware.wrap_tool_call(_make_tool_request("thread-A"), MagicMock(return_value=tool_msg))

        # Thread B should NOT be affected
        req_b = _make_model_request("clean question", thread_id="thread-B")
        middleware.wrap_model_call(req_b, MagicMock(return_value=MagicMock()))
        req_b.override.assert_not_called()

        # Thread A should be affected
        req_a = _make_model_request("clean question", thread_id="thread-A")
        middleware.wrap_model_call(req_a, MagicMock(return_value=MagicMock()))
        req_a.override.assert_called_once_with(model=middleware._safe_model)


# ---------------------------------------------------------------------------
# Tests: tool result with no documents
# ---------------------------------------------------------------------------


class TestToolResultWithNoDocs:
    def test_string_artifact_ignored(self) -> None:
        middleware = _make_middleware(sensitive_source_patterns=["**/hr/**"], threshold=0.99)

        tool_msg = ToolMessage(content="plain string result", tool_call_id="c1", artifact="just a string")
        middleware.wrap_tool_call(_make_tool_request("no-doc-thread"), MagicMock(return_value=tool_msg))

        req = _make_model_request("question", thread_id="no-doc-thread")
        middleware.wrap_model_call(req, MagicMock(return_value=MagicMock()))
        req.override.assert_not_called()

    def test_none_artifact_ignored(self) -> None:
        middleware = _make_middleware(sensitive_source_patterns=["**/hr/**"], threshold=0.99)

        tool_msg = ToolMessage(content="result", tool_call_id="c1")
        middleware.wrap_tool_call(_make_tool_request("none-artifact"), MagicMock(return_value=tool_msg))

        req = _make_model_request("question", thread_id="none-artifact")
        middleware.wrap_model_call(req, MagicMock(return_value=MagicMock()))
        req.override.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: YAML-style construction
# ---------------------------------------------------------------------------


class TestYamlStyleConstruction:
    def test_flat_kwargs(self) -> None:
        middleware = SensitivityRouterMiddleware(
            safe_llm="my_safe@openai",
            sensitive_source_patterns=["**/secret/**"],
        )
        assert middleware._config.safe_llm == "my_safe@openai"
        assert middleware._config.sensitive_source_patterns == ["**/secret/**"]
