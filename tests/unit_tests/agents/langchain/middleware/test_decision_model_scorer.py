"""Unit tests for the decision-model-backed sensitivity scorer and its router integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage
from pydantic import Field

from genai_tk.agents.langchain.middleware.decision_model_scorer import DecisionModelSensitivityScorer
from genai_tk.agents.langchain.middleware.sensitivity_router_middleware import (
    SensitivityRouterConfig,
    SensitivityRouterMiddleware,
)
from genai_tk.core.decision_models import ClassifierRequest, ClassifierResponse, FakeDecisionModel, NoulAnswer


class _RecordingModel(FakeDecisionModel):
    """Fake decision model returning a fixed noul and recording incoming requests."""

    stub_noul: float = 0.5
    fail_with: str | None = None
    requests: list[ClassifierRequest] = Field(default_factory=list)

    def invoke(
        self,
        input: ClassifierRequest | dict[str, object],
        config: object | None = None,
        **kwargs: object,
    ) -> ClassifierResponse:
        if self.fail_with is not None:
            raise RuntimeError(self.fail_with)
        request = input if isinstance(input, ClassifierRequest) else ClassifierRequest.model_validate(input)
        self.requests.append(request)
        return ClassifierResponse(model="stub", answers={"sensitivity": NoulAnswer(noul=self.stub_noul)})


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------


class TestDecisionModelSensitivityScorer:
    def test_empty_text_short_circuits_without_model_call(self) -> None:
        stub = _RecordingModel(stub_noul=0.99)
        scorer = DecisionModelSensitivityScorer(decision_model=stub)

        result = scorer.assess("   ")

        assert result.is_sensitive is False
        assert result.score == 0.0
        assert stub.requests == []

    def test_high_probability_is_sensitive(self) -> None:
        stub = _RecordingModel(stub_noul=0.9)
        scorer = DecisionModelSensitivityScorer(decision_model=stub)

        result = scorer.assess("My password is hunter2")

        assert result.is_sensitive is True
        assert result.score == 0.9
        assert result.level == "critical"
        assert result.details["model"] == "stub"

    def test_low_probability_is_not_sensitive(self) -> None:
        stub = _RecordingModel(stub_noul=0.1)
        scorer = DecisionModelSensitivityScorer(decision_model=stub)

        result = scorer.assess("What is the capital of France?")

        assert result.is_sensitive is False
        assert result.level == "low"

    def test_threshold_is_configurable(self) -> None:
        stub = _RecordingModel(stub_noul=0.3)
        scorer = DecisionModelSensitivityScorer(decision_model=stub, threshold=0.25)

        assert scorer.assess("anything").is_sensitive is True

    def test_message_is_passed_as_state_with_instructions(self) -> None:
        stub = _RecordingModel(stub_noul=0.5)
        scorer = DecisionModelSensitivityScorer(decision_model=stub)

        scorer.assess("hello world")

        request = stub.requests[0]
        assert request.state == "hello world"
        assert "sensitive" in scorer.instructions.lower()

    def test_non_noul_answer_raises_type_error(self) -> None:
        class _NoAnswerModel(FakeDecisionModel):
            def invoke(
                self,
                input: ClassifierRequest | dict[str, object],
                config: object | None = None,
                **kwargs: object,
            ) -> ClassifierResponse:
                return ClassifierResponse(model="stub", answers={})  # no 'sensitivity' answer

        scorer = DecisionModelSensitivityScorer(decision_model=_NoAnswerModel())
        with pytest.raises(TypeError, match="expected NoulAnswer"):
            scorer.assess("hello")

    def test_lazy_model_resolution_via_factory(self) -> None:
        """Without a pre-built model, the factory resolves model_id (offline 'fake' tag)."""
        scorer = DecisionModelSensitivityScorer(model_id="fake")
        assert isinstance(scorer.decision_model, FakeDecisionModel)


# ---------------------------------------------------------------------------
# Middleware integration
# ---------------------------------------------------------------------------


def _make_request(text: str, thread_id: str = "t1") -> MagicMock:
    request = MagicMock()
    request.messages = [HumanMessage(content=text)]
    request.config = {"configurable": {"thread_id": thread_id}}
    request.override = MagicMock(return_value=MagicMock())
    return request


class TestMiddlewareWithDecisionModel:
    def test_config_decision_model_builds_decision_scorer(self) -> None:
        middleware = SensitivityRouterMiddleware(
            SensitivityRouterConfig(safe_llm="safe_model@fake", decision_model="fake")
        )
        assert isinstance(middleware._scorer, DecisionModelSensitivityScorer)

    def test_sensitive_verdict_routes_to_safe_model(self) -> None:
        adapter = DecisionModelSensitivityScorer(decision_model=_RecordingModel(stub_noul=0.99))
        middleware = SensitivityRouterMiddleware(SensitivityRouterConfig(safe_llm="safe_model@fake"), scorer=adapter)
        middleware._safe_model = MagicMock(name="safe_model")

        request = _make_request("please help", thread_id="dm-1")
        middleware.wrap_model_call(request, MagicMock(return_value=MagicMock()))

        request.override.assert_called_once_with(model=middleware._safe_model)

    def test_clean_verdict_keeps_default_model(self) -> None:
        adapter = DecisionModelSensitivityScorer(decision_model=_RecordingModel(stub_noul=0.05))
        middleware = SensitivityRouterMiddleware(SensitivityRouterConfig(safe_llm="safe_model@fake"), scorer=adapter)
        middleware._safe_model = MagicMock(name="safe_model")

        request = _make_request("please help", thread_id="dm-2")
        middleware.wrap_model_call(request, MagicMock(return_value=MagicMock()))

        request.override.assert_not_called()

    def test_scorer_failure_fails_closed_to_safe_model(self) -> None:
        adapter = DecisionModelSensitivityScorer(decision_model=_RecordingModel(fail_with="backend down"))
        middleware = SensitivityRouterMiddleware(SensitivityRouterConfig(safe_llm="safe_model@fake"), scorer=adapter)
        middleware._safe_model = MagicMock(name="safe_model")

        request = _make_request("please help", thread_id="dm-3")
        middleware.wrap_model_call(request, MagicMock(return_value=MagicMock()))

        request.override.assert_called_once_with(model=middleware._safe_model)

    def test_yaml_style_kwargs(self) -> None:
        middleware = SensitivityRouterMiddleware(
            safe_llm="safe_model@fake",
            decision_model="fake",
            sensitivity_threshold=0.7,
        )
        assert middleware._config.decision_model == "fake"
        assert middleware._config.sensitivity_threshold == 0.7
        assert isinstance(middleware._scorer, DecisionModelSensitivityScorer)
        assert middleware._scorer.threshold == 0.7


# ---------------------------------------------------------------------------
# Laya (local, in-process router)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def laya_scorer() -> DecisionModelSensitivityScorer:
    pytest.importorskip("laya")
    return DecisionModelSensitivityScorer(model_id="laya@laya", model_kwargs={"device": "cpu"})


class TestLayaLocal:
    def test_laya_local_returns_valid_assessment(self, laya_scorer: DecisionModelSensitivityScorer) -> None:
        result = laya_scorer.assess("My login is john.doe@corp.com and my password is hunter2")

        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.is_sensitive, bool)
        assert result.details["provider"] == "laya"

    def test_laya_local_middleware_routing_decision(self, laya_scorer: DecisionModelSensitivityScorer) -> None:
        middleware = SensitivityRouterMiddleware(
            SensitivityRouterConfig(safe_llm="safe_model@fake"), scorer=laya_scorer
        )
        request = _make_request("The weather is nice today.", thread_id="laya-1")

        routed, reason = middleware._should_route(request, "laya-1")

        assert isinstance(routed, bool)
        assert (reason != "") == routed
