"""OpenAI decision model implementation using LangChain's OpenAIDecisions integration."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import SecretStr

from genai_tk.core.decision_models.base import BaseDecisionModel
from genai_tk.core.decision_models.types import (
    Choice,
    ChoiceAnswer,
    ClassifierRequest,
    ClassifierResponse,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    Usage,
    serialize_decision_state,
)
from genai_tk.core.providers import get_provider_api_key


class OpenAIDecisionModel(BaseDecisionModel):
    """Decision model implementation backed by langchain_openai.decisions.OpenAIDecisions.

    Converts ClassifierRequest (Noul, Choice, Score) to OpenAIDecisions
    (Predicate, Choice, Score) and normalizes answers back to ClassifierResponse.
    """

    model: str = "gpt-6-luna"
    api_key: SecretStr | None = None
    base_url: str | None = None
    timeout: float = 30.0

    def _get_api_key_str(self) -> str:
        if self.api_key:
            return self.api_key.get_secret_value()
        env_key = get_provider_api_key("openai")
        if env_key:
            return env_key.get_secret_value()
        raw = os.getenv("OPENAI_API_KEY")
        if raw:
            return raw.strip("\"' \t\n\r")
        raise EnvironmentError("OpenAI API key not found. Set OPENAI_API_KEY or pass api_key.")

    def _get_client(self) -> Any:
        from langchain_openai.decisions import OpenAIDecisions

        kwargs: dict[str, Any] = {
            "model": self.model,
            "api_key": SecretStr(self._get_api_key_str()),
            "timeout": self.timeout,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAIDecisions(**kwargs)

    def _convert_questions(self, request: ClassifierRequest) -> dict[str, Any]:
        from langchain_openai.decisions import Choice as LcChoice
        from langchain_openai.decisions import Predicate as LcPredicate
        from langchain_openai.decisions import Score as LcScore

        converted: dict[str, Any] = {}
        for q_id, q in request.questions.items():
            if isinstance(q, Noul):
                converted[q_id] = LcPredicate(instructions=str(q.instructions))
            elif isinstance(q, Choice):
                choices_dict = {str(k): str(v) for k, v in q.criteria.items()}
                converted[q_id] = LcChoice(instructions=str(q.instructions), choices=choices_dict)
            elif isinstance(q, Score):
                levels_list = [str(x) for x in q.criteria]
                converted[q_id] = LcScore(instructions=str(q.instructions), levels=levels_list)
            else:
                converted[q_id] = LcPredicate(instructions=str(getattr(q, "instructions", "")))
        return converted

    def _convert_response(self, lc_response: Any) -> ClassifierResponse:
        answers: dict[str, Any] = {}

        for q_id, ans in getattr(lc_response, "predicates", {}).items():
            answers[q_id] = NoulAnswer(noul=ans.probability)

        for q_id, ans in getattr(lc_response, "choices", {}).items():
            probs = ans.probabilities or {}
            answers[q_id] = ChoiceAnswer(
                choice=ans.choice,
                confidence=ans.confidence if ans.confidence is not None else 1.0,
                probabilities=probs,
            )

        for q_id, ans in getattr(lc_response, "scores", {}).items():
            answers[q_id] = ScoreAnswer(
                score=ans.score,
                confidence=ans.confidence if ans.confidence is not None else 1.0,
                probabilities=ans.probabilities or {},
                legend=ans.legend or {},
            )

        usage = Usage(
            input_tokens=getattr(lc_response.usage, "input_tokens", 0) if hasattr(lc_response, "usage") else 0,
            output_tokens=getattr(lc_response.usage, "output_tokens", 0) if hasattr(lc_response, "usage") else 0,
        )

        return ClassifierResponse(
            model=getattr(lc_response, "model", self.model),
            answers=answers,
            usage=usage,
            provider="openai",
            request_id=getattr(lc_response, "request_id", None),
        )

    def invoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        request = input if isinstance(input, ClassifierRequest) else ClassifierRequest.model_validate(input)
        self.validate_question_count(request)
        with self._trace_scope(request) as handle:
            client = self._get_client()
            converted_questions = self._convert_questions(request)
            input_val = serialize_decision_state(request.state)

            lc_resp = client.invoke({"input": input_val, "questions": converted_questions}, config=config)
            response = self._convert_response(lc_resp)
            self._emit_trace_event(handle, response)
            return response

    async def ainvoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        request = input if isinstance(input, ClassifierRequest) else ClassifierRequest.model_validate(input)
        self.validate_question_count(request)
        with self._trace_scope(request) as handle:
            client = self._get_client()
            converted_questions = self._convert_questions(request)
            input_val = serialize_decision_state(request.state)

            lc_resp = await client.ainvoke({"input": input_val, "questions": converted_questions}, config=config)
            response = self._convert_response(lc_resp)
            self._emit_trace_event(handle, response)
            return response
