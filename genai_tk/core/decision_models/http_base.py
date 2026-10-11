"""Base class for HTTP-based System One decision models.

Encapsulates:
- Serialization of DecisionState and Question schemas
- NeMo Relay evaluator tracing
- Synchronous and asynchronous HTTP execution
- Uniform error handling and response parsing
"""

from __future__ import annotations

import abc
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig
from pydantic import SecretStr

from genai_tk.core.decision_models.base import BaseDecisionModel
from genai_tk.core.decision_models.types import (
    ClassifierRequest,
    ClassifierResponse,
    serialize_decision_state,
)


class HttpSystemOneDecisionModel(BaseDecisionModel, abc.ABC):
    """Abstract base class for decision models invoking an HTTP API."""

    model: str
    api_key: SecretStr | None = None
    api_base: str
    timeout: float = 30.0
    provider_name: str = "http"

    @abc.abstractmethod
    def _get_api_key_str(self) -> str:
        """Return the API key string or raise an EnvironmentError/ValueError."""
        ...

    def _build_headers(self) -> dict[str, str]:
        """Construct standard HTTP headers for bearer authentication."""
        key = self._get_api_key_str()
        headers = {
            "Content-Type": "application/json",
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _build_payload(self, request: ClassifierRequest) -> dict[str, Any]:
        """Convert ClassifierRequest into standard System One JSON payload."""
        serialized_state = serialize_decision_state(request.state)
        questions_payload: dict[str, Any] = {}
        for q_id, q in request.questions.items():
            questions_payload[q_id] = q.model_dump(exclude_none=True)

        return {
            "model": self.model,
            "state": serialized_state,
            "questions": questions_payload,
        }

    def _parse_response_data(self, data: dict[str, Any]) -> ClassifierResponse:
        """Parse JSON response dictionary into a validated ClassifierResponse."""
        raw_answers = data.get("answers", {})
        normalized_answers: dict[str, Any] = {}
        for q_id, ans in raw_answers.items():
            if isinstance(ans, dict):
                ans_copy = ans.copy()
                if "type" not in ans_copy:
                    if "noul" in ans_copy:
                        ans_copy["type"] = "noul"
                    elif "choice" in ans_copy:
                        ans_copy["type"] = "choice"
                    elif "score" in ans_copy:
                        ans_copy["type"] = "score"
                normalized_answers[q_id] = ans_copy
            else:
                normalized_answers[q_id] = ans

        return ClassifierResponse.model_validate(
            {
                "model": data.get("model", self.model),
                "answers": normalized_answers,
                "usage": data.get("usage", {}),
                "provider": data.get("provider", self.provider_name),
                "request_id": data.get("id"),
            }
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
            payload = self._build_payload(request)
            headers = self._build_headers()

            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(self.api_base, json=payload, headers=headers)
                if not resp.is_success:
                    raise RuntimeError(
                        f"{self.__class__.__name__} API error ({resp.status_code}) on {self.api_base}: {resp.text}"
                    )
                data = resp.json()

            response = self._parse_response_data(data)
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
            payload = self._build_payload(request)
            headers = self._build_headers()

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.api_base, json=payload, headers=headers)
                if not resp.is_success:
                    raise RuntimeError(
                        f"{self.__class__.__name__} API error ({resp.status_code}) on {self.api_base}: {resp.text}"
                    )
                data = resp.json()

            response = self._parse_response_data(data)
            self._emit_trace_event(handle, response)
            return response
