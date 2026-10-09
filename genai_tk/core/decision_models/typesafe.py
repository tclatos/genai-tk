"""TypeSafe decision model implementation.

Communicates with TypeSafe's /v1/systemone API.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig
from pydantic import Field, SecretStr

from genai_tk.core.decision_models.base import BaseDecisionModel
from genai_tk.core.decision_models.types import (
    ClassifierRequest,
    ClassifierResponse,
    serialize_decision_state,
)

_DEFAULT_TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"


class TypeSafeDecisionModel(BaseDecisionModel):
    """Decision model implementation for TypeSafe System One API."""

    model: str = "jev-latest"
    api_key: SecretStr | None = None
    api_base: str = Field(default=_DEFAULT_TYPESAFE_URL)
    timeout: float = 30.0

    def _get_api_key_str(self) -> str:
        if self.api_key:
            return self.api_key.get_secret_value()
        env_key = os.getenv("TYPESAFE_API_KEY")
        if env_key:
            return env_key
        raise ValueError("TypeSafe API key not found. Set TYPESAFE_API_KEY environment variable or pass api_key.")

    def _build_payload(self, request: ClassifierRequest) -> dict[str, Any]:
        serialized_state = serialize_decision_state(request.state)
        questions_payload: dict[str, Any] = {}
        for q_id, q in request.questions.items():
            questions_payload[q_id] = q.model_dump(exclude_none=True)

        return {
            "model": self.model,
            "state": serialized_state,
            "questions": questions_payload,
        }

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_api_key_str()}",
            "Content-Type": "application/json",
        }

    def invoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        request = input if isinstance(input, ClassifierRequest) else ClassifierRequest.model_validate(input)
        self.validate_question_count(request)
        payload = self._build_payload(request)
        headers = self._build_headers()

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self.api_base, json=payload, headers=headers)
            if not resp.is_success:
                raise RuntimeError(f"TypeSafe API error ({resp.status_code}): {resp.text}")
            data = resp.json()

        return ClassifierResponse.model_validate(
            {
                "model": data.get("model", self.model),
                "answers": data.get("answers", {}),
                "usage": data.get("usage", {}),
                "provider": "typesafe",
                "request_id": resp.headers.get("x-typesafe-request-id"),
            }
        )

    async def ainvoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        request = input if isinstance(input, ClassifierRequest) else ClassifierRequest.model_validate(input)
        self.validate_question_count(request)
        payload = self._build_payload(request)
        headers = self._build_headers()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.api_base, json=payload, headers=headers)
            if not resp.is_success:
                raise RuntimeError(f"TypeSafe API error ({resp.status_code}): {resp.text}")
            data = resp.json()

        return ClassifierResponse.model_validate(
            {
                "model": data.get("model", self.model),
                "answers": data.get("answers", {}),
                "usage": data.get("usage", {}),
                "provider": "typesafe",
                "request_id": resp.headers.get("x-typesafe-request-id"),
            }
        )
