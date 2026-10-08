"""OpenRouter decision model implementation.

Communicates with OpenRouter's /api/alpha/decisions (or /systemone) endpoint.
"""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig
from pydantic import Field, SecretStr

from genai_tk.core.decision.base import BaseDecisionModel
from genai_tk.core.decision.types import (
    ClassifierRequest,
    ClassifierResponse,
    serialize_decision_state,
)
from genai_tk.core.providers import get_provider_api_key

_DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"


class OpenRouterDecisionModel(BaseDecisionModel):
    """Decision model implementation for OpenRouter Decisions API."""

    model: str
    api_key: SecretStr | None = None
    api_base: str = Field(default=_DEFAULT_OPENROUTER_URL)
    http_referer: str | None = None
    x_open_router_title: str | None = None
    timeout: float = 30.0

    def _get_api_key_str(self) -> str:
        if self.api_key:
            return self.api_key.get_secret_value()
        env_key = get_provider_api_key("openrouter")
        if env_key:
            return env_key.get_secret_value()
        raise ValueError(
            "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable or pass api_key."
        )

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
        headers = {
            "Authorization": f"Bearer {self._get_api_key_str()}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_open_router_title:
            headers["X-OpenRouter-Title"] = self.x_open_router_title
        return headers

    def invoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        request = input if isinstance(input, ClassifierRequest) else ClassifierRequest.model_validate(input)
        payload = self._build_payload(request)
        headers = self._build_headers()

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self.api_base, json=payload, headers=headers)
            if not resp.is_success:
                raise RuntimeError(
                    f"OpenRouter Decisions API error ({resp.status_code}): {resp.text}"
                )
            data = resp.json()

        # If OpenRouter returned answers dict
        return ClassifierResponse.model_validate(
            {
                "model": data.get("model", self.model),
                "answers": data.get("answers", {}),
                "usage": data.get("usage", {}),
                "provider": data.get("provider", "openrouter"),
                "request_id": data.get("id"),
            }
        )

    async def ainvoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        request = input if isinstance(input, ClassifierRequest) else ClassifierRequest.model_validate(input)
        payload = self._build_payload(request)
        headers = self._build_headers()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.api_base, json=payload, headers=headers)
            if not resp.is_success:
                raise RuntimeError(
                    f"OpenRouter Decisions API error ({resp.status_code}): {resp.text}"
                )
            data = resp.json()

        return ClassifierResponse.model_validate(
            {
                "model": data.get("model", self.model),
                "answers": data.get("answers", {}),
                "usage": data.get("usage", {}),
                "provider": data.get("provider", "openrouter"),
                "request_id": data.get("id"),
            }
        )
