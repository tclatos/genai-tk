"""Laya Decision Model implementation (local Router and HTTP server).

Supports:
1. In-process Router (convaiinnovations/laya) for sub-35ms multilingual decisions.
2. Jev-compatible HTTP server mode (laya-serve at http://localhost:8000/v1/systemone).
"""

from __future__ import annotations

import os
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from loguru import logger
from pydantic import Field, SecretStr

from genai_tk.core.decision_models.base import BaseDecisionModel
from genai_tk.core.decision_models.http_base import HttpSystemOneDecisionModel
from genai_tk.core.decision_models.types import (
    ClassifierRequest,
    ClassifierResponse,
    serialize_decision_state,
)

_DEFAULT_LAYA_SERVER_URL = "http://localhost:8000/v1/systemone"


class LayaDecisionModel(BaseDecisionModel):
    """Decision model implementation for Laya.

    Attributes:
        model: Model name/variant ('laya', 'multilingual', 'typed-decisions', etc.)
        mode: 'router' for in-process laya.Router, or 'server' for HTTP laya-serve.
        preload: Preload checkpoints into memory in router mode.
        device: 'cuda', 'cpu', etc.
        server_url: Base URL when running in 'server' mode.
    """

    model: str = "laya"
    mode: Literal["router", "server"] = "router"
    preload: bool = False
    device: str | None = None
    server_url: str = Field(default=_DEFAULT_LAYA_SERVER_URL)
    api_key: SecretStr | None = None
    timeout: float = 30.0

    _router: Any = None
    _http_client: HttpSystemOneDecisionModel | None = None

    def model_post_init(self, __context: dict) -> None:
        if self.mode == "server":
            self._init_server()
        else:
            self._init_router()

    def _init_server(self) -> None:
        class _LayaServerClient(HttpSystemOneDecisionModel):
            model: str = self.model
            api_key: SecretStr | None = self.api_key
            api_base: str = self.server_url
            timeout: float = self.timeout
            provider_name: str = "laya"

            def _get_api_key_str(self) -> str:
                if self.api_key:
                    return self.api_key.get_secret_value()
                return os.getenv("LAYA_API_KEY", "")

        self._http_client = _LayaServerClient(
            model=self.model,
            api_key=self.api_key,
            api_base=self.server_url,
            timeout=self.timeout,
        )

    def _init_router(self) -> None:
        try:
            from laya import Router

            kwargs: dict[str, Any] = {"preload": self.preload}
            if self.device:
                kwargs["device"] = self.device
            self._router = Router(**kwargs)
        except ImportError:
            logger.warning(
                "Package 'laya' is not installed. Falling back to Laya server mode at "
                f"{self.server_url}. Install with 'uv add laya' or pip install laya."
            )
            self.mode = "server"
            self._init_server()

    def invoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        request = input if isinstance(input, ClassifierRequest) else ClassifierRequest.model_validate(input)
        self.validate_question_count(request)

        if self.mode == "server" and self._http_client is not None:
            return self._http_client.invoke(request, config=config, **kwargs)

        with self._trace_scope(request) as handle:
            serialized_state = serialize_decision_state(request.state)
            questions_payload: dict[str, Any] = {}
            for q_id, q in request.questions.items():
                questions_payload[q_id] = q.model_dump(exclude_none=True)

            model_override = kwargs.get("model") or (self.model if self.model not in ("laya", "default") else None)
            predict_kwargs: dict[str, Any] = {}
            if model_override:
                predict_kwargs["model"] = model_override
            if "max_len" in kwargs:
                predict_kwargs["max_len"] = kwargs["max_len"]

            result = self._router.predict(serialized_state, questions_payload, **predict_kwargs)

            answers = result.get("answers", {})
            routing_info = result.get("routing", {})
            response = ClassifierResponse.model_validate(
                {
                    "model": routing_info.get("model", self.model),
                    "answers": answers,
                    "usage": result.get("usage", {"input_tokens": 0, "output_tokens": 0}),
                    "provider": "laya",
                    "request_id": result.get("id"),
                }
            )
            self._emit_trace_event(handle, response)
            return response

    async def ainvoke(
        self,
        input: ClassifierRequest | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        # In-process Laya router is CPU/GPU bound synchronous; run via invoke or delegate to server client
        if self.mode == "server" and self._http_client is not None:
            return await self._http_client.ainvoke(input, config=config, **kwargs)
        return self.invoke(input, config=config, **kwargs)
