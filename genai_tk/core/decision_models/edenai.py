"""EdenAI Decision Model implementation.

Communicates with EdenAI's /v3/alpha/decisions endpoint.
"""

from __future__ import annotations

import os

from pydantic import Field, SecretStr

from genai_tk.core.decision_models.http_base import HttpSystemOneDecisionModel
from genai_tk.core.providers import get_provider_api_key

_DEFAULT_EDENAI_DECISIONS_URL = "https://api.edenai.run/v3/alpha/decisions"


class EdenAIDecisionModel(HttpSystemOneDecisionModel):
    """Decision model implementation for EdenAI Decisions API."""

    model: str = "typesafe/jev-latest"
    api_key: SecretStr | None = None
    api_base: str = Field(default=_DEFAULT_EDENAI_DECISIONS_URL)
    timeout: float = 30.0
    provider_name: str = "edenai"

    def _get_api_key_str(self) -> str:
        if self.api_key:
            return self.api_key.get_secret_value()
        env_key = get_provider_api_key("edenai")
        if env_key:
            return env_key.get_secret_value()
        raw = os.getenv("EDENAI_API_KEY")
        if raw:
            return raw.strip("\"' \t\n\r")
        raise EnvironmentError("EdenAI API key not found. Set EDENAI_API_KEY environment variable or pass api_key.")
