"""Reusable mixins for LLM providers.

Encapsulates provider-specific behaviors including:
- Seed placement (root, model_kwargs, or omit)
- Routing strategies (OpenRouter sort/quantization, EdenAI routing)
- Reasoning / thinking parameters (effort, enable_thinking)
- Proxy environment handling (e.g. for Ollama)
"""

from __future__ import annotations

from typing import Any

SEED = 42
DEFAULT_MAX_RETRIES = 2


class SeedHandlingMixin:
    """Handles placement of the 'seed' parameter for deterministic generation."""

    @classmethod
    def apply_seed(
        cls,
        params: dict[str, Any],
        location: str | None = "root",
        default_seed: int = SEED,
    ) -> dict[str, Any]:
        """Adjust params for seed location: 'root', 'model_kwargs', or None/omit."""
        out = params.copy()
        if location == "model_kwargs":
            seed = out.pop("seed", default_seed)
            model_kwargs = out.setdefault("model_kwargs", {})
            model_kwargs["seed"] = seed
        elif location in (None, "omit"):
            out.pop("seed", None)
        else:
            out.setdefault("seed", default_seed)
        return out


def _to_plain_container(value: Any) -> Any:
    """Recursively convert OmegaConf DictConfig/ListConfig (even inside plain dicts/lists) to plain Python types."""
    from omegaconf import DictConfig, ListConfig, OmegaConf

    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {k: _to_plain_container(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_container(v) for v in value]
    return value


class OpenRouterRoutingMixin:
    """Applies provider routing and quantization preferences for OpenRouter."""

    @classmethod
    def apply_routing(
        cls,
        create_params: dict[str, Any],
        routing_strategy: str | dict[str, Any] | None,
        model_name: str,
        extra_body_default: dict[str, Any] | None = None,
    ) -> None:
        extra_body: dict[str, Any] = {}
        if extra_body_default:
            plain_default = _to_plain_container(extra_body_default)
            extra_body = plain_default if isinstance(plain_default, dict) else {}
            vendor = model_name.partition("/")[0]
            if vendor not in ["openai", "anthropic", "mistralai"]:
                default_quantization = extra_body.pop("default_quantization", [])
                if default_quantization:
                    extra_body = {"provider": {"quantizations": list(default_quantization)}}
                else:
                    extra_body = {}
            else:
                extra_body = {}

        existing_extra = create_params.get("extra_body", {})
        if isinstance(existing_extra, dict):
            plain_existing = _to_plain_container(existing_extra)
            if isinstance(plain_existing, dict):
                extra_body.update(plain_existing)

        # default_quantization is genai-tk internal config, not an OpenRouter API field
        extra_body.pop("default_quantization", None)

        if routing_strategy:
            provider_payload = extra_body.get("provider", {})
            if not isinstance(provider_payload, dict):
                provider_payload = {}
            else:
                provider_payload = provider_payload.copy()

            if isinstance(routing_strategy, dict):
                provider_payload.update(routing_strategy)
            else:
                strat = str(routing_strategy).strip().lower()
                if strat in ("speed", "throughput", "fastest", "nitro"):
                    provider_payload["sort"] = "throughput"
                elif strat in ("price", "cost", "cheapest", "floor"):
                    provider_payload["sort"] = "price"
                elif strat in ("latency", "fast", "lowest_latency"):
                    provider_payload["sort"] = "latency"
                elif strat in ("exact", "exacto", "precision", "quality"):
                    provider_payload["require_parameters"] = True
                else:
                    provider_payload["sort"] = strat

            extra_body["provider"] = provider_payload

        if extra_body:
            create_params["extra_body"] = extra_body
        else:
            create_params.pop("extra_body", None)


class EdenAIRoutingMixin:
    """Applies EdenAI gateway routing strategies (cost, speed, latency, exact)."""

    @classmethod
    def apply_routing(
        cls,
        create_params: dict[str, Any],
        routing_strategy: str | dict[str, Any] | None,
    ) -> None:
        if not routing_strategy:
            return

        extra_body = create_params.get("extra_body", {}).copy() if "extra_body" in create_params else {}
        routing_payload = extra_body.get("routing", {})
        if not isinstance(routing_payload, dict):
            routing_payload = {}
        else:
            routing_payload = routing_payload.copy()

        if isinstance(routing_strategy, dict):
            routing_payload.update(routing_strategy)
        else:
            strat = str(routing_strategy).strip().lower()
            if strat in ("cost", "price", "cheapest", "floor"):
                routing_payload["sort"] = "cost"
            elif strat in ("speed", "throughput", "fastest", "nitro"):
                routing_payload["sort"] = "speed"
            elif strat in ("latency", "lowest_latency"):
                routing_payload["sort"] = "latency"
            elif strat in ("exact", "precision", "exacto", "quality"):
                routing_payload["sort"] = "exact"
            else:
                routing_payload["sort"] = strat

        extra_body["routing"] = routing_payload
        create_params["extra_body"] = extra_body


class ReasoningMixin:
    """Applies standardized reasoning/thinking parameters to OpenAI-compatible payloads."""

    @classmethod
    def apply_reasoning(
        cls,
        create_params: dict[str, Any],
        reasoning_payload: dict[str, Any] | None,
    ) -> None:
        if not reasoning_payload:
            return
        extra_body = create_params.get("extra_body", {}).copy() if "extra_body" in create_params else {}
        existing_reasoning = extra_body.get("reasoning", {})
        if not isinstance(existing_reasoning, dict):
            existing_reasoning = {}
        existing_reasoning.update(reasoning_payload)
        extra_body["reasoning"] = existing_reasoning
        create_params["extra_body"] = extra_body


class EdenAIReasoningMixin:
    """Applies reasoning options and vendor-specific thinking flags for EdenAI."""

    @classmethod
    def apply_reasoning(
        cls,
        create_params: dict[str, Any],
        reasoning_payload: dict[str, Any] | None,
        model_name: str,
    ) -> None:
        if not reasoning_payload:
            return
        effort = reasoning_payload.get("effort")
        if effort:
            create_params["reasoning_effort"] = effort
        if model_name.startswith("qwen/"):
            extra_body = create_params.get("extra_body", {}).copy() if "extra_body" in create_params else {}
            extra_backend = extra_body.get("extra_body", {})
            if not isinstance(extra_backend, dict):
                extra_backend = {}
            if effort == "none":
                extra_backend.update({"enable_thinking": False, "thinking": {"type": "disabled"}})
            else:
                extra_backend.update({"enable_thinking": True})
            extra_body["extra_body"] = extra_backend
            create_params["extra_body"] = extra_body
