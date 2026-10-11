# Reasoning-Effort Handling for Heterogeneous LLM Providers

## Decision Summary

Treat reasoning effort as a requested capability with provider- and
model-specific transport, rather than as a universal request parameter.

Keep the existing inline model syntax:

```python
get_llm(llm="model-name (high)@provider")
```

Normalize the selected effort into a provider-neutral internal payload, map it
to the LangChain/provider parameter accepted by the selected model, and warn
when the local model metadata does not mark the model as thinking-capable.

Effort values are validated against a finite recognized set (`low`, `medium`,
`high`, `minimal`, `xhigh`, `max`, `none`). The `none` value explicitly
disables reasoning: no reasoning payload is forwarded to the provider. Any
other unrecognized value raises a `ValueError` upfront instead of being passed
through, so typos surface before a provider request is made rather than as a
generic API error later. The recognized set deliberately includes the gateway
extras (`minimal`, `xhigh`, `max`, `none`) so legitimate values are accepted.

If the provider rejects a reasoning-specific request at invocation time, log a
warning and retry the same request once with the reasoning controls removed.
The retry is best-effort and must not hide unrelated failures such as invalid
credentials, quota exhaustion, networking problems, or invalid prompts.

## Context

The toolkit already parses a parenthesized effort suffix in
`genai_tk.core.factories.llm_factory` and normalizes it into a reasoning
payload. Explicit `reasoning={...}` options override the suffix, and legacy
flat options remain supported:

```python
get_llm(
    llm="gpt-oss-120b (high)@openrouter",
    reasoning={"effort": "medium", "max_tokens": 2048},
)
```

The current factory sends every such payload through the OpenAI-compatible
path as:

```python
extra_body = {"reasoning": {"effort": "high"}}
```

This is correct for OpenRouter, but it is not a general contract. Provider
integrations and even model generations within the same provider expose
different configuration shapes. Construction may succeed while the provider
later rejects the request, so constructor-only validation cannot provide the
requested fallback.

## Current-State Assessment

| Concern | Current behavior | Gap |
| --- | --- | --- |
| Inline effort | `model (high)@provider` is parsed and normalized | Recognized set now includes the gateway extras `minimal`, `xhigh`, `max`, and `none`; unknown values hard-fail upfront rather than being passed through. |
| Capability warning | A warning is logged when the local `models.dev` metadata lacks `thinking` | The warning is appropriate but catalog metadata can be incomplete or stale. It cannot determine the exact parameter contract. |
| OpenAI-compatible APIs | The payload is always placed in `extra_body.reasoning` | Native OpenAI and Azure LangChain classes expose `reasoning_effort`; other compatible endpoints may reject either shape. |
| Provider-specific APIs | Ollama receives a boolean `reasoning` option | Anthropic and Google are routed through `init_chat_model()` without a mapping from the normalized payload. |
| Failure recovery | LangChain agent middleware retries empty visible responses | It does not handle provider request errors and does not protect direct `get_llm(...).invoke()` callers. |

## Verified Provider and LangChain Contracts

The installed LangChain integrations expose the following relevant fields:

| Integration | Available fields | Recommended handling |
| --- | --- | --- |
| `ChatOpenAI` / `AzureChatOpenAI` | `reasoning_effort`, `reasoning` | Use native `reasoning_effort` for supported OpenAI reasoning models. |
| `ChatAnthropic` | `reasoning_effort`, `thinking` | Use `reasoning_effort` where the selected model supports it; use `thinking={"type": "enabled", "budget_tokens": ...}` only for configured extended-thinking models. |
| `ChatGoogleGenerativeAI` | `reasoning_effort`, `thinking_budget`, `thinking_config` | Map effort to `thinking_level` for Gemini 3+ and an explicit token budget to `thinking_budget` for Gemini 2.5. |
| `ChatOllama` | `reasoning` | Preserve the boolean enable/disable control; effort is not a generic Ollama control. |
| `ChatGroq` | `reasoning_effort`, `reasoning_format` | Only send a supported native control for known compatible models. |
| OpenRouter through `ChatOpenAI` | `extra_body.reasoning` | Send OpenRouter's unified reasoning object, which the gateway maps to its selected upstream model. |

OpenRouter is deliberately a special gateway case. Its `reasoning` object
accepts `effort`, `max_tokens`, `enabled`, and `exclude`, then maps them to the
underlying model where possible. Its model catalogue can additionally report
per-model supported efforts and whether reasoning is mandatory. A generic
factory must not assume those gateway semantics apply to another
OpenAI-compatible endpoint.

The relevant external documentation is:

- [OpenRouter reasoning tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [LangChain OpenAI `reasoning_effort`](https://reference.langchain.com/python/langchain-openai/chat_models/base/BaseChatOpenAI/reasoning_effort)
- [LangChain ChatAnthropic extended thinking and effort](https://docs.langchain.com/oss/python/integrations/chat/anthropic)
- [LangChain ChatGoogleGenerativeAI thinking support](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai)
- [LangChain ChatOllama reasoning models](https://docs.langchain.com/oss/python/integrations/chat/ollama)
- [LangChain ChatGroq reasoning format](https://docs.langchain.com/oss/python/integrations/chat/groq)

## Proposed Design

### 1. Preserve a Provider-Neutral Request

Normalize user input once, before provider construction. The internal payload
should preserve the requested cross-provider fields without pretending every
provider supports each one:

```python
{
    "effort": "high",
    "max_tokens": 2048,
    "enabled": True,
    "exclude": False,
}
```

The factory continues to accept the parenthesized effort syntax. Explicit
`reasoning` options retain precedence over inline syntax and legacy flat
options. Effort is matched case-insensitively against the recognized set
`{low, medium, high, minimal, xhigh, max, none}`.

`none` is a sentinel that disables reasoning: `_extract_reasoning_settings`
returns no payload for it so the provider receives no reasoning control and
uses its default (reasoning off). It is therefore not forwarded as an effort.

Values outside the recognized set raise a `ValueError` at normalization time
with a message listing the valid values and the `none` disable sentinel. This
rejects typos and unsupported levels before any network call. To support a new
effort level a gateway exposes, add it to `REASONING_EFFORT_VALUES`; do not
rely on silent passthrough.

### 2. Make Transport a Provider Configuration Concern

Extend `ProviderInfo` and `providers.yaml` with a reasoning transport policy.
The policy is declarative and can be overridden by model family or explicit
model configuration. It prevents scattered provider-name conditionals in the
factory.

The initial policy set is:

| Policy | Request shape | Scope |
| --- | --- | --- |
| `openrouter_reasoning` | `extra_body.reasoning` | OpenRouter gateway models |
| `native_reasoning_effort` | `reasoning_effort=<effort>` | OpenAI, Azure, Anthropic, or Groq models known to support the field |
| `anthropic_thinking_budget` | `thinking={"type": "enabled", "budget_tokens": ...}` | Explicit extended-thinking Claude models |
| `gemini_thinking_level` | `thinking_level=<effort>` | Gemini 3+ models |
| `gemini_thinking_budget` | `thinking_budget=<tokens>` | Gemini 2.5 models |
| `ollama_reasoning` | `reasoning=<bool>` | Ollama models with supported reasoning mode |
| `none` | Do not send a provider control | Unknown or unsupported combinations |

Per-model selection matters. For example, Gemini 2.5 and Gemini 3 use
different controls, and OpenRouter model metadata determines whether an effort
value is accepted. The local `thinking` capability remains a useful UI and
warning signal, not the authority for all request serialization details.

### 3. Warn Before Sending a Suspect Request

When an inline effort or explicit reasoning payload is present and
`LlmInfo.supports_thinking` is false, log a warning that includes the selected
model, provider, requested payload, and the fact that the request will still
be attempted best-effort.

This preserves user intent. It avoids rejecting models merely because the local
catalogue has not caught up, while making a likely no-op or rejected request
visible to operators.

### 4. Retry Only Reasoning-Configuration Failures

Build two equivalent models when reasoning was requested:

1. A primary model containing the mapped reasoning settings.
2. A fallback model without any reasoning-only settings.

Wrap invocation at the factory boundary so direct model use and agent use
share the behavior. The wrapper must cover `invoke`, `ainvoke`, `stream`, and
`astream`; a factory-only `try` block is insufficient because most providers
validate options when requests are sent.

On a failure, retry exactly once only if the exception is a provider request or
validation failure that identifies a reasoning-specific parameter. Match known
parameter names such as `reasoning`, `reasoning_effort`, `thinking`,
`thinking_level`, and `thinking_budget`, including the configured transport
field. Log the initial error and that the retry removes reasoning controls.

Do not retry on:

- authentication or authorization failures;
- rate limiting, quota, or billing failures;
- connection, timeout, or server failures;
- prompt/content policy errors;
- malformed tool schemas or unrelated request parameters.

This limitation is intentional. Retrying all errors can duplicate a paid
request and conceal an operational or application defect.

### 5. Preserve LangChain Model Semantics

The fallback wrapper must retain the capabilities expected from a
`BaseChatModel`, especially tool binding, structured-output helpers, callbacks,
sync and async invocation, and streaming. Returning a generic runnable from
`get_llm()` is insufficient because agent construction expects chat-model
behavior.

Implementation should therefore prefer a small delegating `BaseChatModel`
wrapper, or a LangChain-supported retry abstraction only after verifying that
it preserves `bind_tools()` and model metadata. This is a compatibility
boundary that needs focused tests before broad adoption.

## Trade-offs and Non-Goals

### Best-Effort Does Not Mean Universal Support

The toolkit can normalize a user request and select the best known transport.
It cannot force a model to reason, make a non-thinking model expose chain of
thought, or discover every provider's current per-model limits without a live
provider catalogue query.

### Do Not Expose Chain of Thought by Default

Reasoning controls and reasoning display are separate concerns. Some providers
return no reasoning tokens, some return encrypted/summarized data, and some
require thought signatures or reasoning blocks to be preserved between tool
calls. The factory should configure inference only; callers and UIs must
separately decide whether and how to render returned reasoning content.

### Preserve Provider-Specific Escapes

Advanced callers may still pass provider-native options directly through
`llm_params`. The normalized `reasoning` payload should own only the fields it
recognizes and should not overwrite an explicit provider-native setting without
an explicit precedence rule.

## Implementation Plan

1. Extend reasoning normalization and tests in
   `genai_tk/core/factories/llm_factory.py` and
   `tests/unit_tests/core/test_llm_factory.py`.
2. Add a typed reasoning transport policy to
   `genai_tk/core/providers.py` and configure base policies in
   `genai_tk/default_config/providers/providers.yaml`.
3. Add model-family overrides in the LLM configuration for cases where the
   provider alone is insufficient.
4. Route the normalized payload through the selected policy during provider
   construction, preserving the current OpenRouter request format.
5. Introduce the focused invocation fallback wrapper and test synchronous,
   asynchronous, and streaming behavior.
6. Verify that a non-thinking model with `(high)` logs the warning and still
   creates a usable model.
7. Document supported mappings and the best-effort fallback behavior in the
   core user documentation.

## Acceptance Criteria

- `model (high)@provider` remains valid and is stripped before normal model
  resolution.
- `model (none)@provider` disables reasoning: no reasoning payload is sent to
  the provider.
- An unrecognized inline effort such as `model (blabla)@provider` raises a
  `ValueError` at factory construction time, listing the valid values, instead
  of being forwarded to the provider.
- The recognized effort set is `{low, medium, high, minimal, xhigh, max, none}`
  and matching is case-insensitive.
- The user receives a warning when effort is requested for a model that is not
  marked as thinking-capable.
- OpenRouter continues to receive `extra_body.reasoning`.
- Native integrations receive only their documented reasoning controls.
- A request rejected specifically for reasoning configuration is retried once
  without that configuration, with a warning in logs.
- An unrelated provider failure is raised without a hidden duplicate request.
- Direct factory consumers and LangChain agents receive the same fallback
  behavior.
- Existing factory behavior unrelated to reasoning remains unchanged.
