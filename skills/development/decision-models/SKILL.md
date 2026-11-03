---
name: decision-models
description: Use and configure System One / Decision Models (OpenRouter, TypeSafe, Eden.ai, Laya, OpenAI, Cloudflare Clef, Solar Decide, chat adapter fallback) in genai-tk for fast, low-cost classification, routing, and scoring.
---

# Decision Models (System One)

Decision models evaluate structured state against narrow, typed questions returning calibrated probabilities rather than generating text. They run non-autoregressively (output tokens are typically free and latency is ~50-100ms).

## Supported Providers & Backends

- **OpenRouter**: Cloudflare Clef / Clef-flash, TypeSafe Jev, Solar Decide, Perplexity Decider (`clef_flash@openrouter`).
- **TypeSafe**: Direct System One API (`jev@typesafe`).
- **Eden.ai**: Decisions API (`jev@edenai` via `https://api.edenai.run/v3/alpha/decisions`).
- **Laya**: Fast multilingual non-autoregressive decision model (`laya@laya`, `laya_multilingual@laya`, `laya_typed@laya`):
  - In-process `mode="router"`: Sub-35ms GPU / 100ms CPU execution with automated script/language detection.
  - Remote `mode="server"`: Connects to self-hosted `laya-serve` HTTP daemon on `/v1/systemone`.
- **OpenAI**: Native `OpenAIDecisions` integration via `langchain-openai>=1.7.0` (`gpt_6_luna@openai`, `gpt_6_sol@openai`).
- **Chat Model Fallback**: Structured prompting adapter for any standard LangChain `BaseChatModel`.
- **Fake**: Deterministic offline model for fast unit testing (`fake_decision@fake`).

## Core Primitives

- **`Noul`**: Binary boolean judgment. Returns `noul` $\in [0.0, 1.0]$ representing probability of affirmative/yes.
- **`Choice`**: Categorical classification over labeled criteria. Returns `choice`, `confidence`, and `probabilities` distribution.
- **`Score`**: Ordinal evaluation along an ordered rubric list. Returns `score` (expected value), `confidence`, and `probabilities`.

## Key Imports

```python
from genai_tk.core.decision_models import (
    Choice,
    ClassifierRequest,
    ClassifierResponse,
    Noul,
    Score,
)
from genai_tk.core.factories import (
    get_decision_model,
    get_decision_model_from_chat_model,
)
```

## Quick Python Usage

```python
model = get_decision_model("clef_flash@openrouter")  # or "laya@laya", "jev@edenai", "default", "fake"

# 1. Quick binary decision
p_urgent = model.decide_noul(
    state="Checkout page gives 500 error",
    instructions="Does this indicate an active production outage?",
)

# 2. Categorical routing
dept = model.decide_choice(
    state="I need a receipt for my payment",
    instructions="Which department should handle this?",
    criteria={
        "billing": "Invoices, payments, charges",
        "tech": "Bugs, outages, crashes",
        "sales": "New accounts, plan upgrades",
    },
)
print(dept.choice, dept.confidence, dept.probabilities)

# 3. Rubric scoring
frustration = model.decide_score(
    state="This is the third time it fails!",
    instructions="How frustrated is the user?",
    criteria=["Calm", "Frustrated", "Very angry"],
)
print(frustration.score, frustration.probabilities)

# 4. Multi-question batch on shared state
response = model.invoke(
    {
        "state": {"ticket": "Cancel subscription", "customer_tier": "pro"},
        "questions": {
            "churn_risk": Noul(instructions="Is the customer canceling due to dissatisfaction?"),
            "team": Choice(instructions="Assign to team", criteria={"retention": "Cancel", "billing": "Pay"}),
        },
    }
)

# 5. Automatic chunking with batch_invoke (when exceeding model limits)
response = model.batch_invoke(request, batch_size=32)
```

## Local Laya Decision Model

Laya runs locally in-process without cloud API calls or costs:

```python
from genai_tk.core.factories import get_decision_model

# In-process multilingual router (preloads checkpoints or on-demand)
laya_dm = get_decision_model("laya@laya")
res = laya_dm.invoke(
    {
        "state": "Hi, we were billed twice for March. Please refund the duplicate today.",
        "questions": {
            "dept": Choice(
                instructions="Which dept?",
                criteria={"billing": "Payment & invoices", "tech": "Bugs & crashes"},
            ),
            "refund": Noul(instructions="Is refund requested?"),
        },
    }
)
print(res.choices["dept"].choice)  # "billing"
print(res.nouls["refund"].noul)  # e.g. 0.8899
```

## Limits & Batching

Decision models are capped on **four independent axes**. Providers bound them
separately, so a model can accept any number of questions while still capping the
options inside a single `Choice`:

| Axis | Meaning |
|---|---|
| `context_length` | Tokens accepted in one request |
| `max_questions` | Questions in one request |
| `max_options` | Options inside a single `Choice` question |
| `max_levels` | Levels inside a single `Score` question |

**TypeSafe Jev** publishes no cap on the number of questions per request — any number
is accepted as long as the request fits the context (64k tokens per request, with
`state` + the longest single question capped at 32k). Its real per-question limits are
**255 options** per `Choice` and **2–10 levels** per `Score`. An optional long-context
mode raises both context limits to 262k when the request carries the
`TypeSafe-Danger-Zone-Long-Context: true` header, at the cost of speed and accuracy.

**Cloudflare Clef / Clef-flash** do cap the question count: **1–64 questions per call**,
enforced by the API itself (`maxProperties: 64`), plus **255 options** and **10 levels**
per question. This is a hard constraint, not a batching recommendation.

For Laya, Solar Decide, Mercury Decide, Perplexity Decider and the OpenAI decision
models the provider publishes no per-question caps, so none are enforced locally.

Limits live in `config/providers/decision_models.yaml`. A number is a documented hard
limit; `null` means no documented limit (either the provider imposes none, or it is not
published) — never a guess. `validate_request()` raises a descriptive `ValueError`
before any network call when a request exceeds a configured limit.

`recommended_batch_size` is separate from all of the above: it is an application
batching policy, not a model limit. `batch_invoke()` slices on
`batch_size or recommended_batch_size or max_questions`, which is what keeps a large
request safe on a model whose only real bound is the context window:

```python
# 200 questions on Jev: no hard cap, but chunked 32 at a time by policy
response = model.batch_invoke(request)

# Override the policy per call
response = model.batch_invoke(request, batch_size=64)
```

## Chat Model Fallback Adapter

Any standard LangChain chat model can be converted to a decision model (via structured output prompting):

```python
from genai_tk.core.factories import get_decision_model_from_chat_model, get_llm

chat_llm = get_llm("gpt-4o-mini@openai")
decision_model = get_decision_model_from_chat_model(chat_llm)
```

## Agent Evaluation & Grader Helpers

Decision Models serve as typed evaluators, replacing legacy prompt-based LLM judges (such as `openevals`):

```python
from genai_tk.core.decision_models.evaluators import (
    evaluate_correctness,
    evaluate_conciseness,
    evaluate_groundedness,
    evaluate_tool_selection,
)

# 1. Correctness (returns NoulAnswer with noul probability [0.0, 1.0])
c_verdict = evaluate_correctness(model, question="What is 2+2?", gold_answer="4", agent_answer="4")

# 2. Conciseness (returns ScoreAnswer on a 3-point scale: 0=padded, 1=acceptable, 2=concise)
s_verdict = evaluate_conciseness(model, question="What is 2+2?", agent_answer="4")

# 3. Groundedness (returns NoulAnswer)
g_verdict = evaluate_groundedness(model, evidence=["Revenue in 2025 was $10M"], agent_answer="Revenue was $10M")

# 4. Tool Selection Quality (returns ChoiceAnswer: optimal | suboptimal | incorrect)
t_verdict = evaluate_tool_selection(
    model,
    task="Calculate 15*3",
    available_tools=["calculator", "web_search"],
    selected_tools=["calculator"],
)
```

## CLI Usage

```bash
# Binary question
uv run cli core classifier -s "Database connection timeout" --noul "Is this an infrastructure issue?"

# Choice question
uv run cli core classifier -s "Need refund" --choice "team" -c "billing: refunds, tech: bugs, sales: pricing"

# Run built-in demo
uv run cli core classifier --demo --model fake
uv run cli core classifier --demo --model clef_flash@openrouter
```
