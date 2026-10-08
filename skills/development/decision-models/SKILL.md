---
name: decision-models
description: Use and configure System One / Decision Models (OpenRouter, TypeSafe, Cloudflare Clef, Solar Decide, chat adapter fallback) in genai-tk for fast, low-cost classification, routing, and scoring.
---

# Decision Models (System One)

Decision models evaluate structured state against narrow, typed questions returning calibrated probabilities rather than generating text. They run non-autoregressively (output tokens are typically free and latency is ~50-100ms).

## Core Primitives

- **`Noul`**: Binary boolean judgment. Returns `noul` $\in [0.0, 1.0]$ representing probability of affirmative/yes.
- **`Choice`**: Categorical classification over labeled criteria. Returns `choice`, `confidence`, and `probabilities` distribution.
- **`Score`**: Ordinal evaluation along an ordered rubric list. Returns `score` (expected value), `confidence`, and `probabilities`.

## Key Imports

```python
from genai_tk.core.decision import (
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
model = get_decision_model("clef_flash@openrouter")  # or "default", "fake"

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
response = model.invoke({
    "state": {"ticket": "Cancel subscription", "customer_tier": "pro"},
    "questions": {
        "churn_risk": Noul(instructions="Is the customer canceling due to dissatisfaction?"),
        "team": Choice(instructions="Assign to team", criteria={"retention": "Cancel", "billing": "Pay"}),
    },
})
```

## Chat Model Fallback Adapter

Any standard LangChain chat model can be converted to a decision model (via structured output prompting):

```python
from genai_tk.core.factories import get_decision_model_from_chat_model, get_llm

chat_llm = get_llm("gpt-4o-mini@openai")
decision_model = get_decision_model_from_chat_model(chat_llm)
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
