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

## Question Limits & Batching

Decision models impose different per-request limits:
- **Cloudflare Clef / Clef-flash**: Enforced at **64 questions max** by Workers AI backend.
- **TypeSafe Jev**: Recommended up to **32 questions** for optimal latency/consistency (allows more on OpenRouter).
- **Perplexity Decider**: Up to **128 questions**.
- **OpenAI GPT-6 Luna Decisions**: Up to **200 questions**.

Every decision model in `genai-tk` has a configured `max_questions` property. If a request exceeds `model.max_questions`, a descriptive `ValueError` is raised before making network calls. Use `model.batch_invoke(request)` or `await model.abatch_invoke(request)` to automatically slice questions into batches and merge all results.

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
