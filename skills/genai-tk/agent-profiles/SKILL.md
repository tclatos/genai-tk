---
name: genai-tk-agent-profiles
description: Build or modify LangChain, DeepAgent, SmolAgents, DeerFlow profiles, agent tools, middleware, checkpointing, and skills wiring in genai-tk.
---

# GenAI Toolkit Agent Profiles

## Read First

- `docs/agents.md`
- `docs/deer-flow.md`
- `docs/middleware-pii-and-routing.md`
- `genai_tk/agents/langchain/config.py`
- `genai_tk/agents/langchain/factory.py`
- `config/agents/langchain/defaults.yaml`

## Configuration Shape

LangChain profiles are dict-keyed. The key is what users pass to `-p`.

```yaml
langchain_agents:
  research:
    name: Research
    type: deep
    llm: gpt_41@openai
    tools:
      - spec: web_search
    skills:
      directories:
        - ${paths.project}/skills
```

## Code Map

| Concern | Paths |
|---|---|
| LangChain profiles and factory | `genai_tk/agents/langchain/` |
| Middleware | `genai_tk/agents/langchain/middleware/` |
| Tool specs and factories | `genai_tk/agents/tools/` |
| SmolAgents CLI | `genai_tk/agents/smolagents/` |
| DeerFlow bridge | `genai_tk/agents/deer_flow/` |
| DeepAgent CLI bridge | `genai_tk/agents/deepagent_cli/` |
| Sandbox backend | `genai_tk/agents/sandbox/`, `genai_tk/agents/langchain/sandbox_backend.py` |

## Change Workflow

1. Decide whether the change is profile-only, a new tool factory, middleware, or agent runtime behavior.
2. For profile-only changes, edit `config/agents/**.yaml` and verify with `uv run cli agents langchain --list`.
3. For tools, expose a factory under `genai_tk/agents/tools/...` and reference it from YAML.
4. For middleware, add a Pydantic config model if the YAML accepts options.
5. Add structural tests under `tests/unit_tests/agents/` and integration tests only for real agent behavior.

## Commands

```bash
uv run cli agents langchain --list
uv run cli agents langchain -p simple "What is 2+2?"
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/agents -q
```

## Avoid

- Do not use display `name` where the profile key is required.
- Do not hardcode system prompts in Python when they belong in YAML.
- Do not pass credentials through normal tools; use secure credential tooling for browser profiles.
