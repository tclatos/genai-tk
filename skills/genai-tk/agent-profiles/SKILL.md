---
name: agent-profiles
description: Build or modify LangChain, DeepAgent, DeerFlow profiles, agent tools, middleware, checkpointing, skills wiring, and the shared harness layer in genai-tk.
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

Profiles are dict-keyed under one unified `agents:` dict (see
`config/examples/agents/*.yaml`). The key is what users pass to
`cli agents run <key>`. `harness: langchain` profiles inherit from
`agent_defaults` in `defaults.yaml`; DeerFlow profiles (`harness: deerflow`)
are fully self-describing.

```yaml
agents:
  research:
    harness: langchain          # langchain | deerflow (default: langchain)
    name: "Research"
    type: deep                  # react (default) | deep | custom
    llm: gpt_41@openai
    tools:
      - factory: genai_tk.agents.tools.langchain.search_tools_factory.create_search_function
        verbose: true            # extra keys become factory kwargs
    skill_directories:
      - ${paths.project}/skills
```

Tool specs are flat dicts discriminated by `class:` / `function:` / `factory:`
key (see `genai_tk/agents/tools/tool_specs.py`); any other key is passed
through as a constructor/factory kwarg.

## Choosing the Agent Type — and Whether Skills Apply

| Type | Tool-use loop | Planning/subagents | Sandbox backend | Skills honored at runtime? |
|---|---|---|---|---|
| `react` (default) | ✓ | ✗ | ✗ | **No** — `skill_directories` is parsed but silently ignored, no `SkillsMiddleware` wired |
| `deep` | ✓ | ✓ | optional (`AioSandboxBackend`) | **Yes** — via deepagents' `SkillsMiddleware` |
| `custom` | your code | your code | your code | No — unless you wire it yourself |
| `harness: deerflow` (any mode) | ✓ | ✓ (mode-dependent) | optional (`local`/`docker`) | **Yes** — `profile.skills` / `skill_directories` are loaded by `genai_tk/agents/deer_flow/config_bridge.py` and passed to the upstream DeerFlow runtime |

**Key gotcha:** for `harness: langchain` profiles, `skill_directories` (and
`subagents`, deep-only `backend` types) are parsed for every profile but
only `_create_deep_agent` in `genai_tk/agents/langchain/factory.py` wires
them into the running agent. A `type: react` profile with
`skill_directories:` set will load and run fine — the SKILL.md content is
simply never injected into the model's context. DeerFlow profiles
(`harness: deerflow`) are the exception to this rule: they honor skills
regardless of `mode` (flash/thinking/pro/ultra) because DeerFlow wires its
own skill loading independently of the LangChain react/deep split — see
`genai_tk/agents/deer_flow/config_bridge.py::load_skills_from_directories`.

If a task needs progressive-disclosure domain knowledge at *runtime*, use
`type: deep` (LangChain) or any DeerFlow profile. If a SKILL.md is only
meant to guide a human or a coding agent (like Copilot) while *building*
the profile, `react` is fine and the skill never needs to be wired into
`skill_directories` at all.

Default to `react` for simple tool-calling agents; switch to `deep` (or
DeerFlow) only when you need multi-step planning, subagents, a sandbox
backend, or runtime skill injection.

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
2. For profile-only changes, edit `config/agents/**.yaml` and verify with `uv run cli agents list`.
3. For tools, expose a factory under `genai_tk/agents/tools/...` and reference it from YAML.
4. For middleware, add a Pydantic config model if the YAML accepts options.
5. Add structural tests under `tests/unit_tests/agents/` and integration tests only for real agent behavior.

## Commands

```bash
uv run cli agents list
uv run cli agents run simple "What is 2+2?"
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/agents -q
```

## Avoid

- Do not use display `name` where the profile key is required.
- Do not hardcode system prompts in Python when they belong in YAML.
- Do not pass credentials through normal tools; use secure credential tooling for browser profiles.
