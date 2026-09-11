---
name: kg-docgraph-agent
description: Build and run the genai-graph Document Graph deep agent — genai_graph/agent/docgraph_agent.py (create_docgraph_agent, prepare_docgraph_agent, run_docgraph_agent, resolve_db_path, build_docgraph_system_prompt), the colocated runtime skills under genai_graph/agent/skills, the cli docgraph agent command, the deepseek_v4flash@openrouter LLM registry entry, and the docgraph profile. Use when editing the agent module, its runtime skills, the agent CLI, or wiring a downstream project (e.g. rfq_pricing) to extract information from a document folder.
---

# Document Graph Deep Agent

## Read First

- `genai_graph/agent/docgraph_agent.py` — `create_docgraph_agent`, `prepare_docgraph_profile`, `run_docgraph_agent`, `resolve_db_path`, `build_docgraph_system_prompt`
- `genai_graph/agent/skills/navigate-document-graph/SKILL.md` — runtime skill: the vectorless agentic-RAG loop
- `genai_graph/agent/skills/document-graph-tools/SKILL.md` — runtime skill: tool + schema reference
- `genai_graph/core/commands_docgraph.py` — `cli docgraph agent` subcommand
- `config/agents/docgraph.yaml` — unified `agents:` profile (`docgraph`, type: deep)
- `config/providers/llm.yaml` — `deepseek_v4flash` → `deepseek/deepseek-v4-flash` on openrouter
- `genai_graph/kg/query/document_graph_tools.py` — the navigation tools the agent calls

## What it is

A genai-tk `type: deep` agent (DeepAgents SDK) that answers questions by
**navigating** the Ladybug Document Graph with read-only tools — no embeddings,
no chunking. The agent orients with `get_folder_toc`, gets a document's section
map with `get_document_toc`, reads only the relevant sections with
`get_section_content`, and keyword-searches with `search_sections`.

The navigation tools and target folder are **injected at runtime** by
`create_docgraph_agent` (via the harness `extra_tools`) so `--db` / `--folder` /
`--llm` overrides work without editing the profile YAML.

## Skills

Runtime skills ship colocated with the agent under `genai_graph/agent/skills/`:

- `navigate-document-graph` — the navigation loop and grounding/citation rules.
- `document-graph-tools` — exact tool arguments, return shapes, and the
  Folder → Document → MarkdownSection schema.

They are resolved from the **package location** (`Path(__file__).parent/"skills"`),
so they load whether the agent is launched from genai-graph or a downstream
project. `prepare_docgraph_profile` sets a `FilesystemBackend` rooted at the
**common ancestor** of all skill directories (package skills + any caller/project
skills) so DeepAgents' `SkillsMiddleware` can read them under `virtual_mode=True`.

## Public API

```python
from genai_graph.agent import (
    create_docgraph_agent,  # profile -> LangChainHarness (tools + skills + folder injected)
    prepare_docgraph_profile,  # mutate profile in place (system prompt, skills, backend)
    run_docgraph_agent,  # async: stream one turn, return assistant text
    resolve_db_path,  # db_path or graph_db.default
    create_document_graph_tools_from_config,
    build_docgraph_system_prompt,
    DEFAULT_LLM,  # "deepseek_v4flash@openrouter"
    DEFAULT_PROFILE,  # "docgraph"
)
```

`create_docgraph_agent(profile, *, llm=None, db_path=None, folder_id=None,
extra_skill_dirs=None)` → `LangChainHarness`. The harness lazily compiles the
deep agent on first use. Use `run_docgraph_agent(harness, query)` to stream a
one-shot turn, or `genai_tk.agents.harness.chat_repl.run_chat_repl(harness)` for
an interactive REPL.

## CLI

```bash
cli docgraph agent "What IT services is Alko requesting?" --folder folder_273e65da416b2e72
cli docgraph agent --chat --folder folder_273e65da416b2e72
cli docgraph agent "Summarize the SLAs" --llm deepseek_v4flash@openrouter --db ./data/kg/tree.db
cli docgraph agent "..." --profile docgraph --skill-dir ./extra-skills --recursion-limit 160
```

## Wiring a downstream project (e.g. rfq_pricing)

1. Add `deepseek_v4flash` to the project's `config/providers/llm.yaml` (openrouter
   slug `deepseek/deepseek-v4-flash`). Reference it as `deepseek_v4flash@openrouter`
   (the bare id fuzzy-resolves to azure).
2. Add a unified `agents:` profile (`type: deep`) in `config/agents/<name>.yaml`
   with the use-case `system_prompt` and `skill_directories` pointing at the
   project's use-case skills (e.g. `${paths.project}/skills/custom`).
3. In a CLI command, call `create_docgraph_agent(profile, llm=..., db_path=...,
   folder_id=..., extra_skill_dirs=[...])` then `run_docgraph_agent`. The generic
   `genai_graph/agent/skills` are injected automatically; pass the project's
   use-case skills via `extra_skill_dirs` or list them in the profile.
4. Install the `harnessing` extra (`uv sync --extra harnessing`) — `deepagents` is
   required for `type: deep` agents.

## Change Workflow

1. Navigation-tool changes live in `document_graph_tools.py` (keep them
   schema-tolerant); the agent calls them by name.
2. Navigation-loop guidance changes go in the `navigate-document-graph` skill.
3. Profile/LLM changes: `config/agents/docgraph.yaml` and `config/providers/llm.yaml`.
4. Keep `prepare_docgraph_profile`'s common-ancestor backend-root logic in sync
   with DeepAgents' `FilesystemBackend` virtual-mode path checks.

## Commands

```bash
uv run cli docgraph agent "..." --folder folder_273e65da416b2e72
uv run ruff check genai_graph/agent genai_graph/core/commands_docgraph.py
uv run just test
```

## Avoid

- Do not put the runtime skills under `skills/genai-graph/` — that tree holds
  **developer** skills (for editing the codebase). Runtime skills belong in
  `genai_graph/agent/skills/` so they ship with the package.
- Do not reference the LLM as a bare `deepseek_v4flash` in profiles — use the
  explicit `deepseek_v4flash@openrouter` id (the bare id auto-resolves to azure).
- Do not inject the navigation tools via the profile `tools:` list — they depend
  on `--db`, so inject them at runtime via `create_docgraph_agent`/`extra_tools`.
- Do not add embeddings/Chunk nodes to support the agent — it is deliberately
  vectorless.

## Complements

- `kg-document-graph` — the Document Graph schema, factories, and navigation tools.
- `kg-query` — the broader Cypher/text-to-Cypher query story.
- `genai-tk/agent-profiles` — the unified `agents:` profile format and `type: deep`.
