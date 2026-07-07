# Unified Agent Profiles

All agent profiles — both LangChain (`react` | `deep` | `custom`, including the
DeepAgents SDK) and DeerFlow — live under one `agents:` top-level dict,
discriminated by the `harness:` field on each profile. `load_agent_profiles()`
merges every `*.yaml` file in this directory alphabetically, so each profile key
must be unique across files.

## Files

| File | Contents |
|------|----------|
| `defaults.yaml` | `agent_defaults:` — inheritable defaults applied to `harness: langchain` profiles (type, llm, middlewares, checkpointer, backend, skills, `default_profile`). DeerFlow profiles do **not** inherit from it. |
| `simple.yaml` | Lightweight ReAct agents (`simple`, `filesystem`, `eval_test`, `weather`). |
| `deep.yaml` | Deep agents with planning + file system (`research`, `coding`, `data_analysis`, `web_research`, `documentation_writer`, `stock_analysis`). |
| `browser.yaml` | Browser-automation deep agents (`browser_agent`, `browser_agent_direct`). |
| `text2sql.yaml` | SQL agents (`text2sql` deep, `chinook` react). |
| `deerflow.yaml` | DeerFlow profiles (`Research Assistant`, `Web Browser`, `Chinook DB + Research`, `Privacy-Safe Research`). |

DeerFlow **runtime settings** (skills mount, `general` title/summarization/memory,
`recursion_limit`, `default_profile`) live in `config/examples/deerflow.yaml` —
not here, since they are runtime config rather than agent profiles.

## Schema

```yaml
agent_defaults:          # optional, langchain-only inheritance
  type: react
  middlewares:
    - class: genai_tk.agents.langchain.middleware.rich_middleware.RichToolCallMiddleware
  default_profile: simple

agents:
  research:              # profile key/slug — used by `cli agents run research`
    harness: langchain   # langchain | deerflow
    name: "Research"     # display name
    type: deep           # react | deep | custom  (langchain only)
    ...
  "Research Assistant":
    harness: deerflow
    mode: pro
    ...
```

A profile's dict key is its identifier (slug), used by `cli agents run <key>`
and `create_harness(<key>)`. When the profile omits `name`, the slug is injected
as the name.

## Usage

```bash
uv run cli agents list                                   # all profiles, both harnesses
uv run cli agents run research "Summarise recent AI news" # auto-resolves the harness
uv run cli agents run research --chat                     # --chat/--mode/--sandbox/--mcp work across harnesses
uv run cli agents run "Research Assistant" --chat         # DeerFlow profile (auto-resolved)
```

See [docs/agents.md](../../../docs/agents.md) and [docs/harness.md](../../../docs/harness.md).
