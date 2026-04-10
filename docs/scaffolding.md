# Project Scaffolding (`cli init`)

`cli init` initialises a new genai-tk project.  By default it generates a
**full project scaffold** — a working Python package with example CLI commands,
an LCEL chain, a Streamlit webapp page, and Copilot Agent support files — in
addition to copying the default `config/` tree and `Makefile`.

---

## Quick start

```bash
mkdir my-project && cd my-project
uv init                                       # create a bare uv project
uv add git+https://github.com/tclatos/genai-tk@main

uv run cli init --name "My AI Project"        # full scaffold (recommended)
uv run cli init                               # same, project name from cwd
uv run cli init --minimal                     # config + Makefile only, no example code
uv run cli init --deer-flow                   # also clone the Deer-flow backend

uv sync                                       # install the generated package
```

---

## What gets generated

### Config tree (`config/`)

Copied from the genai-tk defaults on first run; skipped if already present.

```
config/
├── app_conf.yaml          ← CLI commands, webapp navigation, service defaults
├── baseline.yaml          ← default LLM, embeddings, cache
├── overrides.yaml         ← empty — per-environment overrides
├── webapp.yaml            ← Streamlit app name, pages_dir, navigation
├── providers/
│   ├── llm.yaml           ← LLM model aliases
│   └── embeddings.yaml    ← Embeddings model aliases
└── agents/
    └── langchain.yaml     ← ReAct / deep / custom agent profiles
```

### Python package (`<package_name>/`)

Package name is derived from the project name (lowercased, spaces → underscores).

```
<package_name>/
├── __init__.py
├── main/
│   └── streamlit.py       ← Streamlit entry point (delegates to genai-tk)
├── commands/
│   └── example_commands.py  ← ExampleCommands: joke / chain / agent / deerflow
├── chains/
│   └── joke_chain.py      ← Simple LCEL chain registered with chain_registry
└── webapp/
    └── pages/
        └── demos/
            └── hello_agent.py  ← ReAct agent page with chat UI + calculator tool
```

### Copilot support

```
AGENTS.md                             ← Architecture overview and coding conventions
.github/
└── copilot-instructions.md           ← Always-active Copilot hints
```

### Project files

```
pyproject.toml    ← uv project with genai-tk dependency and scoped package discovery
README.md         ← Project overview and getting started guide
Makefile          ← make webapp / test / example-joke / example-chain / example-agent
```

---

## Example commands

After `uv sync`, the generated `ExampleCommands` group is immediately runnable:

```bash
# Show all example sub-commands
uv run cli example --help

# Primary LLM call via a simple LCEL chain
uv run cli example joke "software engineers"

# Pre-registered chain via the chain registry
uv run cli example chain "Python devs"

# ReAct agent with a built-in calculator tool
uv run cli example agent "What is 2 + 2?"

# Deer-flow deep-research agent (requires --deer-flow init)
uv run cli example deerflow "Explain transformer attention"
```

Or use the convenience `make` targets:

```bash
make example-joke    # cli example joke "programming"
make example-chain   # cli example chain "AI engineers"
make example-agent   # cli example agent "What is 2+2?"
```

---

## Config auto-patching

`cli init` patches several config files automatically to wire the generated
code into the toolkit's discovery mechanisms.

### `config/app_conf.yaml` — CLI command registration

Appends the generated command group to the `cli.commands` list:

```yaml
cli:
  commands:
    - ...existing entries...
    - my_project.commands.example_commands.ExampleCommands
```

### `config/webapp.yaml` — Streamlit navigation

Activates the generated pages directory and adds the hello-agent demo page:

```yaml
ui:
  app_name: My AI Project
  pages_dir: ${paths.project}/my_project/webapp/pages
  navigation:
    demos:
      - demos/hello_agent.py
```

`${paths.project}` is an OmegaConf interpolation that resolves to the project
root at config-load time, making the path work regardless of current working
directory.

### `Makefile` — Streamlit entry point

Sets `STREAMLIT_ENTRY` so `make webapp` launches the generated entry point
(which Streamlit uses to resolve the pages directory):

```make
STREAMLIT_ENTRY ?= my_project/main/streamlit.py
```

### `pyproject.toml` — package discovery

Adds scoped `setuptools` discovery so `uv sync` installs the package without
accidentally picking up `config/` as a namespace package:

```toml
[tool.uv]
package = true

[tool.setuptools.packages.find]
include = ["my_project*"]
```

---

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--name TEXT` | cwd name | Human-readable project name |
| `--minimal` | false | Config + Makefile only — skip example code |
| `--deer-flow` | false | Also clone the Deer-flow backend |
| `--path TEXT` | `~/deer-flow` | Clone path for Deer-flow |
| `--force` | false | Overwrite files that already exist |

`cli init` is **idempotent** — re-running it skips files that already exist
unless `--force` is passed.  Config patches are also idempotent (checked before
writing).

---

## Webapp

After `uv sync`:

```bash
make webapp          # launches Streamlit on http://localhost:8501
```

The **Hello Agent** demo page at `demos/hello_agent.py` shows:

- LLM selector (sidebar) — swap models without restarting
- Chat history display
- A built-in `calculator` tool wired to a ReAct agent

Extend the sidebar with `render_llm_selector()`, or add new pages following
the [add-webapp-page](../skills/copilot/add-webapp-page/SKILL.md) skill.

---

## Next steps

Once the scaffold is running, extend it:

| Goal | How |
|------|-----|
| Add a new CLI command group | Follow [skills/copilot/add-cli-command](../skills/copilot/add-cli-command/SKILL.md) |
| Add a Streamlit page | Follow [skills/copilot/add-webapp-page](../skills/copilot/add-webapp-page/SKILL.md) |
| Add an LCEL chain | Follow [skills/copilot/add-chain](../skills/copilot/add-chain/SKILL.md) |
| Configure an agent profile | Follow [skills/copilot/add-agent-profile](../skills/copilot/add-agent-profile/SKILL.md) |
| Add MCP servers | See [docs/mcp-servers.md](mcp-servers.md) |
| Enable Docker sandbox | See [docs/sandbox_support.md](sandbox_support.md) |

See [docs/copilot-agent-support.md](copilot-agent-support.md) for details on
how the generated Copilot files help GitHub Copilot understand and extend the project.
