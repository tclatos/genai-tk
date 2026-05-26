# Project Scaffolding (`cli init`)

`cli init` bootstraps a new genai-tk project in the current directory. It copies
the default `config/` tree, scaffolds a Python package from a **template preset**,
and generates `AGENTS.md`, a `justfile`, and multi-agent support files so AI coding
assistants (Copilot, Cursor, Windsurf, Codex) work out of the box.

---

## Quick start

```bash
mkdir my-project && cd my-project
uv init
uv add git+https://github.com/tclatos/genai-tk@main

uv run cli init                             # interactive template picker (recommended)
uv run cli init -t agent-app --name "My AI Project"
uv run cli init --deer-flow                 # also clone the Deer-flow backend

uv sync
just run                                    # start the application
```

---

## Templates

Choose a template with `--template / -t`. When omitted, an interactive picker is shown.

| Template | What you get | Run command |
|----------|-------------|-------------|
| `agent-app` | Tools, agent profiles, skills, webapp page | `cli agents langchain --chat` |
| `rag-app` | Document ingestion, vector store, retrieval | `cli rag query "…"` |
| `workflow-app` | YAML-driven multi-step pipeline + Prefect | `cli workflow run example` |
| `minimal` | Config + justfile only, no example code | `cli --help` |

```bash
# Interactive picker (uses questionary; falls back to text prompt)
uv run cli init

# Explicit template
uv run cli init -t agent-app
uv run cli init -t rag-app     --name "My RAG App"
uv run cli init -t workflow-app
uv run cli init -t minimal

# Overwrite existing files
uv run cli init -t agent-app --force
```

---

## What gets generated

### Common (all templates)

```
config/               ← copied from genai-tk defaults
AGENTS.md             ← architecture map for AI agents (Codex, Claude, OpenCode, …)
justfile              ← task runner (just run / just lint / just skills)
README.md             ← project overview
pyproject.toml        ← package mode enabled, genai-tk dependency
docs/
├── SKILLS.md         ← skills guide: format, skills.sh, community sources
└── EXTENDING.md      ← how to add CLI commands, tools, chains, webapp pages
.github/
└── copilot-instructions.md   ← lean pointer to AGENTS.md (auto-loaded by Copilot)
.cursor/
└── rules/genai-tk.mdc        ← Cursor rules
.windsurfrules                ← Windsurf rules
skills/
├── custom/           ← your skills (committed to repo)
├── community/        ← installed via `cli skills add` (gitignored)
└── bundled/          ← copies of genai-tk bundled skills (optional)
```

### `agent-app` additions

```
<package>/
├── __init__.py
├── commands/
│   └── agent_commands.py     ← AgentCommands CLI group
├── tools/
│   └── example_tool.py       ← example LangChain tool
├── webapp/pages/demos/
│   └── hello_agent.py        ← Streamlit chat UI
└── main/
    └── streamlit.py          ← Streamlit entry point
config/agents/
└── langchain.yaml            ← default + research agent profiles
skills/custom/
└── getting-started/
    └── SKILL.md              ← example skill with skills.sh format
```

### `rag-app` additions

```
<package>/
├── __init__.py
└── commands/
    └── rag_commands.py       ← RagCommands CLI group (ingest + query)
data/
├── raw/                      ← drop source documents here
└── processed/
```

### `workflow-app` additions

```
<package>/
├── __init__.py
└── workflows/
    └── steps/
        └── example_step.py   ← Prefect flow step
config/workflows/
└── pipeline.yaml             ← workflow + profile YAML
```

---

## Skills system

Every project gets a `skills/` tree and `docs/SKILLS.md` guide out of the box.
Skills are `SKILL.md` files (YAML frontmatter + markdown) that give agents
domain knowledge on demand — loaded progressively, not injected on every call.

```bash
# List all skills in the project
just skills
cli skills list

# Add a bundled skill (from genai-tk)
cli skills add getting-started

# Install from a GitHub repo (skills.sh format)
cli skills add --skillssh langchain-ai/langchain-skills

# Install from a git repo
cli skills add --git https://github.com/your-org/my-skills --path my-skill

# Create a new skill interactively
cli skills create my-domain-skill

# Validate all skills
cli skills validate --all
```

See [docs/skills.md](../docs/skills.md) *(if present)* or the generated `docs/SKILLS.md`
in your project for the complete reference, including skills.sh format and community sources.

---

## Config auto-patching

`cli init` patches several config files automatically after rendering templates.

### `config/app_conf.yaml` — CLI command registration

```yaml
cli:
  commands:
    - genai_tk.main.cli.register_commands
    - my_project.commands.agent_commands.AgentCommands   # ← appended
```

### `config/webapp.yaml` — Streamlit navigation (agent-app only)

```yaml
ui:
  pages_dir: ${paths.project}/my_project/webapp/pages
  navigation:
    demos:
      - demos/hello_agent.py
```

### `pyproject.toml` — package discovery

```toml
[tool.uv]
package = true

[tool.setuptools.packages.find]
include = ["my_project*"]
```

All patches are idempotent — re-running `cli init` is safe.

---

## CLI reference

| Option | Default | Description |
|--------|---------|-------------|
| `--template / -t TEXT` | interactive | Template: `agent-app` \| `rag-app` \| `workflow-app` \| `minimal` |
| `--name / -n TEXT` | cwd name | Human-readable project name |
| `--deer-flow / -d` | false | Also clone the Deer-flow backend |
| `--deer-flow-path TEXT` | `~/deer-flow` | Clone path for Deer-flow |
| `--force / -f` | false | Overwrite files that already exist |

`cli init` is **idempotent** — re-running skips files that already exist unless `--force` is passed.

---

## Multi-agent support

The generated files target four AI coding assistants:

| File | Tool |
|------|------|
| `AGENTS.md` | Codex, Claude Code, OpenCode, Gemini CLI (primary source of truth) |
| `.github/copilot-instructions.md` | GitHub Copilot (auto-injected, ≤25 lines, pointer to AGENTS.md) |
| `.cursor/rules/genai-tk.mdc` | Cursor |
| `.windsurfrules` | Windsurf |

All files follow the **progressive disclosure** principle: agents get a concise
map with pointers to skills and docs, not a full manual.

---

## justfile tasks

```bash
just          # list all tasks
just run      # start the application
just lint     # ruff format + check + cli skills validate --all
just skills   # cli skills list
just webapp   # uv run cli webapp  (agent-app template)
```

---

## Webapp (agent-app)

After `uv sync`:

```bash
just webapp   # or: uv run cli webapp
```

The **Hello Agent** demo page shows a chat UI with LLM selector and a built-in
`calculator` tool wired to a ReAct agent.

---

## Extending the project

See the generated `docs/EXTENDING.md` and `AGENTS.md` in your project for
step-by-step guides on:

- Adding a CLI command group
- Adding a LangChain tool
- Adding an agent profile
- Adding a webapp page
- Adding a SKILL.md
- Adding an MCP server

Or use the toolkit's own contributor skills:

```bash
cli skills add add-tool       # bundled skill: how to create a tool
cli skills add add-skill      # bundled skill: how to create a skill
cli skills add add-mcp-server # bundled skill: how to add an MCP server
```
