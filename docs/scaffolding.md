# Project Scaffolding (`cli init`)

`cli init` bootstraps a new genai-tk project in the current directory. It copies
the default `config/` tree, scaffolds a Python package with agent infrastructure,
and generates `AGENTS.md` and a `justfile` — everything you need for AI-assisted coding.

---

## Quick start

```bash
mkdir my-project && cd my-project
uv init
uv add git+https://github.com/tclatos/genai-tk@main

# Bootstrap the project (always scaffolds agent-app structure)
uv run cli init --name "My AI Project"

# Optional: add heavy components later
uv run cli init --with-deer-flow   # install deerflow-harness + config
uv run cli init --with-sandbox     # install aio-sandbox (Docker support)

# Done!
uv sync
just run                           # start the application
```

---

## What `cli init` does

1. **Copies config/** — LLM/embedding providers, agent profiles, MCP server configs, webapp settings
2. **Scaffolds package** — Python module with CLI commands, tools, skills, webapp pages
3. **Generates docs** — `AGENTS.md` (architecture map), `EXTENDING.md` (how-to guide)
4. **Configures IDE support** — `.github/copilot-instructions.md` (auto-loaded by Copilot)
5. **Sets up workflows** — `justfile` for common tasks (lint, skills, run)

All scaffolding follows **agent-friendly defaults**: no IDE-specific files (Cursor/Windsurf rules are
shown as post-init commands instead), modular structure for tools/skills/chains.

---

## Generated structure

### Always generated (agent-app structure)

```
config/                           ← copied from genai-tk defaults
  app_conf.yaml                   ← CLI command registry
  agents/
    langchain.yaml                ← agent profiles (default + research)
  providers/
    llm.yaml                      ← LLM model definitions
    embeddings.yaml               ← embedding model definitions
  *.yaml                          ← other configs (mcp, webapp, workflows, etc.)

<my_project>/                     ← Python package
  __init__.py
  commands/
    agent_commands.py             ← AgentCommands CLI group (auto-registered)
  tools/
    example_tool.py               ← example LangChain tool
  utils/
    __init__.py
  webapp/
    pages/demos/
      hello_agent.py              ← demo: chat with ReAct agent
  main/
    streamlit.py                  ← Streamlit app entry point

data/                             ← runtime data
  kv_store/                       ← vector store, checkpoints, cache
  models_dev.json                 ← downloaded model metadata

docs/
  EXTENDING.md                    ← how to add CLI commands, tools, chains, webapp pages
  SKILLS.md                       ← skills reference (created by cli skills add)

skills/
  custom/                         ← your SKILL.md files (committed)
  community/                      ← installed via cli skills add (gitignored)

AGENTS.md                         ← architecture map for AI agents
justfile                          ← task runner: just run / just lint / just skills
README.md                         ← project overview
pyproject.toml                    ← package config with genai-tk dependency

.github/
  copilot-instructions.md         ← Copilot agent instructions (points to AGENTS.md)

.gitignore                        ← includes /skills/community, /data/*, etc.
```

### Optional (`--with-deer-flow`)

```
config/agents/
  deerflow.yaml                   ← Deer-flow profiles (chat + research by default)

# In your Python environment:
# uv add "deerflow-harness @ git+https://github.com/bytedance/deer-flow@main#subdirectory=backend/packages/harness"
```

### Optional (`--with-sandbox`)

```
# In your Python environment:
# uv sync --group aio-sandbox
# Installs: agent-sandbox, opensandbox, opensandbox-server
```

---

## IDE setup

After `cli init`, copy `AGENTS.md` to your IDE's rules location:

```bash
# Cursor
cp AGENTS.md .cursor/rules/project.md

# Windsurf
cp AGENTS.md .windsurfrules

# Claude Code / OpenCode / Codex
# Use the file as-is (they auto-discover AGENTS.md)
```

Or ask your AI assistant directly to read `AGENTS.md` for context.

---

## Optional heavy components

By default, `cli init` keeps things lightweight. Install heavy components on demand:

### DeerFlow (multi-agent reasoning with planning)

```bash
uv run cli init --with-deer-flow
```

Installs `deerflow-harness` package + config profiles. Then:

```bash
uv run cli agents deerflow --chat -p "Research Assistant"
```

See [docs/deer-flow.md](deer-flow.md) for profiles and advanced usage.

### AIO Sandbox (Docker-based code execution)

```bash
uv run cli init --with-sandbox
```

Installs `agent-sandbox`, `opensandbox`, `opensandbox-server` packages. Then:

```bash
opensandbox-server start
uv run cli agents langchain --sandbox docker "write and run code"
```

See [docs/sandbox_support.md](sandbox_support.md) for setup and configuration.

---

## Skills system

Every project gets a `skills/` directory for SKILL.md files — YAML+markdown documents
that give agents domain knowledge on demand.

```bash
# List skills in this project
just skills

# Add a bundled skill (from genai-tk)
cli skills add getting-started

# Install community skills
cli skills add --skillssh langchain-ai/langchain-skills

# Create a new skill
cli skills create my-domain-skill
```

See [docs/skills.md](skills.md) for the complete guide.

---

## Config auto-patching

`cli init` patches several config files automatically:

### `config/app_conf.yaml` — CLI commands

Your project's `AgentCommands` class is auto-registered:

```yaml
cli:
  commands:
    - genai_tk.main.cli.register_commands
    - my_project.commands.agent_commands.AgentCommands   # ← added
```

### `config/webapp.yaml` — Streamlit pages

If you have a webapp, pages are registered automatically:

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

All patches are **idempotent** — re-running `cli init` is safe.

---

## CLI reference

| Option | Default | Description |
|--------|---------|-------------|
| `--name / -n TEXT` | cwd name | Human-readable project name |
| `--with-deer-flow` | false | Install deerflow-harness (heavy: multi-agent planning) |
| `--with-sandbox` | false | Install aio-sandbox (heavy: Docker code execution) |
| `--force / -f` | false | Overwrite files that already exist |

`cli init` is **idempotent** — re-running skips files that exist unless `--force` is set.

---

## justfile tasks

```bash
just              # list all tasks
just run          # start the webapp (uv run cli webapp)
just lint         # ruff format + check + cli skills validate
just skills       # cli skills list
just test         # run unit tests
```

---

## Next steps

1. **Run the app**: `uv sync && just run`
2. **Try the agent**: Open http://localhost:8501 → "Hello Agent" demo
3. **Extend**: Add tools, skills, profiles — see `docs/EXTENDING.md`
4. **Deploy**: Check `docs/` for deployment guides

---

## Troubleshooting

**Q: How do I use a different LLM?**

Edit `config/profiles/local/providers/llm.yaml` (or your active profile) and change the `default` model.

**Q: Can I have multiple agent profiles?**

Yes. Edit `config/agents/langchain.yaml` to add more profiles. Use `cli agents langchain -p <profile>` to select.

**Q: How do I add my own tools?**

Create a file in `<package>/tools/` and register it in an agent profile's `tools:` section. See `docs/EXTENDING.md`.

**Q: I want to scaffold multiple projects in the same directory.**

`cli init` is idempotent and skips existing files. To regenerate, use `--force`.
