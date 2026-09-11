# Project Scaffolding (`cli init`)

`cli init` bootstraps a new genai-tk project in the current directory. It copies
the default `config/` tree, scaffolds a Python package with agent infrastructure,
merges tiered skills, and generates `AGENTS.md` and a `justfile` — everything you need for AI-assisted coding.

---

## Quick start

```bash
mkdir my-project && cd my-project
uv init
uv add "genai-tk @ git+https://github.com/tclatos/genai-tk@main"

# Bootstrap a standard agent application:
uv run cli init --name "My AI Project"

# Bootstrap with GenAI Graph and Benchmark framework support:
uv run cli init --name "My Benchmark Suite" --with-graph --graph-path ../genai-graph

# Optional: install extras during init
uv run cli init --extra harnessing --extra browser

# Done!
uv sync
just run                           # start the application
```

---

## What `cli init` does

1. **Copies config/** — LLM/embedding providers, agent profiles, MCP server configs, webapp settings, and optional `bench.yaml`.
2. **Scaffolds package** — Python module with CLI commands, tools, starter benchmark adapter, skills, and webapp pages.
3. **Installs & Merges Skills** — Copies 4-tier skills (`runtime`, `development`, `governance`, `vendor`) from `genai-tk` and merges `genai-graph` skills when `--with-graph` is passed.
4. **Generates docs** — `AGENTS.md` (architecture map), `EXTENDING.md` (how-to guide), and `SKILLS.md`.
5. **Configures IDE support** — `.github/copilot-instructions.md` (auto-loaded by Copilot).
6. **Sets up workflows** — `justfile` for common tasks (lint, test, skills, run).

---

## Flags & Options

| Option | Flag | Description |
|---|---|---|
| `--name` | `-n` | Human-readable project name (determines package name). |
| `--with-graph` | `-g` | Enable `genai-graph` dependency, benchmark framework CLI (`cli bench`), starter `adapter.py`, and `config/bench.yaml`. |
| `--graph-path` | | Path to local editable `genai-graph` checkout (defaults to `../genai-graph`). |
| `--extra` | `-e` | Install optional extras at init time (repeatable: `harnessing`, `browser`, `nlp`, `postgres`, `streamlit`, `baml`, `chromadb`). |
| `--force` | `-f` | Overwrite existing files. |

---

## Generated Structure

```
config/                           ← copied from genai-tk defaults
  app_conf.yaml                   ← CLI command registry (includes AgentCommands & BenchCommands)
  agents.yaml                     ← unified agent profiles (default + research)
  bench.yaml                      ← benchmark profiles & adapter reference (when --with-graph)
  providers/
    llm.yaml                      ← LLM model definitions
    embeddings.yaml               ← embedding model definitions
  *.yaml                          ← other configs (mcp, webapp, markdownize, etc.)

<my_project>/                     ← Python package
  __init__.py
  adapter.py                      ← starter BaseBenchmarkAdapter (when --with-graph)
  commands/
    agent_commands.py             ← AgentCommands CLI group
    bench_commands.py             ← Benchmark CLI group (when --with-graph)
  tools/
    example_tool.py               ← example LangChain tool
  webapp/
    pages/demos/
      hello_agent.py              ← demo: chat with ReAct agent
  main/
    streamlit.py                  ← Streamlit app entry point

skills/                           ← 4-tier skills architecture
  runtime/                        ← runtime user capabilities (query-writing, browser-automation, kg-query)
  development/                    ← developer & scaffolding skills (benchmark-framework, agent-profiles, kg-schema)
  governance/                     ← quality & consistency (evaluation-testing, repo-map, pii-anonymization)
  vendor/                         ← imported third-party skills (atos-slidev)
  custom/                         ← project-local custom skills
  community/                      ← installed via skills.sh

AGENTS.md                         ← architecture map for AI agents
justfile                          ← task runner: just run / just lint / just skills
README.md                         ← project overview
pyproject.toml                    ← package config with dependencies and [tool.uv.sources]

.github/
  copilot-instructions.md         ← Copilot agent instructions (points to AGENTS.md)
```
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
uv run cli agents run "Research Assistant" --chat
```

See [docs/deer-flow.md](deer-flow.md) for profiles and advanced usage.

### AIO Sandbox (Docker-based code execution)

```bash
uv run cli init --with-sandbox
```

Installs `agent-sandbox`, `opensandbox`, `opensandbox-server` packages. Then:

```bash
opensandbox-server start
uv run cli agents run coding "write and run code"   # coding profile has backend: aio_sandbox
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

Yes. Edit `config/agents.yaml` to add more profiles. Use `cli agents run <profile>` to select.

**Q: How do I add my own tools?**

Create a file in `<package>/tools/` and register it in an agent profile's `tools:` section. See `docs/EXTENDING.md`.

**Q: I want to scaffold multiple projects in the same directory.**

`cli init` is idempotent and skips existing files. To regenerate, use `--force`.
