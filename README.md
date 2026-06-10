# GenAI Toolkit (`genai-tk`)

A toolkit for building Gen AI and Agentic applications with LangChain, LangGraph, and 100+ LLM providers.

See also the [great DeepWiki generated documentation](https://deepwiki.com/tclatos/genai-tk).

## Three Domains

The toolkit is organized around three complementary domains:

### 🧠 **Core GenAI**
Build intelligent applications with multi-provider LLM/embeddings support and state management.
- Multi-provider LLM and embeddings factory (OpenAI, Groq, Anthropic, Ollama, local, …)
- Vector stores: Chroma, PgVector, in-memory
- LLM caching, prompt templates, structured output
- See: `cli core llm`, `cli info models`, [docs/core.md](docs/core.md)

### 🤖 **Agents**
Four agent frameworks (ReAct, Deep, Deer-flow, SmolAgents) sharing YAML profiles, LLM factory, tools, and Docker sandbox.
- **ReAct** — standard Thought → Action → Observation loop (LangChain)
- **Deep agent** — multi-step planning + subagent delegation (LangChain)
- **Deer-flow** — native web research, multi-agent orchestration (LangGraph / ByteDance)
- **SmolAgents** — code-first automation (Hugging Face)
- **Skills system** — `SKILL.md` domain-knowledge files loaded on demand; managed with `cli skills`
- Docker sandbox — isolated execution, browser automation
- MCP servers — protocol-standard tool integration
- See: `cli agents`, [docs/agents.md](docs/agents.md), [AGENTS.md](AGENTS.md)

### ⚙️ **Workflows**
Orchestrate multi-step AI pipelines with Prefect and a YAML DSL for composable, reusable workflows.
- **Workflow DSL** — YAML-configured steps, dependencies, and sub-workflows (no Python needed)
- **Prefect server** — explicit local server managed via `cli prefect start/stop/status`; auto-starts before workflow runs
- **Document pipelines** — markdownize, OCR, PDF extraction, chunking
- **RAG pipeline** — full retrieval pipeline with BM25 + dense hybrid search
- **Structured extraction** — BAML-based extraction with type-safe output
- See: `cli workflow`, `cli prefect`, [docs/workflows.md](docs/workflows.md), [docs/prefect.md](docs/prefect.md)

---

**What it gives you:**
- YAML-configured profiles — swap models, tools, MCP servers, and sandboxes without code changes
- Rich CLI that mirrors every capability; easily extensible with one class + one YAML line

---

## Installation

**Start from scratch:**

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new project directory
mkdir my-genai-project && cd my-genai-project

# Initialize a Python project (creates pyproject.toml and a stub main.py)
uv init

# Add genai-tk to your project
uv add git+https://github.com/tclatos/genai-tk@main

# Interactive template picker (recommended)
# cli init scaffolds config/, skills/, justfile, sets hatchling as the build
# backend, removes the uv-init stub (main.py), and runs `uv sync` automatically.
uv run cli init

# Or choose a template directly:
uv run cli init -t agent-app  --name "My AI Project"   # tools, skills, agent profiles
uv run cli init -t rag-app    --name "My RAG App"       # document ingestion + retrieval
uv run cli init -t workflow-app                         # YAML-driven pipelines
uv run cli init -t minimal                              # config + justfile only

# (optional) also clone the Deer-flow backend
uv run cli init --deer-flow

just run                                                # start the application

# After running cli init --deer-flow, add this to your .env:
# DEER_FLOW_PATH=~/deer-flow  (or wherever you cloned it)
```

**Add to existing project:**

```bash
# Add to your project  
uv add git+https://github.com/tclatos/genai-tk@main

# With PostgreSQL / Playwright extras
uv add "genai-tk[extra] @ git+https://github.com/tclatos/genai-tk@main"

# Initialize config in the current directory
# (scaffolds config/, skills/, justfile, hatchling build backend, runs uv sync)
uv run cli init                      # interactive template picker
uv run cli init -t agent-app         # agent app with tools + skills
uv run cli init --deer-flow          # also clone the Deer-flow backend
```

**Development (clone & edit):**

```bash
git clone https://github.com/tclatos/genai-tk.git && cd genai-tk
uv sync              # core + dev
uv sync --all-groups # + postgres, browser, evals
```

> **Using your clone in another project:** add it as a local path dependency so changes are picked up immediately:
> ```bash
> uv add --editable /path/to/genai-tk
> ```

**Add your API key** to `.env` in the project root (auto-loaded at startup):

```bash
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=...  GROQ_API_KEY=...  etc.
```

---

## Quick Start by Domain

### 🧠 Core GenAI

```bash
# Call any LLM
uv run cli core llm -i "tell me a joke" -m gpt-4o-mini@openai --stream

# List available models (1000+ supported)
uv run cli info models

# Show active config and API key status
uv run cli info config
```

**Python:**
```python
from genai_tk.core.factories import get_llm

llm = get_llm()  # uses default from config
response = llm.invoke("Tell me a joke")
print(response.content)
```

---

### 🤖 Agents

```bash
# ReAct agent — interactive chat
uv run cli agents langchain --chat

# Single query with a specific profile (by KEY)
uv run cli agents langchain -p coding "Explain async generators"

# Show all agent profiles and frameworks
uv run cli agents langchain --list
```

**Python:**
```python
from genai_tk.agents.langchain import LangchainAgent

agent = LangchainAgent("Research")
result = agent.run("What is the latest AI news?")
print(result)
```

---

### ⚙️ Workflows

```bash
# List available workflow profiles
uv run cli workflow list profiles

# Show the execution plan (dry-run)
uv run cli workflow run markdownize_docs --dry-run

# Execute a workflow
uv run cli workflow run markdownize_docs
```

**YAML:**
```yaml
# config/workflows.yaml
workflows:
  my_pipeline:
    steps:
      - id: extract_pdf
        uses: genai_tk.workflow.prefect.flows.pdf_to_markdown_flow
        inputs:
          input_dir: "${paths.pdfs}"
          output_dir: "${paths.markdown}"
```

Then: `uv run cli workflow run my_pipeline`

---

## More Examples

### CLI Reference

After completing installation and running `uv run cli init`, explore the full CLI:

```bash
# Discover all available commands
uv run cli --help

# Inspect a specific model profile
uv run cli info llm-profile gpt-4o-mini

# Test with a fake model (no API key needed)
uv run cli core llm -i "tell me a joke" -m parrot_local@fake

# Try the generated example commands (after uv sync)
uv run cli example joke "software engineers"   # simple LLM call
uv run cli example agent "What is 2 + 2?"     # ReAct agent with tools
```

See [docs/cli.md](docs/cli.md) for the full command reference.

---

## First steps — Web UI

Test agents interactively without code:

```bash
just webapp          # launches Streamlit on http://localhost:8501
```

Three built-in demo pages are included in an **Agents** section:

- **🦌 DeerFlow Agent** — 2-panel trace + chat, artifacts, streaming
- **🤖 ReAct Agent** — tool-call trace, MCP servers, slash commands
- **🤖 SmolAgents** — SmolAgents step display

**Downstream projects** (like genai-blueprint) can embed these pages in their
own navigation using the `genai_tk://` prefix — no copy-pasting, no wrappers:

```yaml
# config/app_conf.yaml in your project
ui:
  pages_dir: myapp/webapp/pages
  navigation:
    agents:
      - genai_tk://demos/deer_flow_agent.py   # served from the installed package
      - genai_tk://demos/reAct_agent.py
    demos:
      - demos/my_custom_page.py               # your own page
```

See [docs/webapp.md](docs/webapp.md) for configuration, cross-package navigation,
custom pages, and running from a new project via `cli init --name "My Project"`.

---

## Monitoring and Observability

Track LLM calls, agent steps, and pipeline execution across **LangSmith**, **LangFuse**, **OpenTelemetry**, and **local JSONL logs**.

```bash
# Check monitoring status
uv run cli monitoring status

# View local trace log (most recent first)
uv run cli monitoring tail                    # last 20 entries
uv run cli monitoring tail --n 50 --json      # raw JSON for piping

# Start self-hosted LangFuse
uv run cli monitoring start langfuse
uv run cli monitoring open langfuse
```

**Configuration** — YAML aliases make it easy to switch between cloud and self-hosted:

```yaml
monitoring:
  _langfuse_cloud: &langfuse_cloud
    host: https://cloud.langfuse.com
    public_key: ${oc.env:LANGFUSE_PUBLIC_KEY,""}
    secret_key: ${oc.env:LANGFUSE_SECRET_KEY,""}

  backends: [langfuse, local]      # Multiple backends active in parallel
  project: MyProject
  langfuse: *langfuse_cloud        # Change to *langfuse_local for docker-compose
  local_log:
    path: ${paths.data_root}/traces/llm_calls.jsonl
    include_prompts: true
```

Set API keys in `~/.env`:
```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGSMITH_API_KEY=...
```

See [docs/monitoring.md](docs/monitoring.md) for configuration, self-hosted setup, and troubleshooting.

---

## Python API Examples


See **[Quick Start by Domain](#quick-start-by-domain)** above for domain-specific examples with explanations.

For complete Python examples:
- **Core GenAI**: [docs/core.md](docs/core.md)
- **Agents**: [docs/agents.md](docs/agents.md) + [AGENTS.md](AGENTS.md)
- **Workflows**: [docs/workflows.md](docs/workflows.md) + [docs/prefect.md](docs/prefect.md)

---

## Agents Deep Dive

The toolkit ships four agent frameworks, all sharing the same YAML profile system, LLM factory, MCP servers, and Docker sandbox.

### Choosing an agent

| Agent | Best for | Multi-turn | Planning | Web research | Code execution |
|-------|----------|:----------:|:--------:|:------------:|:--------------:|
| **ReAct** | General tasks, tool use | ✓ | — | via tools | via tools |
| **Deep agent** | Complex multi-step planning | ✓ | ✓ | via tools | Docker sandbox |
| **Deer-flow** | Deep web research & reports | ✓ | ✓ | ✓ native | Docker sandbox |
| **SmolAgents** | Code-first automation | ✓ | — | via tools | local / E2B |

---

### ReAct agent (LangChain)

Standard Thought → Action → Observation loop. Good general-purpose default.

```bash
cli agents langchain --chat                           # interactive session
cli agents langchain -p coding "Review this code"    # profile by KEY
cli agents langchain --list                           # show profiles
```

```python
from genai_tk.agents.langchain import LangchainAgent

agent = LangchainAgent("research")
result = agent.run("Summarise the 2024 GPT-4 technical report")
```

Profile in `config/agents/langchain.yaml`:

```yaml
langchain_agents:
  profiles:
    - name: Research
      type: react
      llm: gpt_41mini@openai
      tools:
        - spec: web_search
      mcp_servers: []
      checkpointer:
        type: memory       # memory | postgres | sqlite
```

---

### Deep agent (LangChain)

Extends ReAct with multi-step planning, subagent delegation, and optional Docker sandbox execution. Requires the `deepagents` extra package.

```bash
cli agents langchain -p research --chat
```

```yaml
langchain_agents:
  research:                             # Profile KEY
    name: "Research"                    # Display name
    type: deep
    llm: gpt_41@openai
    enable_planning: true
    skills:
      directories:
        - ${paths.project}/skills       # SKILL.md files read on demand
    backend:
      type: aio_sandbox                 # optional Docker sandbox
```

---

### Deer-flow (ByteDance / LangGraph)

[Deer-flow](https://github.com/bytedance/deer-flow) is a multi-agent LangGraph system with native web search, planning, reporting, and sub-agents. genai-tk embeds it **in-process** — no separate server.

**Setup** (one-time):

```bash
cli init --deer-flow          # clones Deer-flow and installs its backend deps
# then add to .env: DEER_FLOW_PATH=~/deer-flow
```

**Run:**

```bash
cli agents deerflow --chat                              # interactive (default profile)
cli agents deerflow -p research_assistant --trace "Explain quantum key distribution"
cli agents deerflow -p research_assistant --mode ultra --chat   # full planning + sub-agents
cli agents deerflow --list                              # show profiles + modes
```

**Modes:**

| Mode | Thinking | Planning | Sub-agents |
|------|:--------:|:--------:|:----------:|
| `flash` | — | — | — |
| `thinking` | ✓ | — | — |
| `pro` | ✓ | ✓ | — |
| `ultra` | ✓ | ✓ | ✓ |

**Profile** in `config/agents/deerflow.yaml`:

```yaml
deerflow_agents:
  - name: Research Assistant
    mode: pro
    llm: gpt_41@openai          # optional; falls back to server default
    mcp_servers: [tavily-mcp]
    skill_directories:
      - ${paths.project}/skills
    available_skills:           # filter which skills are exposed (omit = all)
      - public/deep-research
      - public/data-analysis
    sandbox: local              # local | docker
```

Skills in `$DEER_FLOW_PATH/skills` are discovered automatically when running outside the development tree.

---

### SmolAgents (HuggingFace)

Code-first agent that writes and executes Python. Supports local, E2B, and Docker execution backends.

```bash
cli agents smolagents "Plot the sine function and save to sine.png"
cli agents smolagents --executor docker "Install pandas and analyse data.csv"
cli agents smolagents --executor e2b "Scrape and summarise this webpage: …"    # requires E2B_API_KEY
```

---

### Skills — domain knowledge on demand

Skills are `SKILL.md` files that agents load **when needed** — not injected on every call. This keeps context lean and enables per-task specialisation.

```
skills/
├── custom/              # your project skills (committed)
│   └── my-domain/
│       └── SKILL.md
├── community/           # installed via cli skills add (gitignored)
└── bundled/             # copies of genai-tk bundled skills
```

Every `cli init` project gets a `skills/` tree, a `docs/SKILLS.md` guide, and a
`getting-started` example skill. Use the `cli skills` command group to manage them:

```bash
cli skills list                                           # all discovered skills
cli skills add getting-started                            # bundled skill
cli skills add --skillssh langchain-ai/langchain-skills   # from GitHub (skills.sh format)
cli skills add --git https://github.com/org/repo --path my-skill
cli skills create my-domain-skill                        # interactive scaffold
cli skills validate --all                                 # validate frontmatter + structure
cli skills info my-skill                                  # show full SKILL.md
```

Any agent that supports `skill_directories` (LangChain deep, all Deer-flow profiles)
can discover and use skill files. In Docker sandbox mode the skill directories are
automatically bind-mounted read-only at `/mnt/skills/`.

See [docs/scaffolding.md](docs/scaffolding.md) for the complete skills guide and
[skills.sh](https://www.skills.sh) for the community registry.

---

### Sandbox

Agents can run code in an isolated Docker container via [OpenSandbox](https://github.com/alibaba/OpenSandbox):

```bash
# One-time warm-up (cuts container startup from ~28 s to ~5 s)
cli sandbox start
cli sandbox pull

# Use with any agent
cli agents langchain  -p research --sandbox docker "Write and run a Python script"
cli agents deerflow   -p research_assistant --sandbox docker --chat
cli agents smolagents --executor docker "Run some code"
```

The sandbox provides Chromium (VNC at `localhost:8080/vnc`), Python, Node.js, and a REST shell/file API. Skill directories are mount-inserted automatically.

See [docs/sandbox_support.md](docs/sandbox_support.md) for full setup instructions.

---

### MCP Servers

Any agent profile can load tools from [Model Context Protocol](https://modelcontextprotocol.io) servers:

```yaml
# config/mcp_servers.yaml
mcp_servers_config:
  tavily-mcp:
    command: npx
    args: ["-y", "tavily-mcp"]
    env:
      TAVILY_API_KEY: ${TAVILY_API_KEY}
  math_server:
    command: python
    args: ["-m", "genai_tk.mcp.math_server"]
```

```bash
cli agents deerflow -p research_assistant --mcp math_server "…"
cli agents langchain -p research --mcp custom_server "…"
```

See [docs/mcp-servers.md](docs/mcp-servers.md) for the full reference.

---

## Workflows

Orchestrate multi-step AI pipelines using a YAML-based DSL with Prefect execution.

### Workflow DSL (YAML)

Define workflows as compositions of **steps** with dependencies, templates, and sub-workflows — no Python required:

```yaml
# config/workflows.yaml
step_templates:
  markdownize_step:
    uses: genai_tk.workflow.prefect.flows.markdownize_flow
    inputs:
      root_dir: "${params.input_dir}"
      output_dir: "${params.output_dir}"

workflows:
  document_pipeline:
    steps:
      - id: convert
        ref: markdownize_step
        inputs:
          input_dir: "${paths.pdfs}"

workflow_profiles:
  process_docs:
    workflow: document_pipeline
    values:
      input_dir: ~/Documents/pdfs
      output_dir: ~/Documents/markdown
```

**Run:**

```bash
uv run cli workflow list profiles              # show available profiles
uv run cli workflow run process_docs --dry-run # show the plan
uv run cli workflow run process_docs           # execute
```

### Prefect Flows

Ships with ready-to-use flows:

| Flow | Purpose | CLI |
|------|---------|-----|
| **markdownize** | PDF → Markdown + OCR | `cli tools markdownize` |
| **ppt2pdf** | PowerPoint → PDF | `cli tools ppt2pdf` |
| **baml** | Structured extraction | `cli baml run` |
| **rag** | RAG indexing + retrieval | `cli rag add-files` |

All flows run **in-process** with an ephemeral Prefect client — no Prefect server needed.

See [docs/workflows.md](docs/workflows.md) and [docs/prefect.md](docs/prefect.md) for full reference.

---

## LLM Selection

Models are referenced as `model_id@provider` — a short logical name plus the provider that serves it (`openai`, `openrouter`, `groq`, `ollama`, `fake`, …).

The toolkit ships with a built-in database sourced from [models.dev](https://models.dev) covering 1 000+ models across all major providers. You only need `llm.yaml` entries for models that are **not** in that database or when you want to give a model a short alias:

```yaml
# config/providers/llm.yaml
llm:
  exceptions:
    - model_id: gpt41mini          # short alias used in config and CLI
      providers:
        - openai: gpt-4.1-mini-2025-04-14  # maps to the actual API name
    - model_id: haiku
      providers:
        - openrouter: anthropic/claude-haiku-4-5
    - model_id: parrot_local       # built-in fake model, no API key needed
      providers:
        - fake: parrot
```

Override at runtime with `-m` / `--llm`:

```bash
cli core llm -i "Hello" -m gpt41mini@openai   # declared alias — explicit provider
cli core llm -i "Hello" -m fast_model         # named tag from genai_def.yaml
cli core llm -i "Hello" -m gpt-4o-mini        # raw model name — fuzzy-resolved from models.dev
```

See [docs/llm-selection.md](docs/llm-selection.md) for the full reference (tags, `cli info` commands, models.dev database).

---

## Configuration

The config system uses a hierarchy of YAML files loaded from `config/` (auto-discovered by walking up the directory tree):

```
config/
├── app_conf.yaml           # entry point — profile selection, paths, env vars
├── profiles/
│   ├── local/
│   │   └── genai_def.yaml   # default LLM / embeddings / cache
│   └── pytest/
│       └── genai_def.yaml   # fake models for test runs
├── providers/
│   ├── llm.yaml            # LLM model declarations
│   └── embeddings.yaml     # Embeddings model declarations
└── agents/
    ├── langchain.yaml      # LangChain agent profiles (ReAct / Deep / Custom)
    └── deerflow.yaml       # Deer-flow profiles + skills + sandbox config
```

Environment variables in `.env` override config values at any level.  
Switch deployment environment with `GENAITK_PROFILE=prod` or in code with `switch_profile("prod")`.  
Activate a named in-session overlay (no reload) with `global_config().use_context("training_local")`.

See [docs/configuration.md](docs/configuration.md) for the full reference.

---

## Capabilities by Domain

| **Core GenAI** | CLI | Python | Docs |
|---|---|---|---|
| LLM / Embeddings | `cli core llm` | `get_llm()` / `get_embeddings()` | [docs/core.md](docs/core.md) |
| Model selection | `cli info models` | `llm.yaml` | [docs/llm-selection.md](docs/llm-selection.md) |
| Vector stores | — | `EmbeddingsStore` | [docs/core.md](docs/core.md) |
| LLM caching | — | `LlmCache` | [docs/core.md](docs/core.md) |

| **Agents** | CLI | Python | Docs |
|---|---|---|---|
| ReAct agent | `cli agents langchain` | `LangchainAgent` | [docs/agents.md](docs/agents.md) |
| Deep agent | `cli agents langchain -p <deep>` | `LangchainAgent` (deep) | [docs/agents.md](docs/agents.md) |
| Deer-flow | `cli agents deerflow` | `EmbeddedDeerFlowClient` | [docs/deer-flow.md](docs/deer-flow.md) |
| SmolAgents | `cli agents smolagents` | `SmolAgent` | [docs/agents.md](docs/agents.md) |
| Skills | — | `skill_directories:` in config | [AGENTS.md](AGENTS.md#skills) |
| Docker sandbox | `cli sandbox` | `SandboxBackend` | [docs/sandbox_support.md](docs/sandbox_support.md) |
| MCP servers | `cli mcpserver` | `McpClient` | [docs/mcp-servers.md](docs/mcp-servers.md) |

| **Workflows** | CLI | Python | Docs |
|---|---|---|---|
| Workflow DSL | `cli workflow` | `resolve_workflow_invocation()` | [docs/workflows.md](docs/workflows.md) |
| Prefect flows | `cli tools *` | `run_flow_ephemeral()` | [docs/prefect.md](docs/prefect.md) |
| RAG pipeline | `cli rag` | `RetrieverFactory` | [docs/rag.md](docs/rag.md) |
| BAML extraction | `cli baml` | `BamlStructuredProcessor` | [docs/baml.md](docs/baml.md) |
| Document loaders | — | `MarkdownLoader` | [docs/workflows.md](docs/workflows.md) |

| **Cross-cutting** | CLI | Python | Docs |
|---|---|---|---|
| Configuration | `cli init` | `global_config()` | [docs/configuration.md](docs/configuration.md) |
| Project scaffolding | `cli init --name` | `ProjectScaffolder` | [docs/scaffolding.md](docs/scaffolding.md) |
| Copilot Agent support | `cli init` | — | [docs/copilot-agent-support.md](docs/copilot-agent-support.md) |
| CLI extension | — | `CliTopCommand` | [docs/cli.md](docs/cli.md) |
| Streamlit webapp | `just webapp` | `genai_tk.webapp` | [docs/webapp.md](docs/webapp.md) |
| Testing | `cli test` | pytest | [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md) |
| Browser automation | — | `browser_use` tools | [docs/browser_control.md](docs/browser_control.md) |

Design and investigation notes: [`docs/design/`](docs/design/).

---

## Package Structure

```
genai_tk/
├── core/          # LLM factory, embeddings, vector stores, cache, MCP client
│   └── vector_backends/  # Chroma, InMemory, PgVector (+ Postgres connection mgmt)
├── agents/
│   ├── langchain/ # Unified ReAct / Deep / Custom agents + middleware
│   ├── deer_flow/ # Deer-flow embedded client + CLI
│   └── smolagents/
├── workflow/      # ETL orchestration: Prefect flows, RAG, loaders, retrievers
│   ├── prefect/   # run helpers + flows/ (markdownize, ppt2pdf, rag, baml)
│   ├── rag/       # chunkers, RAG CLI commands
│   ├── loaders/   # Markdown loader, Mistral OCR loader
│   └── retrievers/ # BM25, ZeroEntropy
├── extra/         # Non-pipeline tooling: agent graphs, anonymization, BAML, image analysis
├── tools/         # LangChain and SmolAgents tool sets
├── utils/         # Config manager, Pydantic helpers, LangGraph utilities
└── main/          # CLI entry point + command modules
```

---

## Development

```bash
just fmt    # ruff format + isort
just lint   # ruff lint
just test   # unit + integration (no API keys needed)
just check  # fmt + lint + test

# Run tests that need real models
uv run cli test full_integration
```

See [AGENTS.md](AGENTS.md) for code style, Pydantic conventions, and testing patterns.

---

## License

MIT — see [LICENSE](LICENSE).
