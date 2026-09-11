# GenAI Toolkit Skill Map

This folder organizes skills for agents into a **4-tier architecture**:

- `skills/runtime/` — **Runtime skills**: solve user problems directly during agent runs (query writing, browser automation, code execution, slide generation).
- `skills/development/` — **Development skills**: improve and extend the toolkit (configuring LLMs, creating agent profiles, authoring tools, benchmark framework, workflows).
- `skills/governance/` — **Governance skills**: maintain consistency, quality, testing, and security (evaluation testing, repo mapping, PII anonymization, code review).
- `skills/vendor/` — **Vendor skills**: imported third-party capabilities, not edited directly (e.g. `atos-slidev`).
- `skills/custom/` — **Custom skills**: project-specific skills created by users.
- `skills/community/` — **Community skills**: installed via `skills.sh` or git.

Use `skills/governance/repo-map` first when you need orientation.

---

## 1. Runtime Skills (`skills/runtime/`)

| Skill | Description | Primary Code/Path |
|---|---|---|
| `browser-automation` | Generic browser automation patterns for navigating websites and forms | `genai_tk/agents/tools/*browser/` |
| `codeact` | Safe sandbox code execution for multi-step reasoning | `genai_tk/agents/tools/python_executor/` |
| `ppt-generation` | AI PowerPoint slide generation and layout | `genai_tk/extra/` |
| `query-writing` | SQL and structured query authoring recipes | `genai_tk/extra/` |
| `schema-exploration` | Database and schema discovery patterns | `genai_tk/extra/` |

---

## 2. Development Skills (`skills/development/`)

| Skill | Description | Closest Docs | Primary Code/Config |
|---|---|---|---|
| `benchmark-framework` | Multi-dataset benchmark suite (`cli bench`) & custom adapters | `docs/benchmark_framework.md` | `genai_graph/bench/`, `genai_graph/core/commands_bench.py` |
| `configuration` | OmegaConf YAML config access & profiles | `docs/configuration.md` | `genai_tk/config_mgmt/`, `config/` |
| `core-models` | LLM and Embeddings factory selection | `docs/core.md`, `docs/llm-selection.md` | `genai_tk/core/`, `config/providers/` |
| `agent-profiles` | LangChain / DeerFlow / DeepAgent profiles | `docs/agents.md`, `docs/deer-flow.md` | `genai_tk/agents/`, `config/agents/` |
| `add-tool` | Authoring LangChain & DeepAgent tools | — | `genai_tk/agents/tools/` |
| `add-skill` | Authoring and discovering SKILL.md files | `docs/SKILLS.md` | `skills/` |
| `add-mcp-server` | MCP server configuration & connection | `docs/mcp-servers.md` | `config/mcp_servers.yaml` |
| `add-agent-profile` | Step-by-step agent profile scaffolding | `docs/agents.md` | `config/agents/` |
| `add-cli-command` | Adding Typer CLI command groups | `docs/cli.md` | `genai_tk/cli/` |
| `add-chain` | Registering LCEL runnables | `docs/core.md` | `genai_tk/chains/` |
| `add-webapp-page` | Adding Streamlit navigation pages | `docs/webapp.md` | `genai_tk/webapp/pages/` |
| `rag-systems` | RAG retrievers and indexing pipelines | `docs/rag.md` | `genai_tk/core/retrievers/`, `config/rag.yaml` |
| `workflow-engine` | YAML-driven task orchestration & Prefect | `docs/workflows.md`, `docs/prefect.md` | `genai_tk/workflow/`, `config/workflows.yaml` |
| `browser-and-sandbox` | Playwright browser & OpenSandbox Docker | `docs/browser_control.md`, `docs/sandbox_support.md` | `genai_tk/agents/sandbox/` |
| `mcp-servers` | Built-in and standalone MCP servers | `docs/mcp-servers.md` | `genai_tk/mcp/` |
| `cli-and-scaffolding` | CLI commands & `cli init` scaffolding engine | `docs/cli.md`, `docs/scaffolding.md` | `genai_tk/cli/`, `genai_tk/main/` |
| `cli-chat-interfaces` | Interactive REPL & chat CLI subcommands | `docs/cli.md` | `genai_tk/cli/` |
| `webapp` | Streamlit multi-page webapp configuration | `docs/webapp.md` | `genai_tk/webapp/`, `config/webapp.yaml` |
| `streamlit-workflow-runner` | Visual workflow execution in Streamlit | `docs/webapp.md` | `genai_tk/webapp/components/` |
| `baml-structured-extraction` | BAML structured schema extraction | `docs/baml.md` | `genai_tk/extra/structured/` |
| `nlp` | spaCy, model manager & text classification | `docs/nlp.md` | `genai_tk/extra/nlp/` |
| `docker` | Docker container build & deployment | `docs/docker.md` | `deploy/Dockerfile` |
| `optional-features` | Managing `[project.optional-dependencies]` | — | `genai_tk/config_mgmt/features.py` |
| `python-interpreter` | Python interpreter & venv resolution | — | `genai_tk/agents/tools/python_executor/` |

---

## 3. Governance Skills (`skills/governance/`)

| Skill | Description | Closest Docs | Primary Code/Path |
|---|---|---|---|
| `repo-map` | Full codebase architecture & file locator | `docs/*.md` | `genai_tk/`, `config/`, `tests/` |
| `evaluation-testing` | Pytest conventions, fake models & evals | `docs/TESTING_GUIDE.md` | `tests/` |
| `pii-anonymization` | Presidio PII detection & anonymization | `docs/middleware-pii-and-routing.md` | `genai_tk/extra/nlp/` |
| `code-review-excellence` | Code quality & best practices checklist | — | — |

---

## 4. Vendor Skills (`skills/vendor/`)

| Skill | Description | Primary Origin |
|---|---|---|
| `atos-slidev` | Executive Slidev presentation decks | External Atos branding package |

---

## CLI Management

```bash
uv run cli skills list                              # List all skills by tier
uv run cli skills list --category runtime           # Filter by category
uv run cli skills list --category development
uv run cli skills list --category governance
uv run cli skills list --category vendor
uv run cli skills validate --all                    # Validate frontmatter and syntax
uv run cli skills create my-skill                   # Scaffold a new skill
uv run cli skills add <bundled-skill>               # Install a bundled skill
uv run cli skills add --git <url> --path <subpath>  # Install from git repository
uv run cli skills add --skillssh <owner/repo>       # Install from skills.sh registry
```
