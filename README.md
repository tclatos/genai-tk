# GenAI Toolkit (`genai-tk`)

A toolkit for building Gen AI and Agentic applications with LangChain, LangGraph, and 100+ LLM providers.

**What it gives you:**
- Multi-provider LLM and embeddings factory (OpenAI, Groq, Anthropic, Ollama, local, …)
- YAML-configured agents: ReAct, Deep (planning + subagents), custom LangGraph graphs
- Full RAG pipeline — chunking, BM25 + semantic hybrid retrieval, PGVector
- Structured extraction with BAML (type-safe Pydantic output, incremental manifests)
- Prefect document flows, browser automation, Docker sandbox, MCP servers
- Rich CLI that mirrors every capability; easily extensible with one class + one YAML line

---

## Installation

**Start from scratch:**

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new project directory
mkdir my-genai-project && cd my-genai-project

# Initialize a Python project
uv init

# Add genai-tk to your project
uv add git+https://github.com/tclatos/genai-tk@main

# Initialize configuration and sample agents
uv run cli init              # copies default config/ tree
uv run cli init --deer-flow  # (optional) also clones the Deer-flow backend
```

**Add to existing project:**

```bash
# Add to your project  
uv add git+https://github.com/tclatos/genai-tk@main

# With PostgreSQL / Playwright extras
uv add "genai-tk[extra] @ git+https://github.com/tclatos/genai-tk@main"

# Initialize config in the current directory
uv run cli init              # copies default config/ tree
uv run cli init --deer-flow  # also clones the Deer-flow backend
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

## First steps — CLI

After completing installation and running `uv run cli init`, you can explore the CLI:

```bash
# Discover all available commands
uv run cli --help

# List all known LLM and embeddings models
uv run cli info models

# Inspect a specific model profile — supports fuzzy matching on the model name
uv run cli info llm-profile gpt41mini@openai
uv run cli info llm-profile gpt-4o          # fuzzy match: finds the closest declared model
```

```bash
# Confirm everything works (no API key needed) 
uv run cli core llm -i "tell me a joke" -m parrot_local@fake  # should print "tell me a joke"

# With a real model
uv run cli core llm -i "tell me a joke" -m gpt-4o-mini@openai --stream

# Interactive ReAct chat session (uses the default profile)
uv run cli agents langchain --chat

# Run a single query against a named profile
uv run cli agents langchain -p Coding "Explain async generators in Python"

# Show loaded config, API key status, active model
uv run cli info config
```

See [docs/cli.md](docs/cli.md) for the full command reference.

---

## First steps — Python

```python
# config is auto-discovered from the current directory upwards
from genai_tk.core.llm_factory import get_llm
from genai_tk.core.embeddings_factory import get_embeddings

llm = get_llm()                          # uses default model from config
llm = get_llm("gpt41mini@openai")        # explicit model
response = llm.invoke("Tell me a joke")
print(response.content)
```

```python
# ReAct agent — single query
from genai_tk.agents.langchain import LangchainAgent

agent = LangchainAgent("Research")
result = agent.run("What is the latest news on LLM benchmarks?")
print(result)
```

```python
# Async streaming
import asyncio
from genai_tk.agents.langchain import LangchainAgent

async def main():
    agent = LangchainAgent("Research")
    async for chunk in agent.astream("Explain RAG in one paragraph"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

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
cli core llm -i "Hello" -m fast_model         # named tag from baseline.yaml
cli core llm -i "Hello" -m gpt-4o-mini        # raw model name — fuzzy-resolved from models.dev
```

See [docs/llm-selection.md](docs/llm-selection.md) for the full reference (tags, `cli info` commands, models.dev database).

---

## Configuration

The config system uses a hierarchy of YAML files loaded from `config/` (auto-discovered by walking up the directory tree):

```
config/
├── app_conf.yaml           # entry point — sets default_config and :merge list
└── basic/
    ├── init/baseline.yaml  # defaults: default LLM / embeddings / cache
    ├── providers/
    │   ├── llm.yaml        # LLM model declarations
    │   └── embeddings.yaml # Embeddings model declarations
    └── agents/
        ├── langchain.yaml  # LangChain agent profiles
        └── deerflow.yaml   # Deer-flow profiles
```

Environment variables in `.env` override config values at any level.  
Switch named environments with `BLUEPRINT_CONFIG=production` or `config.select_config("production")`.

See [docs/configuration.md](docs/configuration.md) for the full reference.

---

## Capabilities

| Area | CLI entry point | Python entry point | Reference |
|------|-----------------|--------------------|-----------|
| **LLM / Embeddings** | `cli core llm` | `get_llm()` / `get_embeddings()` | [docs/core.md](docs/core.md) |
| **LLM model config** | `cli info models` | `llm.yaml` | [docs/llm-selection.md](docs/llm-selection.md) |
| **LangChain Agents** | `cli agents langchain` | `LangchainAgent` | [docs/agents.md](docs/agents.md) |
| **Deer-flow** | `cli agents deerflow` | `EmbeddedDeerFlowClient` | [docs/deer-flow.md](docs/deer-flow.md) |
| **RAG** | `cli rag ingest/query` | `extra.rag` | [docs/extra.md](docs/extra.md#rag-systems) |
| **BAML structured output** | `cli baml run/extract` | `BamlStructuredProcessor` | [docs/baml.md](docs/baml.md) |
| **Prefect flows** | `cli tools markdownize` (example) | `run_flow_ephemeral()` | [docs/prefect.md](docs/prefect.md) |
| **MCP servers** | `cli mcpserver` | `McpClient` | [docs/mcp-servers.md](docs/mcp-servers.md) |
| **Docker sandbox** | `cli sandbox` | `SandboxBackend` | [docs/sandbox_support.md](docs/sandbox_support.md) |
| **Browser automation** | — | `browser_use` tools | [docs/browser_control.md](docs/browser_control.md) |
| **Testing** | `cli test unit` / `cli test fast_integration` | pytest | [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md) |
| **Configuration** | `cli init` | `global_config()` | [docs/configuration.md](docs/configuration.md) |
| **CLI extension** | — | `CliTopCommand` | [docs/cli.md](docs/cli.md) |

Design and investigation notes: [`docs/design/`](docs/design/).

---

## Package Structure

```
genai_tk/
├── core/          # LLM factory, embeddings, vector stores, cache, MCP client
├── agents/
│   ├── langchain/ # Unified ReAct / Deep / Custom agents + middleware
│   ├── deer_flow/ # Deer-flow embedded client + CLI
│   └── smolagents/
├── extra/         # RAG, agent graphs, loaders, retrievers, anonymization, BAML
├── tools/         # LangChain and SmolAgents tool sets
├── utils/         # Config manager, Pydantic helpers, LangGraph utilities
└── main/          # CLI entry point + command modules
```

---

## Development

```bash
make fmt    # ruff format + isort
make lint   # ruff lint
make test   # unit + integration (no API keys needed)
make check  # fmt + lint + test

# Run tests that need real models
uv run cli test full_integration
```

See [AGENTS.md](AGENTS.md) for code style, Pydantic conventions, and testing patterns.

---

## License

MIT — see [LICENSE](LICENSE).
