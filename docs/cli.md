# CLI Reference

The `cli` entry-point is the primary interface for interacting with GenAI Toolkit from the
terminal.  All commands are implemented with [Typer](https://typer.tiangolo.com/) and organised
into **command groups** that are discovered dynamically from the application configuration.

## Quick Start

`uv run cli` invokes the `cli` script entry point defined in `pyproject.toml` (`[project.scripts]`):

```toml
cli = "genai_tk.main.cli.main"
```

**Optional:** Create a shell alias to avoid typing `uv run` each time:

```bash
alias cli='uv run cli'
```

Then use `cli` directly instead of `uv run cli`:

```bash
# Show all command groups (no arguments)
cli

# Show commands within a group
cli core --help
```

Or with the full `uv run` prefix:

```bash
# Show all command groups (no arguments)
uv run cli

# Show commands within a group
uv run cli core --help
```

---

## Command Groups

### `core` — AI Model Interactions

Direct access to LLMs and registered chains.

| Sub-command | Description |
|-------------|-------------|
| `core llm`  | Invoke an LLM with a prompt |
| `core run`  | Run a named `ChainRegistry` runnable |

#### `core llm`

```bash
# Basic prompt
uv run cli core llm "Tell me a joke"

# Pipe from stdin
echo "Summarise this text" | uv run cli core llm --input -

# Select model by tag or ID
uv run cli core llm "Explain AI" --llm powerful_model
uv run cli core llm "Explain AI" --llm gpt_4o@openai

# Streaming output
uv run cli core llm "Write a poem" --stream

# Adjust temperature
uv run cli core llm "Be creative" --temperature 0.8

# Enable reasoning/thinking mode (o3, claude thinking, etc.)
uv run cli core llm "Solve this maths problem" --reasoning

# Inspect raw LangChain response object
uv run cli core llm "Hello" --raw

# Enable LangChain debug/verbose tracing
uv run cli core llm "Hello" --debug
uv run cli core llm "Hello" --verbose
```

**Key options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--input TEXT` | `-i` | stdin | Input text, or `-` to read from stdin |
| `--llm TEXT` | `-m` | default | LLM tag or ID from config |
| `--temperature` | | 0.0 | Sampling temperature (0–1) |
| `--stream` | `-s` | false | Stream tokens progressively |
| `--reasoning` | | false | Enable thinking mode |
| `--cache` | | memory | Cache strategy (`memory` / `sqlite`) |
| `--raw` | | false | Print raw LLM response object |

#### `core run`

```bash
# Run a registered chain by name
uv run cli core run my_chain "Input for the chain"
```

---

### `info` — System Information

```bash
# Show config, default components, LLM tags, API key status
uv run cli info config

# Inspect a specific LLM profile
uv run cli info llm-profile gpt_4o@openai
```

`info config` prints:
- Active configuration name
- Default LLM / embeddings / vector-store
- All configured LLM tags with their model IDs and providers
- API key availability per provider
- Configured KV stores

---

### `agents` — Agent Runners

```bash
# LangChain ReAct / deep agent
uv run cli agents langchain --list                          # list all profiles
uv run cli agents langchain "Research quantum computing"    # default profile
uv run cli agents langchain -p research "Deep dive topic"   # specific profile
uv run cli agents langchain --chat                          # interactive chat

# SmolAgents CodeAct
uv run cli agents smol "Write and execute a Python sort"
uv run cli agents smol --chat                               # interactive chat
uv run cli agents smol --tools web_search,python_interpreter "..."

# DeerFlow
uv run cli agents deerflow --list         # list profiles / modes
uv run cli agents deerflow --chat         # interactive chat
uv run cli agents deerflow "Research AI"  # one-shot

# DeepAgents CLI
uv run cli agents deepagent task "Analyse this codebase"
uv run cli agents deepagent list          # list threads
uv run cli agents deepagent skills        # list available skills
```

See [agents.md](agents.md) for full agent configuration reference.

---

### `baml` — Structured Output (BAML)

```bash
# Run a BAML function on text input
uv run cli baml run ExtractResume -i "John Smith; SW engineer"

# Read from stdin
cat resume.txt | uv run cli baml run ExtractResume

# Save result to a JSON file
uv run cli baml run FakeResume -i "Jane Doe, architect" \
    --out-dir ./output --out-file jane_doe.json

# Batch extract markdown files
uv run cli baml extract ./docs ./output \
    --recursive --function ExtractRainbow

# Batch extraction with file filters and force re-run
uv run cli baml extract ./reports ./output \
    --include 'report_*.md' --exclude '*_draft.md' \
    --recursive --force
```

See [baml.md](baml.md) for the full BAML integration guide.

---

### `rag` — Retrieval-Augmented Generation

```bash
# Ingest files into the default vector store
uv run cli rag ingest ./documents

# Semantic search
uv run cli rag query "What are the risk factors?"

# Vector store statistics
uv run cli rag info

# Manage named indexes
uv run cli rag create-index ./data
uv run cli rag add-documents ./more_data
uv run cli rag list-indexes
```

---

### `tools` — Utility Tools

```bash
# Convert PDFs / DOCX / PPTX → Markdown (uses Prefect)
uv run cli tools markdownize ./input_dir ./output_dir --recursive

# Use Mistral OCR for higher-quality PDF extraction
uv run cli tools markdownize ./pdfs ./output --mistral-ocr

# Force re-conversion of already-processed files
uv run cli tools markdownize ./input_dir ./output_dir --force

# Convert PowerPoint files to PDF
uv run cli tools ppt2pdf ./slides ./pdfs --recursive

# GPT Researcher
uv run cli tools gpt-researcher "Latest AI trends 2025"
```

---

### `sandbox` — Code Execution Sandbox

```bash
uv run cli sandbox start   # start the Docker sandbox container
uv run cli sandbox stop    # stop the container
uv run cli sandbox status  # inspect running status
uv run cli sandbox pull    # pull the latest sandbox image
```

See [sandbox_support.md](sandbox_support.md) for setup and integration details.

---

### `mcpserver` — MCP Server Management

```bash
# List all configured MCP servers
uv run cli mcpserver list

# Start a server (stdio / sse / streamable-http transport)
uv run cli mcpserver start --name math_server
uv run cli mcpserver start --name weather_server --transport sse

# Generate a standalone server script
uv run cli mcpserver generate --name math_server
```

See [mcp-servers.md](mcp-servers.md) for configuration and usage.

---

## Global Flags

These flags apply to every command:

| Flag | Description |
|------|-------------|
| `--logging LEVEL` | Set log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Processed before Typer. |
| `--help` | Show help for any command or group. |

---

## Configuration Variables in Arguments

Many options that accept paths support **OmegaConf interpolation**.  Wrap the value in
single quotes to prevent shell expansion:

```bash
uv run cli baml extract '${paths.data_root}/docs' '${paths.data_root}/output' \
    --recursive --function ExtractRainbow
```

The following variables are always available:

| Variable | Value |
|----------|-------|
| `${paths.project}` | Current working directory at startup |
| `${paths.config}` | `<project>/config/basic` |
| `${paths.data_root}` | `<project>/data` |
| `${paths.home}` | `$HOME` |

---

## Extending the CLI — Adding New Commands

All command groups inherit from `CliTopCommand` and are auto-loaded from
`cli.commands` in `config/app_conf.yaml`.

### Step 1 — Create the command class

```python
# src/myapp/commands_hello.py
from typing import Annotated

import typer
from genai_tk.cli.base import CliTopCommand


class HelloCommands(CliTopCommand):
    """Greeting commands."""

    description: str = "Friendly greeting commands."

    def get_description(self) -> tuple[str, str]:
        # Returns (CLI group name, help text)
        return "hello", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command()
        def greet(
            name: Annotated[str, typer.Argument(help="Name to greet")],
            shout: Annotated[bool, typer.Option("--shout", help="Use uppercase")] = False,
        ) -> None:
            """Greet someone by name."""
            msg = f"Hello, {name}!"
            print(msg.upper() if shout else msg)

        @cli_app.command()
        def farewell(name: Annotated[str, typer.Argument(help="Name")]) -> None:
            """Say goodbye."""
            print(f"Goodbye, {name}!")
```

### Step 2 — Register in config

```yaml
# config/app_conf.yaml  (or overrides.yaml in your project)
cli:
  commands:
    # … existing entries …
    - src.myapp.commands_hello.HelloCommands
```

### Step 3 — Use it

```bash
uv run cli hello greet Alice
uv run cli hello greet Alice --shout
uv run cli hello farewell Bob
uv run cli hello --help
```

### `CliTopCommand` contract

| Method | Required | Description |
|--------|----------|-------------|
| `get_description()` | Yes | Returns `(group_name, help_text)`. `group_name` becomes the top-level CLI word. |
| `register_sub_commands(app)` | Yes | Define all `@app.command()` functions inside this method. |
| `register(app)` | No (inherited) | Called automatically — creates the sub-Typer and attaches it. Do not override. |

> **Note:** The class is a Pydantic `BaseModel` — you can add configurable fields with defaults
> and they will be available to all sub-commands via `self`.

### Legacy pattern (function-based)

Older modules use a plain function instead of a class:

```python
def register_commands(cli_app: typer.Typer) -> None:
    @cli_app.command()
    def echo(text: str) -> None:
        print(text)
```

```yaml
cli:
  commands:
    - genai_tk.main.cli.register_commands   # function reference (not a class)
```

This pattern is still supported but **not recommended** for new commands — use
`CliTopCommand` instead as it is more structured and testable.

---

## How Command Discovery Works

At startup (`uv run cli`), `load_and_register_commands()` in `genai_tk/main/cli.py`:

1. Reads `cli.commands` from the merged YAML config.
2. For each entry (`module.ClassName`), dynamically imports the symbol.
3. If the symbol is a `CliTopCommand` subclass → instantiates it, calls `.register(cli_app)`.
4. If the symbol is a callable → calls it directly as `fn(cli_app)` (legacy path).
5. If no arguments are provided to `cli`, displays a Rich command tree and exits.

The command tree can always be viewed by running `cli` with no arguments.
