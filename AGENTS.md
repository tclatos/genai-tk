# Development Guidelines for GenAI Toolkit

## Build Commands

```bash
make install-dev   # install with development dependencies
make fmt           # format with ruff (includes import sorting)
make lint          # lint with ruff
make test          # run all tests (unit + integration)
make test-unit     # unit tests only
make test-integration  # integration tests only
make check         # fmt + lint + test
```

Always use `uv` to run Python code, execute tests, and manage packages.

## Code Style

**Formatting & linting:** ruff, line length 120, isort rules.

**Python version:** 3.12+ required.
- Use `str | None` instead of `Optional[str]`
- Use `list[str]` instead of `List[str]`
- Avoid `Any` unless unavoidable

**Imports:** always absolute — never relative.

```python
# DO
from genai_tk.core import LLMFactory

# DON'T
from ..core import LLMFactory
```

**Naming:**
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_underscore_prefix`

## Data Modeling

Use **Pydantic v2** for all structured data — DTOs, configs, API models, results.

```python
from pydantic import BaseModel, Field

class Result(BaseModel):
    status: str
    count: int
    message: str | None = None
    tags: list[str] = Field(default_factory=list)
```

- Use `model_config = {...}` instead of inner `class Config`
- Use `model_post_init()` instead of `__init__`
- Use `model_dump_json()` instead of `json.dumps(model.model_dump())`
- Avoid `dataclass`, unnamed tuples, and untyped dicts for structured data

## Error Handling & Logging

- Use `loguru` for structured logging
- Raise specific exceptions with descriptive messages
- Use Pydantic validators for input validation at system boundaries
- No defensive checks for internal invariants — trust the type system

## Docstrings

Google style. Do not mention types (they're in the signature). Do not list raised
exceptions. Use fenced code blocks for examples. One line to describe the class —
do not add docstrings to fields or constructor arguments.

```python
def get_llm(llm: str | None = None) -> BaseChatModel:
    """Return a configured LLM instance.

    Args:
        llm: Model ID in `name@provider` format. Uses config default when omitted.

    Returns:
        Configured LangChain chat model.

    Example:
        ```python
        model = get_llm("gpt_41mini@openai")
        response = model.invoke("Hello")
        ```
    """
```

## Configuration

- YAML-based with OmegaConf, env var substitution supported (`${MY_VAR}`)
- Singleton pattern for global config access (`global_config()`)
- Config auto-discovered by searching parent directories — works from any CWD

## Testing

- pytest with asyncio support (`pytest-asyncio`)
- Unit tests in `tests/unit_tests/`, integration tests in `tests/integration_tests/`
- Test files: `test_*.py` or `*_test.py`
- Use `faker` for test data generation
- No Docker required for unit tests; integration tests may require live services

## Agent and Tool Guidelines

### Agents
- Use `cli agents langchain --list` to inspect configured profiles
- Agent type `react` is the default; use `deep` for multi-step planning + skills
- Keep system prompts in YAML (`system_prompt:` field), not hardcoded in Python
- Use `SkillsMiddleware` + SKILL.md files for domain knowledge — avoids bloating
  system prompts and enables progressive disclosure

### Tools
- Define tools as LangChain `BaseTool` subclasses or `@tool`-decorated functions
- Register via factory function (`create_*_tools()`) referenced in agent YAML
- Use `browser_fill_credential` (never plain `browser_type`) for credentials
- Tools that call external APIs must validate inputs at the boundary

### Sandbox
- `--sandbox docker` for isolated execution; `--sandbox local` for development
- Run `cli sandbox start` + `cli sandbox pull` once per boot to pre-warm
- Skill directories are automatically bind-mounted into the container (read-only)
- See `docs/sandbox_support.md` for setup and `docs/browser_control.md` for the browser tooling

### Skills (SKILL.md)
- One SKILL.md per site or domain — kept in `skills/custom/<name>/SKILL.md`
- Skills are read by the agent on demand, not injected into every prompt
- Keep Enedis/site-specific waits and selectors in the skill file, not in Python

## Documentation

Keep docs in `docs/`. Organisation:
- **User-facing features** — one file per major feature (`browser_control.md`, `sandbox_support.md`, etc.)
- **Design notes, investigations, patches** — go in `docs/design/` (internal reference)
- Do not add comments for obvious code; document *why*, not *what*
- Do not duplicate code examples between module docstring and function docstring

Current docs index:

| File | Topic |
|------|-------|
| `docs/cli.md` | All CLI command groups, sub-commands, and how to add new commands |
| `docs/scaffolding.md` | `cli init` — full project scaffold, generated files, config auto-patching |
| `docs/copilot-agent-support.md` | Copilot Agent skills, AGENTS.md, copilot-instructions.md |
| `docs/core.md` | `LlmFactory`, `EmbeddingsFactory`, `EmbeddingsStore`, `LlmCache`, `ChainRegistry` |
| `docs/extra.md` | Agent graphs, RAG, data loaders, anonymization, image analysis, KV store |
| `docs/baml.md` | BAML structured extraction — setup, CLI, programmatic API |
| `docs/prefect.md` | Prefect flows — ephemeral vs. server mode, available flows, writing new flows |
| `docs/agents.md` | LangChain / SmolAgents / DeerFlow / DeepAgents profiles and config |
| `docs/mcp-servers.md` | MCP server configuration, CLI, and standalone scripts |
| `docs/browser_control.md` | Browser automation (sandbox vs. direct Playwright) |
| `docs/sandbox_support.md` | OpenSandbox Docker container setup and integration |
| `docs/deer-flow.md` | DeerFlow integration — profiles, modes, chat commands |
| `docs/webapp.md` | Built-in Streamlit webapp — `make webapp`, built-in agent pages, `genai_tk://` cross-package nav, `cli init` |
| `docs/TESTING_GUIDE.md` | Pytest fixtures, fake LLM/embeddings, async tests |
