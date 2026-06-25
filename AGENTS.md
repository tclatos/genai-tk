# Development Guidelines for GenAI Toolkit

## Build Commands

```bash
just install-dev   # install with development dependencies
just fmt           # format with ruff (includes import sorting)
just lint          # lint with ruff
just test          # run all tests (unit + integration)
just test-unit     # unit tests only
just test-integration  # integration tests only
just check         # fmt + lint + test
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

### Two canonical patterns for reading config

**Case 1 — single Pydantic model** (`section`):
```python
from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.utils.prefect_server import PrefectConfig

cfg = global_config().section("prefect", PrefectConfig)
print(cfg.host, cfg.port)          # typed, validated, with defaults
```

**Case 2 — dict of Pydantic models** (`section_dict`):
```python
from genai_tk.core.embeddings_store import EmbeddingsStoreConfig

stores = global_config().section_dict("embeddings_store", EmbeddingsStoreConfig)
default = stores["default"]        # EmbeddingsStoreConfig, validated at load time
```

`section_dict` also supports Pydantic **discriminated unions** (pass an `Annotated` type):
```python
from genai_tk.extra.kv_store_registry import KvStoreConfig

kv = global_config().section_dict("kv_store", KvStoreConfig, inject_name=False)
```

Both methods return empty model / empty dict when the key is absent — they never raise.

### When to add a new config section

| Scenario | Action |
|---|---|
| New top-level YAML key with fixed fields | Create a Pydantic model, add `section()` accessor |
| New top-level YAML key with named entries | Create a Pydantic model, add `section_dict()` accessor |
| Named entries with `type` discriminator | Create a discriminated union, use `section_dict()` with `inject_name=False` |

### Key models reference

| YAML key | Model | Location |
|---|---|---|
| `prefect` | `PrefectConfig` | `genai_tk.utils.prefect_server` |
| `cli` | `CliConfig` | `genai_tk.main.cli` |
| `sandbox` | `SandboxConfig` | `genai_tk.agents.sandbox.models` |
| `monitoring` | `MonitoringConfig` | `genai_tk.utils.tracing` |
| `auth` | `AuthConfig` | `genai_tk.utils.basic_auth` |
| `kv_store` (dict) | `KvStoreConfig` (union) | `genai_tk.extra.kv_store_registry` |
| `embeddings_store` (dict) | `EmbeddingsStoreConfig` | `genai_tk.core.embeddings_store` |
| `structured` (dict) | `StructuredConfig` | `genai_tk.extra.structured.baml_util` |
| `llm` | `LlmSection` | `genai_tk.core.factories.llm_factory` |
| `embeddings` | `EmbeddingsSection` | `genai_tk.core.factories.embeddings_factory` |
| `paths` | `PathsConfig` | `genai_tk.config_mgmt.config_mngr` |

## Testing

- pytest with asyncio support (`pytest-asyncio`)
- Unit tests in `tests/unit_tests/`, integration tests in `tests/integration_tests/`
- Test files: `test_*.py` or `*_test.py`
- Use `faker` for test data generation
- No Docker required for unit tests; integration tests may require live services

### Test Conventions

**Never hardcode model IDs.** Use the shared fixtures from `tests/conftest.py`:

```python
def test_something(fake_llm_id: str, fake_llm, fake_embeddings_id: str, fake_embeddings):
    ...
```

These are resolved from the `pytest` profile in `config/app_conf.yaml` at run time.
The typed source is `genai_tk.config_mgmt.test_config.PytestConfig` / `get_pytest_config()`.

**Do not call `get_pytest_config()` at module level.** It runs before `switch_profile("pytest")`
fires (session-scoped autouse fixture). Call it only inside fixtures or test functions.

**Avoid `pytest_*` names** for fixtures — pytest treats them as hook registrations and
raises `PluginValidationError` at collection time. Prefer `test_cfg`, `fake_llm_id`, etc.

**Limit mocks.** Prefer fake providers (`parrot_local@fake`, `embeddings_768@fake`) over `unittest.mock` patches. Only mock at true system boundaries (external HTTP, filesystem side-effects); never mock internal genai-tk classes.

**One behaviour per test.** Prefer a single focused assertion over a multi-step scenario.
Use `@pytest.mark.parametrize` for repeated cases instead of a loop inside the test body.

**Use pytest style, not `unittest.TestCase`.** Fixtures with `yield` replace `setUp`/`tearDown`;
plain `assert` replaces `self.assertEqual`; `pytest.raises` replaces `self.assertRaises`.

**Mark tests correctly:**

| Marker | When to use |
|---|---|
| `@pytest.mark.unit` | Pure in-process logic, no I/O |
| `@pytest.mark.integration` | Starts real services or hits the filesystem |
| `@pytest.mark.fake_models` | Uses `parrot_local@fake` / `embeddings_768@fake` |
| `@pytest.mark.real_models` | Requires live API keys (skipped by default) |
| `@pytest.mark.performance_tests` | Throughput / latency benchmarks |

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
- One SKILL.md per domain — kept in `skills/custom/<name>/SKILL.md` (your project) or `skills/genai-tk/` (contributor skills)
- Skills are read by the agent on demand, not injected into every prompt
- Use `cli skills list` to see all discovered skills; `cli skills validate --all` to check them
- Install community skills: `cli skills add --skillssh langchain-ai/langchain-skills`
- Create a new skill: `cli skills create <name>` (scaffolds skills.sh-format SKILL.md)
- Keep site-specific selectors, commands, and domain details in the skill file, not in Python

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
| `docs/scaffolding.md` | `cli init` — template presets, generated files, skills setup, multi-agent support files |
| `docs/copilot-agent-support.md` | Copilot Agent skills, AGENTS.md, copilot-instructions.md |
| `docs/core.md` | `LlmFactory`, `EmbeddingsFactory`, `EmbeddingsStore`, `LlmCache`, `ChainRegistry` |
| `docs/extra.md` | Non-pipeline tooling: agent graphs, NLP, BAML, image analysis, KV store, PgVector |
| `docs/nlp.md` | NLP package (`extra.nlp`): spaCy engine, model manager, preprocessing, PII detection, anonymization, classifiers, French support |
| `docs/rag.md` | RAG deep-dive — `RetrieverFactory`, `ManagedRetriever`, all retriever types, CLI, Prefect flow, agent tools |
| `docs/prefect.md` | Prefect flows in `workflow/prefect/flows/` — markdownize, ppt2pdf, rag, baml |
| `docs/baml.md` | BAML structured extraction — setup, CLI, programmatic API |
| `docs/workflows.md` | YAML-driven task orchestration — defining workflows, profiles, CLI integration, multi-step pipelines |
| `docs/prefect.md` | Prefect flows — ephemeral vs. server mode, available flows, writing new flows, workflow engine integration |
| `docs/agents.md` | LangChain / SmolAgents / DeerFlow agent profiles and config |
| `docs/mcp-servers.md` | MCP server configuration, CLI, and standalone scripts |
| `docs/browser_control.md` | Browser automation (sandbox vs. direct Playwright) |
| `docs/sandbox_support.md` | OpenSandbox Docker container setup and integration |
| `docs/deer-flow.md` | DeerFlow integration — profiles, modes, chat commands |
| `docs/webapp.md` | Built-in Streamlit webapp — `just webapp`, built-in agent pages, `genai_tk://` cross-package nav, `cli init` |
| `docs/docker.md` | Docker image build — generic Dockerfile, just recipes, extras, scaffolded app setup |
| `docs/TESTING_GUIDE.md` | Pytest fixtures, fake LLM/embeddings, async tests |
