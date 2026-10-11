# Development Guidelines for GenAI Toolkit

## Skill Index

Full, up-to-date list of domain skills (dev skills for working on genai-tk
itself): [skills/README.md](skills/README.md). Start with skill `repo-map` for
orientation before making cross-cutting changes — it maps every feature area
to its docs, code, and config. Do not duplicate the skill list here; this
file only references it.

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
- **Avoid `Any` whenever possible:** Maximize explicit typing, domain models, generic parameters, and class inheritance hierarchies. Use `TypeAlias`, discriminated unions (`Annotated[... , Field(discriminator=...)]`), or specific union types instead of untyped `Any` containers.

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
- Read every top-level YAML key via a typed Pydantic model — `global_config().section(...)` for a single object, `global_config().section_dict(...)` for a dict of objects (supports discriminated unions). Both return an empty model/dict when the key is absent — they never raise.
- See skill `configuration` for the full access patterns, the key-models reference table, and the steps to add a new config section.

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
- Use `cli agents list` to inspect configured profiles
- Agent type `react` is the default; use `deep` for multi-step planning + skills
- Keep system prompts in YAML (`system_prompt:` field), not hardcoded in Python
- Use `SkillsMiddleware` + SKILL.md files for domain knowledge — avoids bloating
  system prompts and enables progressive disclosure
- See skill `agent-profiles` for the full profile YAML format and harness options

### Tools
- Define tools as LangChain `BaseTool` subclasses or `@tool`-decorated functions
- Register via factory function (`create_*_tools()`) referenced in agent YAML — see skill `add-tool` for the full contract
- Use `browser_fill_credential` (never plain `browser_type`) for credentials
- Tools that call external APIs must validate inputs at the boundary

### CLI commands
- Subclass `genai_tk.cli.base.CliTopCommand`, register in `config/app_conf.yaml` under `cli.commands`
- See skill `cli-and-scaffolding` for the full `CliTopCommand` contract and registration steps

### Sandbox
- `--sandbox docker` for isolated execution; `--sandbox local` for development
- Run `cli sandbox start` + `cli sandbox pull` once per boot to pre-warm
- Skill directories are automatically bind-mounted into the container (read-only)
- See `docs/sandbox_support.md` for setup and `docs/browser_control.md` for the browser tooling

### Skills (SKILL.md)
- One SKILL.md per domain — organized across 4 tiers in `skills/{runtime,development,governance,vendor,custom}/<name>/SKILL.md`
- Skills are read by the agent on demand, not injected into every prompt
- Use `cli skills list` to see all discovered skills across tiers; `cli skills validate --all` to check them
- Install community skills: `cli skills add --skillssh langchain-ai/langchain-skills`
- Create a new skill: `cli skills create <name>` (scaffolds skills.sh-format SKILL.md)
- Keep site-specific selectors, commands, and domain details in the skill file, not in Python

## Documentation & Skills Invariants (Mandatory on Every Commit)

**Rule: Documentation and skills MUST be updated in the same commit as the code changes.**

Whenever you introduce a new feature, refactor architecture, add a provider/driver, or modify configurations/CLI flags:
1. **Update Skills**: Any change affecting tools, models, workflows, or CLI commands must be reflected in the matching `skills/**/SKILL.md` file. Always run `uv run cli skills validate --all` to ensure zero schema errors.
2. **Update Design / Guides**:
   - `docs/design/`: Foundational architectural blueprints and specifications, organized in a didactic numbered sequence (`01_configuration_and_profiles.md`, `02_ioc_provider_architecture.md`, ...). Update these when internal design patterns, IoC registries, protocols, or class hierarchies evolve.
   - `docs/`: User-facing guides and tutorials (`core.md`, `agents.md`, `workflows.md`, `trajectory.md`). Update these when user-facing APIs, CLI commands, or YAML syntax change.
   - `docs/studies/`: Strictly reserved for research, external surveys, benchmark evaluations, and proposals that have **not** yet been implemented. Once an architecture from a study is implemented, promote the design to `docs/design/` or core documentation and remove it from `docs/studies/`.
3. **Update CHANGELOG.md**: Record notable changes under `## [Unreleased]` in a pedagogical, readable style explaining *what* changed and *why* (leave raw line-by-line diffs to git).
4. **Linkification & Style**: Never use backticks around file links. Use clean Markdown links `[path/file.py](path/file.py)`. Do not cite line numbers with backticks.

Current docs index:

| File | Topic |
|---|---|
| `docs/design/README.md` | Didactic 9-part architectural design curriculum |
| `docs/design/01_configuration_and_profiles.md` | Configuration management, profile overlays, proxy derivation |
| `docs/design/02_ioc_provider_architecture.md` | Inversion of Control (IoC) provider registry, role-based class fields, mixins |
| `docs/design/03_core_llm_and_factories.md` | Core LLM & Embeddings factories, model IDs, semantic caching, LCEL chains |
| `docs/design/04_decision_models_system1.md` | Fast System 1 decision models (Clef, Laya, Jev, OpenAI), calibrated rubrics |
| `docs/design/05_agent_harness_and_middleware.md` | Unified BaseHarness, event streaming, LangChain/DeerFlow adapters, PII/routing |
| `docs/design/06_sandbox_and_codeact.md` | Sandbox isolation, Docker pre-warming, `ulimit core=0`, LibreOffice isolation |
| `docs/design/07_workflow_engine_and_prefect.md` | Declarative workflows, `prefect-yaml`, DAG dependencies, route tables |
| `docs/design/08_document_processing_pipeline.md` | Multi-tier converter ladder (Docling, Mistral, MarkItDown), table preservation |
| `docs/design/09_observability_and_trajectories.md` | NeMo Relay OTLP to Langfuse, ATOF TrajectoryStore, terminal TUI, evaluators |
| `docs/core.md` | `LlmFactory`, `EmbeddingsFactory`, `EmbeddingsStore`, `LlmCache`, `ChainRegistry` |
| `docs/cli.md` | All CLI command groups, sub-commands, and how to add new commands |
| `docs/trajectory.md` | NeMo Relay ATOF trajectory store, replay engines, and TUI |
| `docs/scaffolding.md` | `cli init` — template presets, generated files, skills setup, multi-agent support |
| `docs/extra.md` | Non-pipeline tooling: agent graphs, NLP, BAML, image analysis, KV store, PgVector |
| `docs/nlp.md` | NLP package (`extra.nlp`): spaCy engine, model manager, PII detection, French support |
| `docs/rag.md` | RAG deep-dive — `RetrieverFactory`, `ManagedRetriever`, hybrid search, agent tools |
| `docs/markdownize.md` | Document conversion — Docling, Mistral OCR, MarkItDown profiles |
| `docs/baml.md` | BAML structured extraction — setup, CLI, programmatic API |
| `docs/workflows.md` | YAML-driven task orchestration, route tables, dispatcher |
| `docs/prefect.md` | Prefect flows — ephemeral vs. server mode, available flows, workflow engine |
| `docs/agents.md` | LangChain / DeerFlow agent profiles, config, and the shared harness layer |
| `docs/harness.md` | `agents.harness` — `BaseHarness`, shared event model, adapters, middleware |
| `docs/mcp-servers.md` | MCP 2.0 server configuration, transports (`stdio`, `sse`), and tools |
| `docs/browser_control.md` | Browser automation (sandbox vs. direct Playwright) |
| `docs/sandbox_support.md` | OpenSandbox Docker container setup, pre-warming, and integration |
| `docs/deer-flow.md` | DeerFlow integration — profiles, modes, chat commands |
| `docs/webapp.md` | Built-in Streamlit webapp — `just webapp`, built-in agent pages, `genai_tk://` nav |
| `docs/docker.md` | Docker image build — generic Dockerfile, just recipes, extras |
| `docs/TESTING_GUIDE.md` | Pytest fixtures, fake LLM/embeddings, async tests |
| `docs/benchmark_framework.md` | Unified multi-dataset benchmark framework architecture, stages, and CLI/TUI |
| `docs/benchmarks_financebench_officeqa.md` | Benchmark implementation studies & empirical evaluation on FinanceBench / OfficeQA |
| `docs/deerflow_architecture_and_benchmarks.md` | DeerFlow & meta-harness architecture, sandboxed Python calculations report |
| `docs/studies/README.md` | Index of unimplemented research, external framework surveys, and exploratory studies |
