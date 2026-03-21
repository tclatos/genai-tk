# Changelog

All notable changes to genai-tk will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) — [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`cli sandbox` commands** — `start`, `stop`, `status`, `pull` for managing the OpenSandbox daemon.
  Pre-warming reduces `--sandbox docker` startup from ~28 s to ~5 s.
- **Browser automation** — `Browser Agent` (sandbox) and `Browser Agent Direct` (host Playwright) profiles
  with 13 tools, credential hiding, cookie persistence, and SKILL.md support.
  `SandboxBrowserConfig.launch_mode: fresh` for anti-detection on bot-sensitive sites.
- **Direct browser tools** (`genai_tk/tools/direct_browser/`) — host-local Playwright with identical
  tool names to the sandbox suite. Validated with Enedis, Le Monde, bot.sannysoft.com.
- **Deep Agent Text-to-SQL example** — `text2sql` profile with `SkillsMiddleware` progressive
  skill disclosure (`query-writing/SKILL.md`, `schema-exploration/SKILL.md`).
- **Unified LangChain Agent System** — single `cli agents langchain` with `type: react | deep | custom`,
  global defaults + per-profile overrides, middleware and checkpointer as YAML config.
- **`AioSandboxBackend`** — full `deepagents.SandboxBackendProtocol` implementation: `aexecute`,
  `als_info`, `aread`, `awrite`, `aedit`, `agrep_raw`, `aglob_info`, `aupload_files`, `adownload_files`.
- **`BackendConfig` in `langchain.yaml`** — `type: none | aio_sandbox | class` selects the
  execution backend per profile; lifecycle managed by the factory.

### Fixed
- DeerFlow CLI: removed duplicate response text, suppressed internal print noise, fixed `%s`
  format string bugs, stripped chain-of-thought markers from final response.
- `SandboxBrowserSession._ensure_connected()` no longer reconnects on transient navigation errors.
- Sandbox browser locale/timezone now correctly propagated via `BROWSER_EXTRA_ARGS`.

## [0.1.0] — 2025-09-30

Initial standalone release, extracted from genai-blueprint.

- LLM Factory, Embeddings Factory, Vector Store Factory (Chroma, Weaviate, PGVector)
- ReAct Agent, SQL Agent, custom LangGraph graphs
- MCP Client, Chain Registry, Structured Output
- LangChain tools: web search, SQL, RAG, config loader
- SmolAgents tools: browser, DataFrame, SQL, YFinance
- RAG pipeline: BM25, ensemble retriever, Mistral OCR loader
- Knowledge Graph utilities (Cognee, Kuzu)
- Hierarchical YAML configuration with OmegaConf, env var substitution
- Unified CLI (`cli`) with framework-specific shells
- Python 3.12+, uv, ruff, pytest with asyncio support
