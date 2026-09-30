# Codebase Audit — Security, Performance & Maintainability (genai-tk)

**Date:** 2026-09-30
**Scope:** `genai_tk/` package (this repo). A companion audit of the
`genai-graph` repo lives at
`../../../genai-graph/docs/studies/codebase_audit_2026-09.md` — read it too if
you use both packages together, since the highest-severity finding of the
pair (weak committed credentials) is in genai-graph.

**Method:** static review (grep + targeted reads) of the full `genai_tk/`
tree, cross-checked line-by-line for every finding below — no claim in this
report is speculative; each was confirmed by opening the referenced file.

## TL;DR

genai-tk is in noticeably better shape than the average AI-agent codebase on
the security axis: the CodeAct sandbox is a real AST interpreter (no
`eval`/`exec` on untrusted input), there is no `shell=True` anywhere, and
secrets are consistently sourced from `${oc.env:...}` rather than hardcoded.
The real problems are **maintainability** (several 1,000+ line "god modules")
and a handful of **small, cheap-to-fix** security/reliability gaps (missing
HTTP timeouts, a weak default DB password in an example config, silent
`except Exception` fallbacks). Async/parallelization is mostly fine already;
only a couple of sequential-HTTP and busy-polling spots are worth touching.

---

## 1. Security Findings

### 1.1 Missing HTTP timeouts on external API calls (Low/Medium — fix now, it's free)

[genai_tk/agents/tools/langchain/search_tools_factory.py](../../genai_tk/agents/tools/langchain/search_tools_factory.py#L83)
```python
response = requests.post(url, headers=headers, json={"q": query, "num": max_results})
```
Same pattern at line 280. Neither call passes `timeout=`. `url` is a
hardcoded `https://google.serper.dev/...` endpoint (not attacker-controlled),
so this isn't SSRF — but an unresponsive/slow upstream will hang the calling
agent thread indefinitely (a self-inflicted DoS). Every other outbound call
in the codebase (`docker_executor.py`'s `urllib.request.urlopen(..., timeout=45)`,
the sandbox subprocess calls) already sets a timeout — these two are the
exception, not the rule.

**Fix:** add `timeout=10` (or a configurable constant) to both calls.

### 1.2 Weak default Postgres password in shipped config (Low — hygiene)

[config/app_conf.yaml](../../config/app_conf.yaml#L58)
```yaml
url: postgresql+asyncpg://${oc.env:POSTGRES_USER,postgres}:${oc.env:POSTGRES_PASSWORD,password}@localhost:5432/genai
```
and line 67 (`unknown_user`/`password` defaults again). If a deployment
forgets to set `POSTGRES_PASSWORD`, the app silently connects with the
literal password `password`. Low risk on its own (localhost-only DSN in the
example config) but it's the kind of default that survives into a real
deployment by inertia.

**Fix:** drop the fallback (`${oc.env:POSTGRES_PASSWORD}` with no default) so
a missing env var fails loudly instead of connecting with a guessable
credential.

### 1.3 Sandbox / CodeAct interpreter — reviewed, no bypass found (informational)

[genai_tk/agents/tools/python_executor/executor.py](../../genai_tk/agents/tools/python_executor/executor.py)
(1,685 lines) implements a hand-rolled AST-walking interpreter rather than
`eval()`/`exec()` on raw source: import allow-list (`check_import_authorized`,
`BASE_BUILTIN_MODULES` vs. `DANGEROUS_MODULES` including `os`, `subprocess`,
`socket`, `ctypes`), a dunder-method whitelist (`ALLOWED_DUNDER_METHODS`), an
operation counter (`MAX_OPERATIONS = 10_000_000`) and a wall-clock timeout
(`MAX_EXECUTION_TIME_SECONDS = 30`, enforced via `ThreadPoolExecutor`).
[docker_executor.py](../../genai_tk/agents/tools/python_executor/docker_executor.py#L91)
does call `exec()`/`eval()`, but only on an AST that was already parsed and
validated, inside a container namespace — not on a raw string. No escape
vector was found. This is a well-designed control; keep it as the reference
implementation and don't let the `executor.py` split proposed in §2 change
its security-relevant invariants without a re-review.

### 1.4 Pickle deserialization — controlled, correctly documented (informational)

[genai_tk/utils/langchain_community_repl/sqlite_cache.py](../../genai_tk/utils/langchain_community_repl/sqlite_cache.py#L76)
and
[genai_tk/utils/streamlit/capturing_callback_handler.py](../../genai_tk/utils/streamlit/capturing_callback_handler.py#L55)
both call `pickle.load(s)` on local, developer-created files, and both
already carry an explicit docstring warning ("never open a `.db`/file
received from an untrusted source"). No change needed, but if either file
path is ever made configurable from a web form or CLI flag that a
less-trusted user can influence, this becomes an RCE — worth a comment at the
call site pointing back here if that ever happens.

### 1.5 Subprocess usage — audited, all safe (informational)

Every `subprocess.run`/`Popen` call in the package (`sandbox/manager.py`,
`sandbox/cli_commands.py`, `main/scaffolder.py`, `main/skills_manager.py`,
`cli/commands_test.py`, `workflow/prefect/flows/office2pdf_flow.py`, ...)
passes a list of arguments, never `shell=True`, and never builds the command
via string concatenation of external input. No injection vector found.

### 1.6 Silent `except Exception` fallbacks hide config errors (Medium — maintainability + security-adjacent)

Pattern repeated 20+ times, e.g.
[genai_tk/agents/tools/direct_browser/factory.py](../../genai_tk/agents/tools/direct_browser/factory.py#L33):
```python
try:
    ...load config...
except Exception:
    return DirectBrowserConfig()   # falls back to defaults, no log line
```
Same shape in `sandbox_browser/factory.py` (L41, L64) and several spots in
`cli/commands_info.py`. A malformed or *maliciously edited* config file
(e.g. someone weakens a browser sandbox policy) fails closed to a default
that may be **more permissive**, and nobody is told it happened. Distinguish
"expected, log at debug" from "unexpected, log at warning" and never let a
config-parsing exception silently downgrade a security-relevant default.

**Fix:** at minimum, `logger.warning(f"... falling back to defaults: {exc}")`
before returning the default in every one of these branches.

---

## 2. Large Modules That Should Be Split

All six mix unrelated concerns in one file (config schema + business logic +
CLI/public API), which makes them hard to navigate, test in isolation, and
review in PRs (a one-line change to a Pydantic field touches a 1,500-line
diff-review surface).

| File | Lines | Mixed responsibilities found |
|---|---|---|
| [genai_tk/agents/tools/python_executor/executor.py](../../genai_tk/agents/tools/python_executor/executor.py) | 1,685 | exception types, security allow-lists, ~30 AST `evaluate_*` node visitors, the orchestrator class, LangChain adapter — 6 concerns in one file |
| [genai_tk/core/factories/llm_factory.py](../../genai_tk/core/factories/llm_factory.py) | 1,577 | Pydantic schemas (`LlmModelsConfig`/`LlmSection`/`LlmInfo`), fuzzy-name matching (~120 lines), model DB resolver, the 700-line `LlmFactory` class covering every provider, public `get_llm()`/`get_llm_info()` API |
| [genai_tk/config_mgmt/config_mngr.py](../../genai_tk/config_mgmt/config_mngr.py) | 1,260 | schemas, YAML merge/profile state machine, gitignore-pattern matcher, 20+ accessor methods (`get_str`/`get_bool`/`get_dsn`/...), generic YAML loader utility |
| [genai_tk/cli/commands_info.py](../../genai_tk/cli/commands_info.py) | 978 | a single `InfoCommands` class holding 7 unrelated subcommands (`config`, `models`, `llm-profile` at ~320 lines, `mcp-tools`, `commands`, `ls`, `config-keys`) |
| [genai_tk/core/factories/retriever_factory.py](../../genai_tk/core/factories/retriever_factory.py) | 807 | 7 Pydantic config classes, 3 document-store classes, `ManagedRetriever` orchestrator, the factory itself, a compressor builder |
| [genai_tk/utils/trajectory_store.py](../../genai_tk/utils/trajectory_store.py) | 738 | 8 Pydantic data models + the `TrajectoryStore` I/O/stats/export class + 6 free-function helpers |

**Suggested split pattern** (applies to all six — same shape each time):
`*_models.py` (Pydantic schemas only) → `*_matching.py` / `*_merge.py` /
domain-specific helper module → the original filename kept as a thin
orchestrator/public-API module that imports the above. This is a pure
move-and-import refactor (no behavior change), so it's safe to do
incrementally, file by file, starting with `executor.py` since it's both the
largest and the most security-sensitive (a smaller file is easier to review
for the invariants in §1.3).

---

## 3. Async / Parallelization Opportunities

Overall this codebase already uses `asyncio.gather` + semaphores where it
matters (see the embedding pattern in genai-graph's `ingest.py`, which
genai-tk's own embeddings/retriever code follows). Remaining opportunities
are minor:

- **[genai_tk/utils/prefect_server.py](../../genai_tk/utils/prefect_server.py#L198)** — a 90-iteration `time.sleep(0.5)` polling loop (up to 45s) waiting for the Prefect server to become ready. Fine functionally, but blocks the calling thread the whole time; if this is ever called from an async context, swap for `asyncio.sleep` + `asyncio.wait_for`, or at least shorten the fixed interval with backoff so typical startup (fast) doesn't always cost extra iterations.
- **[genai_tk/agents/tools/langchain/search_tools_factory.py](../../genai_tk/agents/tools/langchain/search_tools_factory.py#L83)** — the two `requests.post` calls noted in §1.1 are synchronous; if a caller ever needs to fan out multiple search queries, this should move to `httpx.AsyncClient` + `asyncio.gather` rather than a Python-level loop of blocking calls. Not urgent today since call sites appear to be single-query.
- No `time.sleep()` was found inside an `async def` function (which would block the event loop) — the loops above are in plain sync functions, so no correctness bug, just a latency one.

---

## 4. Other Code Smells

- **TODOs worth resolving:** [genai_tk/workflow/loaders/mistral_ocr.py](../../genai_tk/workflow/loaders/mistral_ocr.py#L36) has a typo'd `# TODO : Impletent Asnyc` — the loader is a common OCR path and probably genuinely benefits from an async rewrite (multiple document pages processed sequentially); worth turning into an actual ticket rather than a comment.
- **Bare `except Exception:`** is used ~100+ times across the tree; most return a sensible fallback and are fine, but the ones that hide *config/security* decisions (§1.6) should be tightened first — don't try to fix all 100 at once, that's a low-value mechanical change.
- No mutable default arguments were found in actual source (only in a SKILL.md documentation example demonstrating what *not* to do).
- No bare `except: pass` was found anywhere — exceptions are always at least bound to a name, which is good practice already in place.

---

## 5. Prioritized Action List

| # | Effort | Item | Status |
|---|---|---|---|
| 1 | 5 min | Add `timeout=10` to both `requests.post()` calls in `search_tools_factory.py` (§1.1) | ✅ Done |
| 2 | 5 min | Remove the `password`/`unknown_user` fallback defaults in `config/app_conf.yaml` (§1.2) | ✅ Done |
| 3 | 30 min | Add `logger.warning(...)` before every silent config-fallback `except Exception: return Default()` (§1.6) | ✅ Done for `direct_browser/factory.py` and `sandbox_browser/factory.py`; the ~20 other silent-fallback sites noted in §1.6 are still open (mechanical, low-value to batch-fix) |
| 4 | 1–2 days | Split `python_executor/executor.py` into `interpreter_exceptions.py` / `interpreter_allowlists.py` / `interpreter_safety.py` / `ast_evaluators.py` / thin `executor.py` (§2), re-running the sandbox test suite after each extraction | ⚠️ Not done — tracked as follow-up |
| 5 | 1–2 days | Split `llm_factory.py`, `config_mngr.py`, `commands_info.py`, `retriever_factory.py`, `trajectory_store.py` along the lines in §2 | ⚠️ Not done — tracked as follow-up |
| 6 | as-needed | Swap the Prefect-server readiness poll to backoff-based waiting (§3) | ⚠️ Not done — tracked as follow-up |

