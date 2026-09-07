# CodeAct — Python Code as the Agent Action Language

> **Status: experimental — not fully tested.**
> CodeAct is new. The unit tests cover the executor, sandbox tool binding,
> `final_answer` termination, and profile wiring, but end-to-end runs with real
> LLMs have not been systematically validated yet. Expect rough edges and
> behavior changes.

CodeAct is a SmolAgents-inspired agent paradigm: instead of expressing every
action as a discrete tool call, the LLM writes **Python code blocks** that are
executed in a safe in-process sandbox. Multi-step reasoning, loops,
conditionals, and stateful tool composition become ordinary Python. Sibling
tools (web search, …) are bound as plain callables inside the sandbox, printed
output becomes the next observation, tracebacks drive retries, and the run
terminates when the code calls `final_answer(result)`.

## Quick start

```bash
uv run cli agents run codeact \
  "How many seconds does it takes to a leopart at full speed to run through the Pont des Arts. Use Codeact skill" \
  -m glm5.2@openrouter
```

This runs the `codeact` example profile: the agent looks up the Pont des Arts
length (≈155 m) and the leopard top speed (≈58 km/h) with the sandboxed search
tool, computes the answer in Python, and returns it via `final_answer`.

Useful variants:

```bash
# Interactive chat with the same profile
uv run cli agents run codeact --chat -m glm5.2@openrouter

# Raw NDJSON event stream (useful for debugging the CodeAct loop)
uv run cli agents run codeact "..." -m glm5.2@openrouter --json
```

## Architecture

```text
LLM turn 1..N                    Sandbox (in-process)              Tools
─────────────                    ────────────────────              ─────
write python code block  ─────►  AST interpreter (no eval)
                                         │
                                         ├── web_search(...)  ───► search provider
                                         ├── final_answer(x)  ──► stop, return x
                                         └── print(...)       ──► captured as logs
observation ◄────  logs + last expression value (or traceback on error)
```

Components (all under `genai_tk/agents/tools/python_executor/`):

| Piece | Role |
|---|---|
| `executor.evaluate_python_code` | AST walker that interprets a subset of Python: assignments, control flow, functions, classes, comprehensions, try/except, with |
| `executor.LocalPythonExecutor` | Stateful executor: persists variables across calls, enforces timeouts and import allow-lists |
| `tool.PythonExecutorTool` | LangChain `BaseTool` wrapper (name `python_interpreter`) that renders `CodeOutput` as text, with a `FINAL ANSWER:` marker on termination |
| `factory.bind_executor_tools` | At agent-creation time, registers every sibling tool as a callable inside each executor sandbox |

## The protocol

1. The agent plans one small step and writes a short Python block.
2. The block runs in the sandbox. `print()` output and the last expression
   value come back as the observation; variables persist across blocks.
3. On error, the exception message becomes the observation and the agent is
   expected to fix and retry.
4. Calling `final_answer(value)` raises an internal `FinalAnswerException`
   (a `BaseException` subclass, so agent code can't accidentally catch it),
   stops execution, and surfaces the value with a `FINAL ANSWER:` marker.

## Safety model

The interpreter is not `eval()`:

- **Import allow-list** — only modules in `BASE_BUILTIN_MODULES` plus profile
  extras (`additional_authorized_imports`) can be imported; dangerous modules
  (`os`, `sys`, `subprocess`, `socket`, `threading`, …) and functions
  (`eval`, `exec`, `compile`, `__import__`, …) are blocked.
- **Dunder restrictions** — forbidden dunder attribute/function access.
- **Timeout** — per-call execution timeout (default 30 s).
- **Resource caps** — max operations and loop-iteration counters stop runaway
  loops; output is truncated to 50 000 characters.

Note: this runs **in-process** on the host interpreter, sandboxed by AST-level
checks — not in a container. Don't point it at untrusted users without
understanding that distinction.

## Configuration

Profiles live in `config/examples/agents/codeact.yaml`:

- **`codeact`** — standalone deep agent. Tools: the Python executor plus
  `create_search_tool` (bound as `web_search(...)` inside the sandbox). Loads
  the CodeAct skill from `skills/custom/` via `skill_directories`.
- **`codeact-orchestrator`** — deep agent with a single `codeact` **subagent**
  (declared under `subagents:`) that owns the executor; the orchestrator only
  delegates via the `task` tool.

Example tool wiring:

```yaml
tools:
  - factory: genai_tk.agents.tools.python_executor.tool.create_python_executor_tools
  - factory: genai_tk.agents.tools.langchain.search_tools_factory.create_search_tool
```

Only `BaseTool` instances survive profile loading; factories returning raw
functions are silently dropped (this is why the profiles use
`create_search_tool`, not `create_search_function`).

## Extending

- Add imports the sandbox may use with `additional_authorized_imports`.
- See `skills/custom/codeact/SKILL.md` for the runtime prompt protocol.
- For developer usage of the interpreter (embedding it in your own code), read
  the `genai-tk-python-interpreter` dev skill in `skills/genai-tk/`.
- Tests: `tests/unit_tests/tools/test_python_executor.py` and the factory
  structural tests under `tests/unit_tests/agents/langchain/`.
