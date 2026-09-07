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

Only `BaseTool` instances survive profile loading; factories returning raw
functions are silently dropped (this is why the profiles use
`create_search_tool`, not `create_search_function`).

### Example profiles

Standalone CodeAct agent (excerpt of the `codeact` profile):

```yaml
agents:
  codeact:
    harness: langchain
    name: "CodeAct"
    type: deep
    description: "Solves tasks by writing Python code blocks in a sandbox; tools are plain functions, print() is the observation, final_answer(x) terminates"
    system_prompt: |
      You are a CodeAct agent. You solve tasks by writing Python code and
      executing it with the python_interpreter tool — never answer by prose alone.

      ## Protocol

      1. Think about the next single step, then express it as a short Python code block.
      2. Call python_interpreter with that code. Variables persist between calls.
      3. Tools available in your profile (e.g. web_search) are plain functions
         inside the sandbox: call them directly, e.g. `result = web_search("...")`.
      4. Observe: print() output and the last expression value come back as the result.
      5. If the result is an Error/traceback, fix the code and retry — do not give up.
      6. When the task is solved, call final_answer(value) with the final result.

    tools:
      - factory: genai_tk.agents.tools.python_executor.tool.create_python_executor_tools
      - factory: genai_tk.agents.tools.langchain.search_tools_factory.create_search_tool

    skill_directories:          # progressive disclosure of the CodeAct skill
      - ${paths.project}/skills/custom

    enable_planning: true
    enable_file_system: false
```

Orchestrator delegating to a codeact-only subagent (excerpt of
`codeact-orchestrator`; the orchestrator itself has no tools — it delegates
via deepagents' `task` tool):

```yaml
agents:
  codeact-orchestrator:
    harness: langchain
    name: "CodeAct Orchestrator"
    type: deep
    description: "Orchestrator that delegates every computation to a dedicated CodeAct subagent"
    system_prompt: |
      You are an orchestrator. You do NOT compute anything yourself.

      For every task that requires computation, data lookup, or multi-step
      reasoning, delegate to the 'codeact' subagent via the task tool and
      relay its final answer. Summarize results clearly for the user.

    tools: []

    subagents:
      - name: codeact
        description: "CodeAct worker: solves tasks by writing Python code in a sandbox."
        system_prompt: |
          You are a CodeAct agent. Solve tasks exclusively by writing Python
          code executed via the python_interpreter tool.

          - web_search(...) is available as a plain function in the sandbox.
          - Variables persist across calls; use print() for observations.
          - On error, read the traceback, fix the code, and retry.
          - Terminate by calling final_answer(result).

        tools:                        # resolved by _resolve_subagents and
          - factory: genai_tk.agents.tools.python_executor.tool.create_python_executor_tools
          - factory: genai_tk.agents.tools.langchain.search_tools_factory.create_search_tool
```

Subagent dict fields supported by `_resolve_subagents`: `name` and
`description` (required by deepagents), `system_prompt`, `model`, `skills`
(list of paths), and `tools` (list of tool specs, or `null` to inherit the
parent's tools). Subagent toolsets get the same CodeAct sandbox binding as
top-level profiles.

## Extending

- Add imports the sandbox may use with `additional_authorized_imports`.
- See `skills/custom/codeact/SKILL.md` for the runtime prompt protocol.
- For developer usage of the interpreter (embedding it in your own code), read
  the `genai-tk-python-interpreter` dev skill in `skills/genai-tk/`.
- Tests: `tests/unit_tests/tools/test_python_executor.py` and the factory
  structural tests under `tests/unit_tests/agents/langchain/`.
