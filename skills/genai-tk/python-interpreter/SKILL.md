---
name: python-interpreter
description: Use the genai-tk safe AST Python interpreter — embed it in Python code (simple case) or wire it as an agent tool for CodeAct. Use when writing code that executes LLM/agent-generated Python, extending the sandbox, or debugging python_executor behavior.
---

# AST Python Interpreter

Safe, in-process Python execution. Code is parsed to an AST and interpreted —
never `eval()`'d — with an import allow-list, timeouts, and resource caps.

## Read First

- `genai_tk/agents/tools/python_executor/executor.py` — the interpreter and `LocalPythonExecutor`
- `genai_tk/agents/tools/python_executor/tool.py` — LangChain `BaseTool` wrapper and factories
- `genai_tk/agents/tools/python_executor/models.py` — `CodeOutput` / `PythonExecutorInput`
- `tests/unit_tests/tools/test_python_executor.py` — canonical usage examples
- `docs/codeact.md` — the agent-facing CodeAct overview

## Simple case: run a snippet in your own code

Use `LocalPythonExecutor` directly. State (variables) persists across calls:

```python
from genai_tk.agents.tools.python_executor import LocalPythonExecutor

executor = LocalPythonExecutor(
    additional_authorized_imports=["pandas"],  # extras on top of the safe stdlib list
    timeout_seconds=30,                        # per-call execution timeout
)

result = executor("x = 40\nprint(x + 2)\nx / 4")
print(result.output)              # 10.0  (last expression value)
print(result.logs)                # "42\n" (captured print output)
print(result.error)               # None, or the error message on failure
print(result.is_final_answer)     # False
```

Semantics worth knowing:

- The value of the **last expression** is returned as `output`; `print()` goes
  to `logs` (joined into the tool text as `Logs:`).
- Errors are returned in `CodeOutput.error` (not raised); the message includes
  the failing source segment.
- Assigning to a name that is a registered tool raises — tools can't be erased.
- One-shot variant: `evaluate_python_code(code, static_tools=..., state=...)`
  returns `(result, is_final_answer)`.

Injecting tools/functions:

```python
executor = LocalPythonExecutor(tools=[my_base_tool, plain_function])
executor.send_tools([another_tool])   # re-register later (replaces the toolset)
executor.send_variables({"df": df})   # pre-seed state with Python objects
executor.reset()                      # clear state
```

`tools` accepts LangChain `BaseTool` instances (adapted so the sandbox calls
them as plain functions named after `tool.name`) or plain callables (registered
under `__name__`). `final_answer` is always available: calling it stops
execution and sets `is_final_answer=True`.

## CodeAct: as an agent tool

Wrap the executor as a LangChain tool so an LLM can act by writing code:

```python
from genai_tk.agents.tools.python_executor import create_python_executor_tool

tool = create_python_executor_tool(
    additional_authorized_imports=["json"],
    tools=[search_tool],     # siblings bound into the sandbox namespace
    timeout_seconds=30,
)
# tool.name == "python_interpreter"; tool.invoke({"code": "..."}) -> str
```

In agent profiles, declare it in YAML — the factory binds all sibling tools
into the sandbox automatically (`bind_executor_tools` in
`genai_tk/agents/langchain/factory.py`):

```yaml
tools:
  - factory: genai_tk.agents.tools.python_executor.tool.create_python_executor_tools
  - factory: genai_tk.agents.tools.langchain.search_tools_factory.create_search_tool
```

The prompt-side protocol (print observations, retry on traceback, terminate
with `final_answer`) lives in `skills/custom/codeact/SKILL.md`. End-to-end
profiles: `config/examples/agents/codeact.yaml`.

## Safety envelope

Do not weaken it casually:

- Imports are restricted to `BASE_BUILTIN_MODULES` + `additional_authorized_imports`;
  `os`, `sys`, `subprocess`, `socket`, `threading`, … are hard-blocked.
- Dunder attribute/function access is blocked except a small allow-list.
- `FinalAnswerException` subclasses `BaseException` on purpose — generic
  `except Exception` in agent code must not swallow it.
- Execution runs **in-process** on the host: it is sandboxed by AST checks,
  not by an OS container. See `docs/codeact.md` for the full model.

## Tests & validation

```bash
uv run pytest tests/unit_tests/tools/test_python_executor.py -q
```

Add tests there when changing interpreter behavior — they cover arithmetic,
state persistence, tool calls, `final_answer` termination, error formatting,
timeouts, and forbidden operations.
