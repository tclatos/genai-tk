---
name: python-interpreter
description: Configure isolated Python execution for CodeAct or explicitly trusted local AST execution. Use when extending python_executor, changing its trust boundaries, or debugging worker lifecycle, tool relay, and execution limits.
---

# Python Execution and Isolation

Use the Docker executor for agent-generated code. The factory defaults to
`executor_type="docker"`; `auto` and `sandbox` also require isolated execution.
Errors never trigger host fallback. Use `executor_type="local"` or
`LocalPythonExecutor` only for explicitly trusted host development.

## Read First

- [docker_executor.py](../../../genai_tk/agents/tools/python_executor/docker_executor.py) — isolated execution and tool relay
- [executor.py](../../../genai_tk/agents/tools/python_executor/executor.py) — trusted local AST interpreter
- [tool.py](../../../genai_tk/agents/tools/python_executor/tool.py) — LangChain wrapper and factories
- [models.py](../../../genai_tk/agents/tools/python_executor/models.py) — result/input contracts
- [CodeAct guide](../../../docs/codeact.md) and [sandbox design](../../../docs/design/06_sandbox_and_codeact.md)

## Isolated execution

The executor owns a fresh backend and keeps variables within its session:

```python
from genai_tk.agents.tools.python_executor import SandboxedPythonExecutor

executor = SandboxedPythonExecutor(timeout_seconds=30)
try:
    result = await executor.aexecute_code("x = 40\nprint(x + 2)\nx / 4")
    print(result.output)  # 10.0
finally:
    await executor.aclose()
```

Install the `harnessing` extra and prepare Docker/OpenSandbox first. Prefer
`aclose()` in async callers; `close()` is the synchronous cleanup wrapper.
Default sessions omit configured environment variables and volumes. Passing
`backend=` deliberately uses a caller-owned trust boundary and can grant those
capabilities. Never share that backend between mutually untrusted sessions.
Known host-shell backends, including a composite backend's local default, are
rejected. Custom backends must implement isolation themselves. Closing a default
executor retires its container and leaves the shared OpenSandbox daemon running
for peer sessions; stop the daemon explicitly with `uv run cli sandbox stop`.

## Trusted local development

Local state also persists, but code runs with the host process's privileges:

```python
from genai_tk.agents.tools.python_executor import LocalPythonExecutor

executor = LocalPythonExecutor(
    additional_authorized_imports=["pandas"],  # Explicit local import permission
    timeout_seconds=30,  # Does not terminate a running host thread/native call
)

result = executor("x = 40\nprint(x + 2)\nx / 4")
print(result.output)  # 10.0  (last expression value)
print(result.logs)  # "42\n" (captured print output)
print(result.error)  # None, or the error message on failure
print(result.is_final_answer)  # False
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
executor.send_tools([another_tool])  # re-register later (replaces the toolset)
executor.send_variables({"df": df})  # pre-seed state with Python objects
executor.reset()  # clear state
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
    executor_type="docker",
    tools=[search_tool],  # tools exposed inside the sandbox namespace
    timeout_seconds=30,
)
# tool.name == "python_interpreter"; tool.invoke({"code": "..."}) -> str
```

In agent profiles, declare it in YAML with its internal tools configured directly:

```yaml
tools:
  - genai_tk.agents.tools.python_executor.tool.create_python_executor_tools:
      executor_type: docker
      tools:
        - genai_tk.agents.tools.langchain.search_tools_factory.create_search_tool
```

The prompt-side protocol lives in the [CodeAct runtime skill](../../runtime/codeact/SKILL.md).
Example profiles are in [codeact.yaml](../../../config/examples/agents/codeact.yaml).

For sandbox tools, plain callables are registered directly and are not invoked
during initialization. Request factory initialization explicitly through its
qualified name, a configuration mapping, or `FactoryToolSpec`; typed
`UnifiedToolSpec`, `ClassToolSpec`, and `FunctionToolSpec` are also supported.

## Execution contract

- Sandboxed execution opens no host listener. Host tool calls and results use
  bounded JSON over the existing sandbox transport and an explicit tool allowlist.
- Requests authenticate before JSON parsing. Session files and credentials are
  unique, and calls are serialized. Tokens do not isolate hostile peers already
  sharing the same container or OS user.
- The worker supervisor can kill its child process group at the wall deadline;
  logs are bounded while produced. A deadline or infrastructure failure invalidates
  the executor and retires its owned backend. Do not reuse or reset it afterwards.
  Treat failed backend termination as a cleanup failure requiring intervention.
- A caller-owned backend is not destroyed automatically. After a timeout the
  caller must retire it to stop detached descendants, then create a new executor.
- `reset()` clears variables and agent-defined functions. `send_tools()` replaces
  capabilities and revokes retained proxies to removed tools. Sandbox initial
  state and `send_variables()` values must be JSON-compatible.
- Host tools are privileged capabilities. Validate inputs and authorization in
  each tool. Await timeouts cannot undo or always stop a tool's side effects;
  never automatically replay a block with an uncertain outcome.
- `fallback_to_local=True` is rejected. Local execution must be selected explicitly.

Local import/dunder restrictions and operation counters help catch mistakes;
they are not a security boundary. Local thread timeouts cannot kill native code
or host tools. Optional installed packages are not automatically authorized:
declare local imports explicitly. Container code uses its image's packages and
does not apply the local AST allowlist.

## Tests & validation

```bash
uv run pytest tests/unit_tests/tools/test_python_executor.py \
  tests/unit_tests/tools/test_docker_python_executor.py \
  tests/unit_tests/tools/test_sandbox_worker.py -q
uv run cli skills validate --all
```

Test fail-closed behavior, authorization, session separation, full reset,
deadlines that stop side effects, bounded output, and cleanup. Mark socket,
subprocess, and filesystem tests as integration tests. Real container checks
require `--include-docker`; process tests do not prove deployment isolation.
