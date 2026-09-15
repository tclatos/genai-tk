---
name: codeact
description: Execute multi-step reasoning, data retrieval, and calculations via Python code blocks in a safe sandbox. Minimizes agentic loop turns by running tools (like web search) inside Python. Activate when user requests "use codeact skill" or needs programmatic multi-step problem solving.
---

# CodeAct: Code Acts as Agents

CodeAct is a SmolAgents-inspired paradigm where actions are composed and executed as **Python code blocks** inside a safe sandbox. Instead of making discrete tool calls in the outer agentic loop, the agent writes Python scripts that invoke tools as ordinary functions, manipulate data, and perform calculations directly in Python.

## Core Principle: Minimize Agentic Loop Turns

The primary goal of CodeAct is to **minimize expensive LLM round-trips** and **eliminate arithmetic/reasoning errors** by delegating logic to the Python interpreter:

1. **Zero Discrete Outer Tool Calls**: External tools (e.g. `web_search`, `fetch_webpage`, file readers, database queries) MUST NOT be invoked as top-level agent tool calls. They are exposed as plain Python functions inside the `python_interpreter` environment.
2. **Unified Execution in Minimal Turns**: Rather than taking multiple turns for searches and computations (Turn 1: search bridge → Turn 2: search speed → Turn 3: compute → Turn 4: answer), write a unified Python script that searches, extracts values, computes the result, and terminates with `final_answer(...)` in a single execution.
3. **Deterministic Math & Data Handling**: Never calculate numbers in LLM prose. Use Python's built-in math and string processing.

## How It Works

### 1. Setup & Environment

- The only action tool the agent uses in the outer loop is `python_interpreter`.
- All sibling tools registered in the profile (e.g. `web_search`) are automatically bound into the Python sandbox namespace as ordinary callables.
- Standard safe libraries (`math`, `json`, `datetime`, `itertools`, `re`, `collections`, `statistics`, `random`, etc.) are pre-imported or importable.
- State and variables persist across multiple code execution turns in the same session.

### 2. The Execution Protocol

1. **Analyze the task**: Identify all data points, queries, and calculations needed.
2. **Write a complete Python script**:
   - Call sibling tools as functions (e.g. `res = web_search("query")`).
   - Print observations with `print(...)` to log progress.
   - Parse and process data using Python logic, regex, or string manipulation.
   - Perform all arithmetic and conversions in Python.
   - Conclude by calling `final_answer(result)`.
3. **Execute via `python_interpreter`**: Run the code block in the sandbox.
4. **Evaluate output**:
   - If `final_answer(...)` was called, the execution terminates and returns the result.
   - If an exception occurs, read the traceback, correct the script, and re-execute.

## Comprehensive Example: Multi-Step Retrieval & Calculation

**Task**: *"How many seconds would it take for a leopard at full speed to run through Pont des Arts?"*

**Agent Action (Turn 1 — Single `python_interpreter` call)**:
```python
# 1. Retrieve information via in-interpreter web search
bridge_search = web_search("Pont des Arts Paris length meters Wikipedia")
print("--- Bridge Search ---")
print(bridge_search[:400])

leopard_search = web_search("leopard top running speed km/h")
print("\n--- Leopard Search ---")
print(leopard_search[:400])

# 2. Extract key parameters from search findings
# Pont des Arts length is 155 meters
bridge_length_m = 155.0

# Leopard top speed in short bursts is ~58 km/h (range 58–60 km/h)
speed_kmh = 58.0
speed_ms = speed_kmh / 3.6  # convert km/h to m/s (16.11 m/s)

# 3. Compute time
time_seconds = bridge_length_m / speed_ms
time_seconds_upper = bridge_length_m / (60.0 / 3.6)

print(f"\nCalculation: {bridge_length_m}m / ({speed_kmh} km/h = {speed_ms:.2f} m/s) = {time_seconds:.2f}s")
print(f"Upper bound (60 km/h): {time_seconds_upper:.2f}s")

# 4. Conclude with final_answer()
final_answer(
    {
        "bridge_length_m": bridge_length_m,
        "leopard_speed_kmh": speed_kmh,
        "time_seconds": round(time_seconds, 2),
        "summary": f"A leopard running at top speed (58–60 km/h ≈ 16.1 m/s) would take approximately {time_seconds:.1f} seconds (about 9.3 to 9.6 s) to cross the 155-meter Pont des Arts.",
    }
)
```

**Observation**:
```
--- Bridge Search ---
Search results for 'Pont des Arts Paris length meters Wikipedia':
1. Pont des Arts - Wikipedia
   Total length: 155 m (508.5 ft)...

--- Leopard Search ---
Search results for 'leopard top running speed km/h':
1. How Fast Can a Leopard Run?
   Leopards can run up to 58 km/h (36 mph) in short bursts...

Calculation: 155.0m / (58.0 km/h = 16.11 m/s) = 9.62s
Upper bound (60 km/h): 9.30s

FINAL ANSWER:
{'bridge_length_m': 155.0, 'leopard_speed_kmh': 58.0, 'time_seconds': 9.62, 'summary': 'A leopard running at top speed (58–60 km/h ≈ 16.1 m/s) would take approximately 9.6 seconds (about 9.3 to 9.6 s) to cross the 155-meter Pont des Arts.'}
```

Task completed in **1 single turn**.

---

## Rules & Anti-Patterns

| Correct Pattern (CodeAct) | Anti-Pattern to Avoid |
|---|---|
| Call `web_search("...")` inside Python code in `python_interpreter` | ❌ Calling `web_search` as an outer agent tool call |
| Batch queries and calculations in a unified script | ❌ Taking 4–5 separate turns for individual lookups |
| Perform all math, unit conversions, and rounding in Python | ❌ Doing mental arithmetic or calculating in prose tokens |
| Terminate with `final_answer(result)` | ❌ Outputting prose answers without calling `final_answer` |
| Inspect tracebacks on error and fix code in the next block | ❌ Giving up or guessing when an error occurs |

---

## Configuration & Profile Setup

### Standalone CodeAct Profile

```yaml
agents:
  codeact:
    harness: langchain
    type: deep
    name: "CodeAct"
    description: "Solves tasks by writing Python code in a sandbox; tools are in-process functions"
    tools:
      - genai_tk.agents.tools.python_executor.tool.create_python_executor_tools:
          tools:
            - genai_tk.agents.tools.langchain.search_tools_factory.create_search_tool
    skill_directories:
      - ${paths.project}/skills/runtime
      - ${paths.project}/skills/custom
    system_prompt: |
      You are a CodeAct agent. You solve tasks EXCLUSIVELY by writing Python code
      and executing it via the `python_interpreter` tool.

      CRITICAL:
      1. Your ONLY tool is `python_interpreter`. Never invoke external tools (like web_search) directly from the outer agent loop.
      2. External tools (e.g. `web_search`) are exposed as callable functions inside the Python environment: invoke them in code as `web_search("query")`.
      3. Minimize turns: Batch data retrieval, parsing, and arithmetic into a single unified Python script whenever possible.
      4. Use `print()` to record intermediate observations.
      5. Conclude by calling `final_answer(result)` with your final answer.
```

### Subagent with CodeAct Only

```yaml
agents:
  my_orchestrator:
    harness: langchain
    type: deep
    name: "Orchestrator"
    subagents:
      - name: codeact
        description: "CodeAct subagent for code-based reasoning"
        tools:
          - genai_tk.agents.tools.python_executor.tool.create_python_executor_tools:
              tools:
                - genai_tk.agents.tools.langchain.search_tools_factory.create_search_tool
        system_prompt: |
          Solve tasks via Python code executed in python_interpreter.
          web_search(...) is available as a function in code.
          Print observations. Call final_answer(x) to return x.
```

---

## Supported Tools and Imports

### Built-in Tools (Always Available)

- `print(...)` – Print observations (captured as logs).
- `final_answer(value)` – Signal completion; return `value`.
- Standard library: `math`, `json`, `datetime`, `itertools`, `re`, `collections`, `statistics`, `random`, etc. (see `BASE_BUILTIN_MODULES` in the executor).

### Registered Tools (Via Interpreter Configuration)

When tools are passed to `create_python_executor_tools(tools=[...])`, they become callables in the sandbox:

```python
# In code:
results = web_search("climate change")
print(results)  # observe what came back

page = fetch_webpage("https://example.com/article")
print(page[:500])
```

### Custom Imports

Declare additional authorized imports in the profile:

```yaml
agents:
  my_agent:
    tools:
      - genai_tk.agents.tools.python_executor.tool.create_python_executor_tools:
          additional_authorized_imports:
            - pandas
            - numpy
            - requests
```

---

## Safety

The executor uses **AST-based interpretation** (not `eval`), providing:

- **Blocked operations**: No `exec()`, `__import__()`, `os.system()`, etc. in user code.
- **Sandboxed imports**: Only authorized modules can be imported.
- **Timeout enforcement**: Long-running code is interrupted (default: 30 seconds).
- **Dunder method restrictions**: Unsafe magic methods are blocked.
- **Operator count limits**: Runaway loops are halted.

---

## Error Handling and Retry

If code raises an exception:

1. **Traceback capture**: The full exception traceback is captured.
2. **Observation**: The traceback becomes the next observation, given back to the LLM.
3. **Retry mandate**: The agent is expected to revise the code and retry.

---

## See Also

- **SmolAgents**: https://github.com/agentic-ai/smolagents
- **Executor module**: `genai_tk.agents.tools.python_executor`
- **Factory function**: `genai_tk.agents.tools.python_executor.create_python_executor_tools`
- **Documentation**: `docs/codeact.md`
