---
name: codeact
description: Execute multi-step reasoning and tool use via Python code blocks in a safe sandbox. Activate when user explicitly requests to "use the codeact skill" or similar phrases.
---

# CodeAct: Code Acts as Agents

CodeAct is a SmolAgents-inspired skill that enables multi-step reasoning and tool use by composing actions as **Python code blocks** executed inside a safe sandbox. Instead of using discrete tool calls, agents write Python code that calls tools as ordinary functions and print observations to stdout.

## Design Philosophy

- **Code is the action language**: Multi-step reasoning, loops, conditionals, and stateful tool composition are all expressed naturally as Python code.
- **Tools as callables**: All registered tools (web search, file I/O, etc.) are bound as plain Python functions inside the sandbox namespace.
- **Print observations**: The agent prints intermediate results and observations, which become the next step's context.
- **Traceback-driven retry**: If code raises an exception, the traceback becomes the next observation, and the agent retries.
- **Termination via final_answer()**: The agent calls `final_answer(result)` to signal completion and return the result.

## How It Works

### 1. Setup

The skill configures a **CodeAct agent** by:
- Providing a Python executor tool that interprets code via safe AST evaluation (no `eval()`)
- Binding all sibling tools (web_search, etc.) as Python callables in the sandbox
- Injecting a `final_answer(x)` stub that signals completion

### 2. The Loop

Each turn of the agent loop:

1. **Write code**: The LLM generates Python code to solve the current task, making use of bound tools and prior state.
2. **Execute code**: The code runs in the sandbox with:
   - Print outputs captured as logs
   - Tool calls executing inside the sandbox
   - State (variables) persisting across turns
3. **Observe result**: The printed logs and/or the code's final expression value become the observation.
4. **Check for termination**: If the code called `final_answer(x)`, the run ends with result `x`. Otherwise, the loop continues.
5. **On error**: If code raises an exception, the full traceback becomes the next observation, and the agent retries.

### 3. Example Flow

**Task**: "How fast can a leopard run?"

**Turn 1 – Agent writes code**:
```python
# Search for leopard speed
result = web_search("leopard running speed")
print(f"Search results: {result}")
```

**Output**:
```
Search results for 'leopard running speed':

1. Leopard Speed
   URL: https://example.com/leopard
   Leopards can run at up to 58 km/h (36 mph) in short bursts...
```

**Turn 2 – Agent summarizes and terminates**:
```python
answer = "A leopard can run up to 45 miles per hour (72 km/h), making it one of the fastest land carnivores."
final_answer(answer)
```

**Output**:
```
FINAL ANSWER:
A leopard can run up to 45 miles per hour (72 km/h), making it one of the fastest land carnivores.
```

Agent stops. Result returned.

---

## Configuration

### Profile Structure

Use CodeAct in an agent profile by specifying the Python executor tool:

```yaml
agents:
  my_agent:
    harness: langchain
    type: react  # or deep
    name: "My CodeAct Agent"
    description: "Solves tasks by writing and executing Python code"
    tools:
      - factory: genai_tk.agents.tools.python_executor.create_python_executor_tools
    system_prompt: |
      You are a helpful assistant. Solve tasks by writing Python code.
      Available tools are bound as functions in the sandbox (e.g., web_search(...)).
      Use print() to show intermediate results.
      Call final_answer(result) when done.
```

### Subagent with CodeAct Only

For a dedicated CodeAct-only subagent:

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
          - factory: genai_tk.agents.tools.python_executor.create_python_executor_tools
          - factory: genai_tk.agents.tools.langchain.search_tools_factory.create_search_function
        system_prompt: |
          Solve tasks via Python code. Print observations. Call final_answer(x) to return x.
```

---

## Supported Tools and Imports

### Built-in Tools (Always Available)

- `print(...)` – Print observations (captured as logs).
- `final_answer(value)` – Signal completion; return `value`.
- Standard library: `math`, `json`, `datetime`, `itertools`, `re`, `collections`, `statistics`, `random`, etc. (see `BASE_BUILTIN_MODULES` in the executor).

### Registered Tools (Via Binding)

When tools are registered in the profile, they become callables in the sandbox:

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
      - factory: genai_tk.agents.tools.python_executor.create_python_executor_tools
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

1. **Traceback capture**: The full exception traceback (without sensitive internal details) is captured.
2. **Observation**: The traceback becomes the next observation, given back to the LLM.
3. **Retry mandate**: The agent is expected to revise the code and retry.

Example:

**Agent code (Turn 1)**:
```python
result = web_search("leopard speed")
print(f"Top speed: {result[0]['speed']} km/h")  # Wrong: result is a string, not a list
```

**Error**:
```
Error: Code execution failed at line 'print(f"Top speed: {result[0]['speed']} km/h")' due to TypeError: string indices must be integers
```

**Agent code (Turn 2)** — revised after reading the traceback:
```python
result = web_search("leopard speed")
print(result)  # Inspect the actual structure first
speed = "Leopards can reach 58 km/h in short bursts."
final_answer(speed)
```

---

## Tips and Best Practices

1. **Leverage state persistence**: Variables defined in one code block remain available in the next. Use this to build up complex logic incrementally.
2. **Print strategically**: Print intermediate results to see what's happening. The LLM reads these outputs.
3. **Error messages are feedback**: When your code fails, read the traceback carefully and fix it in the next block.
4. **Call final_answer() only once**: Once the agent calls `final_answer(x)`, the run ends.
5. **Use loops and conditionals**: Python's control flow (for, while, if/else) works naturally for iterative refinement.

---

## See Also

- **SmolAgents**: https://github.com/agentic-ai/smolagents  (system prompt : https://github.com/huggingface/smolagents/blob/30bb1161095dbae2271e6bc3cc4c219cc3897a57/src/smolagents/prompts/code_agent.yaml)
- **Executor module**: `genai_tk.agents.tools.python_executor`
- **Factory function**: `genai_tk.agents.tools.python_executor.create_python_executor_tools`
