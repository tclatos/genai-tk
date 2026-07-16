---
name: add-tool
description: Step-by-step procedure to create a new LangChain tool in a genai-tk project and register it in an agent profile.
tags: [tools, langchain, agents]
version: "1.0"
---

# Add a LangChain Tool

Follow these steps to add a new tool that agents can call.

## Prerequisites

- Project initialized with `cli init`
- Agent profile configured in `config/agents/langchain.yaml`

## Step 1: Create the Tool File

Create `<package>/tools/my_tool.py`:

```python
"""My custom tool for <purpose>."""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from pydantic import Field


# Option A: Simple function tool
@tool
def my_tool(input: str) -> str:
    """One-line description visible to the agent. Be specific about what inputs it expects."""
    # Your implementation here
    return f"Result: {input}"


# Option B: Class-based tool (for more control, e.g. injected dependencies)
class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "Detailed description. Include expected input format."
    api_key: str = Field(default="", exclude=True)

    def _run(self, input: str) -> str:
        return f"Result: {input}"

    async def _arun(self, input: str) -> str:
        return self._run(input)


# Always provide a factory function for agent profiles to reference
def create_my_tools(api_key: str = "") -> list[BaseTool]:
    """Factory function referenced in agent profile YAML."""
    return [MyTool(api_key=api_key)]
```

## Step 2: Register in Agent Profile

In your project's `config/agents/*.yaml` (or `config/examples/agents/*.yaml` in
genai-tk itself), add to your profile's `tools:` list under the unified
`agents:` dict:

```yaml
agents:
  my_agent:
    harness: langchain
    type: react                # or deep
    tools:
      # Option A: bare function reference
      - function: my_project.tools.my_tool.my_tool

      # Option B: factory function — any extra keys become factory kwargs
      - factory: my_project.tools.my_tool.create_my_tools
        api_key: ${oc.env:MY_API_KEY,}
```

See `genai_tk/agents/tools/tool_specs.py` for the full `class:` / `function:` /
`factory:` discriminated spec format.

## Step 3: Test the Tool Directly

```python
from my_project.tools.my_tool import my_tool
result = my_tool.invoke("test input")
print(result)
```

## Step 4: Test via CLI

```bash
cli agents run my_agent "Use my_tool to process: hello world"
```

## Best Practices

- **Tool description** is what the agent reads to decide when to use the tool — be precise
- **Validate inputs at the boundary** — use Pydantic field validators
- **Never expose raw credentials** — use `oc.env:VAR` in config, inject via factory kwargs
- **Return structured results as strings** — agents parse the string response
- **Async support** — implement `_arun` if the tool makes I/O calls (HTTP, DB, etc.)

## Code Map

| Concern | Path |
|---------|------|
| Tool implementation | `<package>/tools/<name>.py` |
| Agent profile | `config/agents/langchain.yaml` |
| Tool factory pattern | `genai_tk/agents/tools/` (reference examples) |
| Tests | `tests/unit_tests/tools/test_<name>.py` |
