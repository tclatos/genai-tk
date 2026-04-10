---
name: add-agent-profile
description: Step-by-step procedure to configure a new agent profile with tools and MCP servers in a genai-tk project.
---

# Add an Agent Profile

Follow these steps to add a new agent profile to a genai-tk project.

## Prerequisites
- The project was initialized with `cli init` (has `config/agents/` directory)
- Agent profiles are defined in `config/agents/langchain.yaml`

## Step 1: Edit langchain.yaml

Add a new profile to `config/agents/langchain.yaml`:

```yaml
langchain_agents:
  profiles:
    - name: MyAgent
      type: react                    # react | deep | custom
      llm: default                   # LLM identifier (from providers/llm.yaml)
      system_prompt: |
        You are a helpful assistant specialized in [domain].
        Use the provided tools to answer questions accurately.
      tools:
        - spec: web_search           # Built-in tool spec
          config:
            provider: tavily
        - factory: my_package.tools.my_tools.create_tools  # Custom tool factory
      mcp_servers: []                # Optional: MCP server names from mcp_servers.yaml
      middlewares: []                # Optional: middleware classes
      checkpointer:
        type: memory                 # memory | sqlite | postgres
```

## Step 2: Create Custom Tools (if needed)

Create a tool factory function in `<package>/tools/my_tools.py`:

```python
"""Custom tools for MyAgent."""

from langchain_core.tools import tool


@tool
def my_custom_tool(query: str) -> str:
    """Description of what this tool does — the LLM reads this docstring."""
    # Implementation here
    return f"Result for: {query}"


def create_tools() -> list:
    """Factory function referenced from agent profile YAML."""
    return [my_custom_tool]
```

## Step 3: Test the Agent

```bash
# List available profiles
uv run cli agents langchain --list

# Run in chat mode
uv run cli agents langchain --profile MyAgent --chat

# Run with a single question
uv run cli agents langchain --profile MyAgent -q "What is the weather today?"
```

## Agent Types

| Type | Description | Use for |
|------|-------------|---------|
| `react` | Standard ReAct loop (Thought → Action → Observation) | General-purpose tasks |
| `deep` | Multi-step planning with subagents and skills | Complex research/analysis |
| `custom` | LangGraph Functional API — maximum flexibility | Custom workflows |

## Tool Configuration Options

```yaml
tools:
  # Built-in tool specs (see docs/agents.md for full list)
  - spec: web_search
  - spec: python_repl
  - spec: file_search

  # Custom tool factory (returns list of LangChain tools)
  - factory: mypackage.tools.create_tools
    config:
      key: value

  # Direct LangChain tool class
  - tool_class: langchain_community.tools.wikipedia.WikipediaQueryRun
```

## Adding MCP Servers to Agent

1. Define MCP server in `config/mcp_servers.yaml`
2. Reference by name in the agent profile:

```yaml
mcp_servers:
  - name: my-server
    command: uvx
    args: ["my-mcp-server"]
    env:
      API_KEY: ${oc.env:MY_API_KEY}

# In agent profile:
profiles:
  - name: MyAgent
    mcp_servers: [my-server]
```

## DeerFlow Agents

For multi-step research agents, configure in `config/agents/deerflow.yaml`:

```yaml
deerflow_agents:
  - name: Researcher
    mode: pro              # flash | thinking | pro | ultra
    llm: default
    mcp_servers: [tavily-mcp]
    skill_directories: [${paths.project}/skills]
```

Run with: `uv run cli agents deerflow --profile Researcher --chat`
