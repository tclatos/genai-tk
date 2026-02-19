# Deer-Flow Agent Quick Reference

## Starting the Agent

### Interactive Chat Mode (Recommended)
```bash
# Start with default profile
cd /home/tcl/prj/genai-tk
cli agents deerflow --chat

# Use specific profile
cli agents deerflow --chat --profile "Research Assistant"

# Override LLM
cli agents deerflow --chat --llm gpt_41@openai

# Add MCP servers
cli agents deerflow --chat --mcp tavily-mcp --mcp filesystem

# Enable verbose debug logging
cli agents deerflow --chat --verbose
```

### Single-Shot Mode
```bash
# Run a single query
cli agents deerflow "What is the weather in Paris?"

# With specific profile
cli agents deerflow --profile "Research Assistant" "Summarize AI trends"

# Read from stdin
echo "What's the capital of France?" | cli agents deerflow
```

## Available Commands

### During Chat Session

- `/help` - Show available commands
- `/info` - Display agent configuration and loaded tools
- `/quit` or `/exit` - Exit chat mode
- `/clear` - Clear conversation history (new thread)
- `/trace` - Open LangSmith trace in browser

### List Available Profiles
```bash
cli agents deerflow --list
```

## Understanding the Output

### User Message
```
╭──────────── User ────────────╮
│ What's the weather in Paris? │
╰──────────────────────────────╯
```

### Tool Call
```
╭──────── Tool Call ───────╮
│ web_search               │
│                          │
│ query: weather Paris     │
╰──────────────────────────╯
```

### Tool Result
```
╭────── Tool Result: web_search ──────╮
│ Current weather in Paris:            │
│ • Temperature: 15°C                  │
│ • Conditions: Partly cloudy          │
╰──────────────────────────────────────╯
```

### Assistant Response
```
╭───────── Assistant ─────────╮
│ The weather in Paris today  │
│ is 15°C with partly cloudy  │
│ conditions.                 │
╰─────────────────────────────╯
```

## Startup Information

When the agent starts, you'll see:

```
🦌 Deer-flow Interactive Chat
Profile: Research Assistant
Mode: pro
LLM: gpt_oss120@openrouter
MCP Servers: tavily-mcp

🔧 Tool Groups: web
   Note: Tools from these groups will be loaded if dependencies are available

🎯 Available Skills: 15
   consulting-analysis, video-generation, data-analysis, podcast-generation, ...
   Note: Skills add sub-workflows but not direct tools to the agent
```

### What This Means

- **Profile**: The agent configuration being used
- **Mode**: Reasoning intensity (flash, thinking, pro, ultra)
- **LLM**: The language model powering the agent
- **MCP Servers**: Model Context Protocol servers for external tools
- **Tool Groups**: Categories of tools available (e.g., web, file, search)
- **Skills**: Sub-workflows for complex tasks

## Common Use Cases

### 1. Web Research
```bash
cli agents deerflow --chat --mcp tavily-mcp
>>> Research the latest developments in quantum computing
```

### 2. File Operations (when file tools are added)
```bash
cli agents deerflow --chat --profile "File Assistant"
>>> List files in the current directory
>>> Read the contents of config.yaml
```

### 3. Data Analysis
```bash
cli agents deerflow --chat
>>> What are the trends in AI according to recent news?
```

### 4. Custom LLM
```bash
# Use a specific model
cli agents deerflow --chat --llm claude-3-5-sonnet-20241022@anthropic

# Use a local model
cli agents deerflow --chat --llm llama3.2@ollama
```

## Debugging Tips

### Enable Verbose Logging
```bash
cli agents deerflow --chat --verbose
```
This shows DEBUG-level logs including:
- Middleware execution steps
- Tool loading process
- Internal agent state transitions

### Check Available Tools
During a chat session, use:
```
>>> /info
```
This displays:
- Current profile details
- LLM information
- Loaded tools
- Skills information

### Verify Model Configuration
```bash
# List all available models
cli info models

# Check specific model
cli info models --filter gpt
```

## Troubleshooting

### Issue: "Tool not found" error
**Cause**: Agent is trying to use a tool that isn't loaded

**Solution**: 
1. Check tool groups in profile configuration
2. Verify dependencies are installed (e.g., `pip install readabilipy` for web_fetch)
3. Use `/info` to see which tools are actually loaded

### Issue: Agent claims to create a file but doesn't
**Cause**: Agent is hallucinating file creation capability

**Expected Behavior**: Agent should now acknowledge it cannot create files and explain why

**If still occurring**: Check that the enhanced system prompt is active (look for "IMPORTANT: You have access to the following tools:" in logs)

### Issue: Too much output/debug information
**Solution**: Run without `--verbose` flag to use Rich trace middleware instead of DEBUG logs

### Issue: MCP server connection failed
**Cause**: MCP server not configured or missing dependencies

**Solution**:
1. Check `config/mcp_servers.yaml`
2. Verify required npm packages: `npm list -g tavily-mcp`
3. Check API keys in environment variables

## Configuration

### Profile Configuration
Edit: `config/agents/deerflow.yaml`

Example profile:
```yaml
profiles:
  - name: "Research Assistant"
    description: "Agent for web research and information gathering"
    tool_groups: ["web"]
    mcp_servers: ["tavily-mcp"]
    mode: "pro"
    thinking_enabled: true
    system_prompt: "You are a research assistant..."
```

### MCP Servers
Edit: `config/mcp_servers.yaml`

Example server:
```yaml
mcp_servers:
  tavily-mcp:
    enabled: true
    command: npx
    args: ["-y", "tavily-mcp"]
    env:
      TAVILY_API_KEY: ${TAVILY_API_KEY}
```

## Advanced Usage

### Chain Multiple MCP Servers
```bash
cli agents deerflow --chat \
  --mcp tavily-mcp \
  --mcp filesystem \
  --mcp github
```

### Override Multiple Settings
```bash
cli agents deerflow --chat \
  --profile "Research Assistant" \
  --llm gpt_41@openai \
  --mode ultra \
  --mcp tavily-mcp
```

### Use with Stdin for Scripting
```bash
# Process multiple queries
cat queries.txt | while read query; do
  echo "Query: $query"
  echo "$query" | cli agents deerflow --profile "Quick Answer"
done
```

## Related Documentation

- [Deer-Flow Improvements Summary](deer_flow_improvements_summary.md) - Technical details of recent improvements
- [Agents Guide](agents-guide.md) - General guide for all agent types
- [Deer-Flow Official Docs](https://docs.langchain.com/oss/python/deepagents/overview) - LangChain Deer-Flow documentation

## Getting Help

1. **Built-in help**: `cli agents deerflow --help`
2. **List profiles**: `cli agents deerflow --list`
3. **Chat commands**: Type `/help` during a chat session
4. **Check tools**: Type `/info` during a chat session
