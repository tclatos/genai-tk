# LangChain Agents Configuration

This directory contains LangChain agent profiles organized into logical YAML files.

## File Structure

Configuration is split into focused YAML files for better organization:

### **defaults.yaml**
Global configuration defaults and settings inherited by all profiles:
- `default_profile` - Profile key to use when none specified (e.g., "simple")
- `defaults` section - Default agent type, middleware, checkpointer, backend
- Skill directory setup
- LLM and tool defaults

### **simple.yaml**  
Simple React agents for demonstrations and basic use cases:
- `simple` - Basic agent with web search
- `filesystem` - File system manipulation with web search
- `eval_test` - Test agent (minimal config for fast execution)
- `weather` - Weather assistant

### **text2sql.yaml**
Specialized agents for database queries:
- `text2sql` - Deep Agent with planning and schema exploration for natural language to SQL
- `chinook` - Lightweight React Agent for quick SQL queries

### **browser.yaml**
Advanced Deep Agents for web browser automation:
- `browser_agent` - Sandbox-based browser automation (Docker container)
- `browser_agent_direct` - Host-local Playwright for anti-bot resilience

### **deep.yaml**
Advanced Deep Agents with planning and file system capabilities:
- `research` - Expert researcher with web search and report writing
- `coding` - Python developer with planning and file system tools
- `data_analysis` - Data analyst with visualization capabilities
- `web_research` - Web researcher with browsing and multi-source analysis
- `documentation_writer` - Technical documentation specialist
- `stock_analysis` - Financial analyst with market data tools

## Profile Structure

Each profile uses **dict-keyed structure** where the key is the identifier used in CLI commands:

```yaml
langchain_agents:
  research:              # ← Profile KEY (used in: -p research)
    name: "Research"     # ← Display name (shown in --list)
    type: deep
    llm: gpt_41@openai
    tools:
      - web_search
```

**Important**: Use the KEY (lowercase) in CLI commands:
- ✅ `cli agents langchain -p research`
- ❌ `cli agents langchain -p Research`

## Usage

### CLI Commands

List all available profiles:
```bash
cli agents langchain --list
```

Run a specific profile:
```bash
cli agents langchain -p simple "Your question here"
cli agents langchain -p browser_agent "Navigate to example.com"
cli agents langchain -p text2sql "What are the best selling groups?"
```

Use the default profile (set in defaults.yaml):
```bash
cli agents langchain "Your question here"
```

Interactive chat with a specific profile:
```bash
cli agents langchain -p research --chat
```

### Configuration Loading

The config loader automatically discovers and merges YAML files from this directory (sorted alphabetically). All profiles are combined into a single agent configuration.

## Adding New Profiles

To add a new profile:

1. Choose which file best matches your agent's purpose (or create a new file)
2. Add the profile following the dict-keyed structure:
   ```yaml
   langchain_agents:
     my_profile:          # ← Choose a key (lowercase)
       name: "My Profile" # ← Display name
       type: react
       llm: gpt_4o@openai
       tools: []
   ```
3. The config system will automatically pick it up on next run

## Profile Types

- **react**: Lightweight agents without planning (best for single-step tasks)
- **deep**: Advanced agents with planning, file system, skills (best for multi-step tasks)
- **custom**: Fully custom agent types (advanced usage)

## Profile Fields

Each profile can include:
- `name`: Display name for CLI and web UI
- `type`: Agent type (react, deep, custom)
- `llm`: LLM identifier (null uses global default)
- `tools`: List of tool specifications
- `mcp_servers`: Model Context Protocol servers
- `skill_directories`: For deep agents with skills
- `enable_planning`: Enable/disable planning tool
- `enable_file_system`: Enable/disable file system tools
- `middlewares`: Middleware pipeline
- `checkpointer`: State persistence config
- `backend`: Execution backend (e.g., aio_sandbox)
- `system_prompt`: Agent instructions

## See Also

- [Agent Configuration Documentation](../../docs/agents.md)
- [Configuration Guide](../../docs/configuration.md)


- Use **simple agents** (react) for quick demonstrations and testing
- Use **deep agents** for complex, multi-step workflows
- Browser agents require MCP servers (playwright) to be configured
- Text2SQL requires the Chinook database to be present
- All profiles inherit defaults, so only override what's needed
