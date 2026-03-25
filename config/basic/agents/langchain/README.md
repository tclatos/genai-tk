# LangChain Agents Configuration

This directory contains the organization of LangChain agent profiles split into logical groups.

## File Structure

This configuration has been reorganized from a single monolithic `langchain.yaml` file into separate, focused YAML files:

### **defaults.yaml**
Global configuration defaults and settings inherited by all profiles:
- Default agent type (react)
- Middleware configuration
- Checkpointer and backend defaults
- Skill directory setup
- Default profile name

### **simple.yaml**  
Simple React agents for demonstrations and basic use cases:
- **simple** - Basic agent with web search
- **filesystem** - File system manipulation with web search
- **weather** - Weather assistant

### **text2sql.yaml**
Specialized agents for database queries:
- **text2sql** - Deep Agent with planning and schema exploration for natural language to SQL conversion
- **chinook** - Lightweight React Agent for quick SQL queries against the Chinook demo database

### **browser.yaml**
Advanced Deep Agents for web browser automation:
- **Browser Agent** - Sandbox-based browser automation (Docker container)
- **Browser Agent Direct** - Host-local Playwright for anti-bot resilience

### **deep.yaml**
Advanced Deep Agents with planning and file system capabilities:
- **Research** - Expert researcher with web search and report writing
- **Coding** - Python developer with planning and file system tools
- **Data Analysis** - Data analyst with visualization capabilities
- **Web Research** - Web researcher with browsing and multi-source analysis
- **Documentation Writer** - Technical documentation specialist
- **Stock Analysis** - Financial analyst with market data tools

## Usage

### CLI Commands

List all available profiles:
```bash
uv run cli agents langchain --list
```

Run a specific profile:
```bash
uv run cli agents langchain --profile "simple" "Your question here"
uv run cli agents langchain --profile "Browser Agent" "Navigate to example.com"
uv run cli agents langchain --profile "text2sql" "What are the best selling groups?"
```

Use the default profile (simple):
```bash
uv run cli agents langchain "Your question here"
```

### Configuration Loading

The config loader automatically discovers and merges YAML files from this directory. All files are loaded and their profiles are combined into a single agent configuration.

**Important:** Ensure the `defaults.yaml` file is loaded first as it contains global settings used by other profiles.

## Adding New Profiles

To add a new profile:

1. Choose which file best matches your agent's purpose
2. Add the profile entry following the structure of existing profiles in that file
3. The config system will automatically pick it up on next run

## Profile Types

- **react**: Lightweight agents without planning (best for simple, single-step tasks)
- **deep**: Advanced agents with planning, file system access, and skill integration (best for complex, multi-step tasks)
- **custom**: Fully custom agent types (advanced usage)

## Profile Fields

Each profile can include:
- `name`: Display name for CLI usage
- `type`: Agent type (react, deep, custom)
- `description`: Short description for --list
- `llm`: LLM identifier (null uses global default)
- `system_prompt`: Agent instructions
- `pre_prompt`: Short context for react agents
- `tools`: LangChain tools and functions
- `mcp_servers`: Model Context Protocol servers
- `skill_directories`: For deep agents with skills
- `enable_planning`: Enable/disable planning tool
- `enable_file_system`: Enable/disable file system tools
- `features`: UI badges for display
- `examples`: Sample queries for documentation

## Tips

- Use **simple agents** (react) for quick demonstrations and testing
- Use **deep agents** for complex, multi-step workflows
- Browser agents require MCP servers (playwright) to be configured
- Text2SQL requires the Chinook database to be present
- All profiles inherit defaults, so only override what's needed
