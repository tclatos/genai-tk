# Deer-flow Skills Management Implementation

## Overview

This document describes the skills management system implemented for deer-flow agents in GenAI Toolkit and GenAI Blueprint. The system provides:

1. **Configurable skill directories** - Skills stored in your project, not just deer-flow
2. **Access to deer-flow skills** - Symbolic links to all public skills from deer-flow  
3. **Custom skill support** - Create your own skills easily
4. **Skill loading tracing** - Logs show which skills are available and enabled

## Directory Structure

Both `genai-tk` and `genai-blueprint` now have identical skill directory structures:

```
project-root/
├── skills/
│   ├── public/       # Symlinks to deer-flow public skills
│   │   ├── chart-visualization -> /home/tcl/ext_prj/deer-flow/skills/public/chart-visualization
│   │   ├── data-analysis -> /home/tcl/ext_prj/deer-flow/skills/public/data-analysis
│   │   ├── deep-research -> /home/tcl/ext_prj/deer-flow/skills/public/deep-research
│   │   ├── github-deep-research -> ...
│   │   ├── skill-creator -> ...
│   │   └── ... (15 total public skills)
│   └── custom/       # Your custom skills
│       └── README.md # Guide for creating custom skills
```

## Configuration

### Configuration Files

**genai-tk: `config/basic/agents/deerflow_config.yaml`**
**genai-blueprint: `config/agents/deerflow_config.yaml`**

```yaml
deerflow:
  skills:
    # Path to skills directory (absolute or relative to project root)
    path: ${paths.project}/skills
    
    # Container mount path (for Docker sandbox)
    container_path: /mnt/skills
    
    # Enable skill loading tracing (logs available skills)
    trace_loading: true
  
  sandbox:
    provider: local  # or docker
```

These config files are automatically merged into `app_conf.yaml`.

### Skills Path Resolution

The system resolves skills paths in this order:
1. `deerflow.skills.path` from configuration
2. Expands `~` and environment variables
3. Resolves to absolute path
4. Falls back to `${project_root}/skills` if not configured

## Available Skills

### Public Skills (from deer-flow)

All symlinked from `/home/tcl/ext_prj/deer-flow/skills/public/`:

- **chart-visualization** - Create charts and visualizations  
- **consulting-analysis** - Business consulting workflows
- **data-analysis** - Analyze datasets and generate insights
- **deep-research** - Comprehensive research with citations
- **find-skills** - Search and discover available skills
- **frontend-design** - Frontend development assistance
- **github-deep-research** - Analyze GitHub repositories
- **image-generation** - Generate images
- **podcast-generation** - Create podcast content
- **ppt-generation** - Generate PowerPoint presentations
- **skill-creator** - Guide for creating new skills ✨
- **surprise-me** - Random interesting facts
- **vercel-deploy-claimable** - Vercel deployment automation
- **video-generation** - Generate videos
- **web-design-guidelines** - Web design best practices

### Creating Custom Skills

See `skills/custom/README.md` for detailed instructions.

**Quick Start:**

1. **Use the skill-creator skill** (easiest):
   ```bash
   uv run cli agents deerflow -p "Visual Explainer" --chat
   # Then: "I want to create a skill for analyzing code quality"
   ```

2. **Manual creation**:
   ```bash
   cd skills/custom
   mkdir my-skill
   cd my-skill
   ```
   
   Create `SKILL.md`:
   ```markdown
   ---
   name: my-skill
   description: Brief description
   ---
   
   # My Skill
   
   Instructions for the agent...
   ```

3. **Enable in profile** (`config/agents/deerflow.yaml`):
   ```yaml
   deerflow_agents:
     - name: "My Profile"
       skills:
         - custom/my-skill  # custom/ prefix for custom skills
         - deep-research    # public skills don't need prefix
   ```

## Skill Loading and Tracing

### Logs Output

When creating an agent, you'll see logs like:

```
INFO - Deer-flow skills path: /home/tcl/prj/genai-tk/skills
INFO - Available public skills (15): chart-visualization, consulting-analysis, data-analysis, deep-research, find-skills...
INFO - Available custom skills: my-skill
INFO - Enabling skills: deep-research, data-analysis, chart-visualization
INFO - Wrote Deer-flow config to /home/tcl/ext_prj/deer-flow/config.yaml
```

### Controlling Tracing

Set `trace_loading: false` in `deerflow_config.yaml` to disable skill loading logs.

## Profile Configuration

Skills are configured per profile in `config/agents/deerflow.yaml`:

```yaml
deerflow_agents:
  - name: "Research Assistant"
    description: "Web research with deep reasoning"
    mode: "pro"
    tool_groups:
      - web
      - file
      - bash
    mcp_servers:
      - tavily-mcp
    skills:
      - deep-research          # Public skill (no prefix needed)
      - data-analysis          # Another public skill
      - chart-visualization     # Yet another public skill  
      - custom/my-skill        # Custom skill (needs custom/ prefix)
    features:
      - "🌐 Web Search"
      - "📊 Data Analysis"
    examples:
      - "Research quantum computing developments"
```

**Skill Format:**
- `skill-name` → `public/skill-name` (public skills)
- `custom/skill-name` → `custom/skill-name` (custom skills)
- `category/skill-name` → `category/skill-name` (any category)

## Technical Implementation

### File Locations

**genai-tk:**
- Configuration: `config/basic/agents/deerflow_config.yaml`
- Skills directory: `skills/` (created by setup)
- Config bridge: `genai_tk/extra/agents/deer_flow/config_bridge.py`
- Agent creation: `genai_tk/extra/agents/deer_flow/agent.py`

**genai-blueprint:**
- Configuration: `config/agents/deerflow_config.yaml`
- Skills directory: `skills/` (created by setup)
- Streamlit UI: `genai_blueprint/webapp/pages/demos/deer_flow_agent.py`

### Key Changes

**1. Config Bridge (`config_bridge.py`)**

```python
# Reads skills configuration from OmegaConf
config = global_config().root
skills_path = OmegaConf.select(config, "deerflow.skills.path", default=str(Path.cwd() / "skills"))
trace_loading = OmegaConf.select(config, "deerflow.skills.trace_loading", default=True)

# Logs available skills
if trace_loading:
    logger.info(f"Deer-flow skills path: {skills_path}")
    # Lists public and custom skills found

# Writes config with skills path
config = {
    "skills": {
        "path": str(skills_path),
        "container_path": skills_container_path,
    },
    # ... other config
}

# Enables specific skills in extensions_config.json
extensions_config["skills_state"] = {
    "public": {
        "deep-research": True,
        "data-analysis": True,
    },
    "custom": {
        "my-skill": True,
    }
}
```

**2. Skills Configuration Merging**

`app_conf.yaml` now includes:
```yaml
:merge:
  - ${paths.config}/agents/deerflow_config.yaml
```

This makes `deerflow.skills.*` available throughout the application.

**3. Symbolic Links**

Created automatically during setup:
```bash
cd genai-tk/skills/public
ln -sf /home/tcl/ext_prj/deer-flow/skills/public/* .
```

This gives access to all deer-flow skills without copying files.

## Usage Examples

### CLI Usage

```bash
# List profiles with their configured skills
uv run cli agents deerflow --list

# Run with a profile that has skills enabled
uv run cli agents deerflow -p "Research Assistant" \
  "Research the latest AI developments and create a chart"

# Use skill-creator to create a new skill
uv run cli agents deerflow -p "Visual Explainer" --chat
# Then: "I want to create a skill for analyzing Python code"
```

### Python API Usage

```python
from genai_tk.extra.agents.deer_flow import (
    load_deer_flow_profiles,
    create_deer_flow_agent_simple,
)
from genai_tk.core import get_llm

# Load profiles (includes skills configuration)
profiles = load_deer_flow_profiles()

# Find a profile
research_profile = next(p for p in profiles if p.name == "Research Assistant")

# Check configured skills
print(f"Skills: {research_profile.skills}")
# Output: Skills: ['deep-research', 'data-analysis', 'chart-visualization']

# Create agent (skills are automatically enabled)
llm = get_llm()
agent = create_deer_flow_agent_simple(
    profile=research_profile,
    llm=llm,
)

# Use the agent
response = await agent.ainvoke(
    {"messages": [HumanMessage(content="Research quantum computing")]},
    config={"configurable": {"thread_id": "123"}}
)
```

## Testing

### Verify Skills Setup

```bash
# Check that skills directories exist
ls -la genai-tk/skills/public/
ls -la genai-tk/skills/custom/

# Check symbolic links
ls -la genai-tk/skills/public/ | grep " -> "

# Verify configuration
uv run python -c "
from genai_tk.utils.config_mngr import global_config
from omegaconf import OmegaConf
config = global_config().root
print('Skills path:', OmegaConf.select(config, 'deerflow.skills.path'))
print('Trace loading:', OmegaConf.select(config, 'deerflow.skills.trace_loading'))
"
```

### Test Skill Loading

```bash
# Run agent and check logs for skill loading
uv run cli agents deerflow -p "Research Assistant" "hello" 2>&1 | grep -E "(skills|Enabling|Available)"

# Expected output:
# INFO - Deer-flow skills path: /home/tcl/prj/genai-tk/skills
# INFO - Available public skills (15): chart-visualization, consulting-analysis, data-analysis...
# INFO - Enabling skills: deep-research, data-analysis, chart-visualization
```

### Test Custom Skill

1. Create test skill:
   ```bash
   mkdir -p genai-tk/skills/custom/test-skill
   cat > genai-tk/skills/custom/test-skill/SKILL.md << 'EOF'
   ---
   name: test-skill
   description: Test skill for verification
   ---
   
   # Test Skill
   
   This is a test skill. When the user asks about testing, respond with "Test skill loaded successfully!"
   EOF
   ```

2. Add to profile:
   ```yaml
   skills:
     - custom/test-skill
   ```

3. Test:
   ```bash
   uv run cli agents deerflow -p "Your Profile" --chat
   # Ask: "Test the skill"
   # Should mention the test skill
   ```

## Troubleshooting

### Skills Not Found

**Problem:** "Skills directory not found" warning

**Solution:**
- Check `skills/` directory exists in project root
- Verify symbolic links: `ls -la skills/public/`
- Check configuration: `deerflow.skills.path` in config

### Skills Not Loading

**Problem:** Skills configured but not being used

**Solution:**
- Check logs for "Enabling skills" message
- Verify skill names match directory names (case-sensitive)
- Check SKILL.md exists in skill directory
- Ensure skill description matches your use case

### Symlinks Broken

**Problem:** Symlinks point to non-existent files

**Solution:**
```bash
# Remove broken links
cd skills/public
find . -type l ! -exec test -e {} \; -delete

# Recreate links
ln -sf /home/tcl/ext_prj/deer-flow/skills/public/* .
```

### Skill Not Being Used

**Problem:** Skill is enabled but agent doesn't use it

**Solution:**
- Make your query match the skill's description
- Try explicitly mentioning the skill in your query
- Check that required tools/APIs are configured
- Review skill's SKILL.md to understand when it should activate

## Best Practices

1. **Use skill-creator skill** - Let the agent guide you through skill creation
2. **Keep skills focused** - One skill should do one thing well
3. **Test thoroughly** - Run multiple queries to ensure skill works as expected
4. **Document dependencies** - Note any required tools or APIs in SKILL.md
5. **Use trace_loading** - Keep enabled during development to see what's happening
6. **Organize custom skills** - Use clear, descriptive names

## Future Enhancements

Potential improvements:
- **Skill validation** - Check SKILL.md format before loading
- **Skill search** - CLI command to search for skills by description
- **Skill templates** - Pre-made templates for common skill patterns
- **Skill dependencies** - Skills that depend on other skills
- **Skill versioning** - Track skill versions and updates

## References

- [Deer-flow Documentation](https://github.com/bytedance/deer-flow)
- [Deer-flow Configuration Guide](https://github.com/bytedance/deer-flow/blob/main/backend/docs/CONFIGURATION.md)
- [Skill Creator Skill](../skills/public/skill-creator/SKILL.md)
- [Custom Skills README](../skills/custom/README.md)

## Summary

The skills management system provides:

✅ **Configurable skill directories** - Skills in your project, not just deer-flow  
✅ **Symbolic links to deer-flow skills** - All 15 public skills available
✅ **Custom skill support** - Create your own skills easily  
✅ **Skill loading tracing** - Clear logs showing what's available and enabled
✅ **Per-profile configuration** - Different skills for different use cases
✅ **skill-creator integration** - Agent can guide you through skill creation

Skills are now properly integrated, traced, and ready to use! 🎉
