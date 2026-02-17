# Deer-flow Configuration Improvements

This document summarizes the improvements made to the deer-flow agent configuration system.

## Changes Implemented

### 1. Configuration File Consolidation ✅
- **Merged** `deerflow_config.yaml` into `deerflow.yaml`
- **Removed** separate config file references from `app_conf.yaml`
- **Result**: Single source of truth for deer-flow configuration

### 2. Skill Directory System ✅
- **Replaced** explicit skill lists with `skill_directories` field
- **Added** recursive skill discovery from configured directories
- **Example**: `${paths.project}/skills` auto-discovers all skills in subdirectories
- **Result**: 15 skills automatically discovered from `public/` and `custom/` subdirectories

### 3. Default Profile Configuration ✅
- **Added** `default_profile` key in `deerflow.yaml` using YAML anchor
- **Configuration**:
  ```yaml
  deerflow:
    default_profile: &default_profile "Research Assistant"
  ```
- **CLI Behavior**: When `-p` is not specified, uses default profile automatically
- **Visual Indicator**: ⭐ displays next to default profile in `--list` output

### 4. /info Command ✅
- **Added** new `/info` command in chat mode (alongside `/help`, `/trace`, `/clear`, `/quit`)
- **Displays**:
  - Profile information (name, description, mode)
  - LLM configuration (ID, model name)
  - Tool groups enabled
  - MCP servers configured
  - Skills available
  - Skill directories configured
- **Usage**: Type `/info` during interactive chat session

### 5. Profile LLM Configuration ✅
- **Added** `llm` field to profile configuration
- **Supports**: LLM IDs (e.g., `gpt_41mini@openai`) and tags (e.g., `chat:high`)
- **Example**:
  ```yaml
  - name: Research Assistant
    llm: gpt_41mini@openai
    mode: pro
    # ... other fields
  ```
- **Priority**: Command-line override (`-m`) > Profile LLM > Global default

### 6. LLM Display Format ✅
- **Display Format**: Shows `model@provider` format (e.g., `gpt_41mini@openai`)
- **Fallback**: Shows model name when ID not available
- **Locations**: 
  - Single-shot mode header
  - Chat mode header
  - `/info` command output
  - `--list` output (when applicable)

## Configuration File Structure

### deerflow.yaml
```yaml
# Global deer-flow settings
deerflow:
  skills:
    directories:
      - "${paths.project}/skills"
  sandbox:
    enabled: true
  default_profile: &default_profile "Research Assistant"

# Profile definitions
deerflow_agents:
  - name: &default_profile Research Assistant
    description: "Advanced research assistant with web access"
    llm: gpt_41mini@openai  # Optional: LLM for this profile
    mode: pro
    tool_groups:
      - web
      - file
      - bash
    mcp_servers:
      - tavily-mcp
    skill_directories:
      - "${paths.project}/skills"
    thinking_enabled: true
    subagent_enabled: true
    # ... other profile fields
```

## CLI Usage Examples

### List Profiles (with default indicator)
```bash
uv run cli agents deerflow --list
# Output shows ⭐ next to default profile
```

### Use Default Profile
```bash
echo "What is quantum computing?" | uv run cli agents deerflow
# Automatically uses "Research Assistant" profile
```

### Use Specific Profile
```bash
uv run cli agents deerflow -p "Coder" "Write a sorting algorithm"
```

### Override Profile LLM
```bash
uv run cli agents deerflow -p "Research Assistant" -m "gpt_41_openrouter" "Research AI trends"
```

### Interactive Chat with /info
```bash
uv run cli agents deerflow -p "Research Assistant" --chat
# In chat:
# > /info  # Shows comprehensive agent configuration
# > /help  # Shows all available commands
```

## Technical Implementation

### Files Modified

**genai-tk:**
1. `/config/basic/agents/deerflow.yaml` - Merged config, added llm field
2. `/config/app_conf.yaml` - Removed deerflow_config.yaml reference
3. `/genai_tk/extra/agents/deer_flow/agent.py` - Added llm field to DeerFlowAgentConfig
4. `/genai_tk/extra/agents/deer_flow/cli_commands.py` - Added LLM selection, /info command, display formatting
5. `/genai_tk/extra/agents/deer_flow/config_bridge.py` - Added load_skills_from_directories()

**genai-blueprint:**
1. `/config/agents/deerflow.yaml` - Same changes as genai-tk
2. `/config/app_conf.yaml` - Same changes as genai-tk

### Key Functions

**load_skills_from_directories()**
- Scans configured directories recursively
- Discovers skills in `public/` and `custom/` subdirectories
- Returns list of discovered skills (e.g., `['public/deep-research', ...]`)

**_get_llm_display_name(llm, llm_id=None)**
- Formats LLM display name for user output
- Tries `llm_id` parameter first (shows `model@provider` format)
- Falls back to `llm.model` attribute when ID not available
- Used in single-shot mode, chat mode, and /info command

**_display_agent_info(profile_dict, llm, llm_id=None)**
- Shows comprehensive agent configuration
- Displays profile info, LLM details, tools, skills, etc.
- Invoked by `/info` command in chat mode

## Testing

### Verification Tests
```python
# Test 1: Profile loading with LLM
profiles = load_deer_flow_profiles()
profile = next((p for p in profiles if p.name == "Research Assistant"), None)
assert profile.llm == "gpt_41mini@openai"

# Test 2: LLM resolution
llm_id, error = LlmFactory.resolve_llm_identifier_safe(profile.llm)
assert llm_id == "gpt_41mini@openai"
assert error is None

# Test 3: Display format with resolved ID
display = _get_llm_display_name(llm, llm_id)
assert "@" in display  # Should show model@provider format
assert display == "gpt_41mini@openai"

# Test 4: Skills discovery
from genai_tk.extra.agents.deer_flow.config_bridge import load_skills_from_directories
skills = load_skills_from_directories(["${paths.project}/skills"])
assert len(skills) == 15  # Should discover all skills
```

### CLI Verification
```bash
# Test default profile usage
uv run cli agents deerflow --list  # Shows ⭐ next to "Research Assistant"

# Test LLM display format
echo "What is 2+2?" | uv run cli agents deerflow -p "Research Assistant"
# Output should show: "LLM: gpt_41mini@openai"

# Test skill directory loading  
# Logs should show: "Available public skills: chart-visualization, consulting-analysis, ..."
```

## Benefits

1. **Simpler Configuration**: Single file instead of multiple files
2. **Auto-discovery**: Skills automatically found in configured directories
3. **Better Defaults**: Default profile reduces command-line verbosity
4. **Transparency**: /info command provides full visibility into agent configuration
5. **Flexibility**: Per-profile LLM configuration with command-line override
6. **User-Friendly**: Clear display format showing model@provider notation

## Migration Notes

### Breaking Changes
- `deerflow_config.yaml` no longer used (merged into `deerflow.yaml`)
- `skills` list is still supported but `skill_directories` is preferred

### Backward Compatibility
- Existing `skills` lists still work
- Both fields can be used simultaneously (merged by agent)
- Old CLI commands continue to work unchanged

## Future Enhancements

Possible future improvements:
- Skill dependency management
- Profile inheritance (extend base profiles)
- Dynamic skill reloading without restart
- Profile validation command
- Profile export/import functionality
