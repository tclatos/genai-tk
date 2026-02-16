# Deer-flow Error Handling Improvements

## Overview

This document describes the error handling improvements made to the deer-flow agent system. These improvements provide clear, helpful error messages when users provide invalid configurations, and they work consistently across both CLI and Streamlit UI.

## Changes Made

### 1. Custom Exception Classes

Added four custom exception classes in [agent.py](../genai_tk/extra/agents/deer_flow/agent.py):

```python
class DeerFlowError(Exception):
    """Base exception for Deer-flow agent errors."""
    
class ProfileNotFoundError(DeerFlowError):
    """Raised when a profile name is not found in configuration."""
    # Includes profile_name and available_profiles attributes
    
class InvalidModeError(DeerFlowError):
    """Raised when an invalid mode is specified."""
    # Includes mode and valid_modes attributes
    
class MCPServerNotFoundError(DeerFlowError):
    """Raised when MCP server names are not found in configuration."""
    # Includes invalid_servers and available_servers attributes
```

Each exception provides:
- Detailed context about what went wrong
- List of valid options to help users fix the issue
- Structured attributes for programmatic access

### 2. Validation Functions

Added reusable validation functions in [agent.py](../genai_tk/extra/agents/deer_flow/agent.py):

#### `validate_profile_name(profile_name, profiles)`
- Validates profile name against available profiles
- Case-insensitive matching
- Raises `ProfileNotFoundError` with list of available profiles

#### `validate_mode(mode)`
- Validates agent mode (flash, thinking, pro, ultra)
- Case-insensitive
- Raises `InvalidModeError` with list of valid modes

#### `validate_mcp_servers(server_names)`
- Validates MCP server names against configured servers
- Checks all servers and reports all invalid ones
- Raises `MCPServerNotFoundError` with invalid and available servers

#### `validate_llm_identifier(llm_id)`
- Validates LLM identifier format (provider/model)
- Attempts to create LLM to verify configuration
- Raises `ValueError` with helpful error message

### 3. Helper Functions

Added helper functions to get available options:

```python
get_available_profile_names(profiles) -> list[str]
get_available_modes() -> list[str]
get_available_mcp_servers() -> list[str]
```

### 4. Improved Config Loading

Enhanced `load_deer_flow_profiles()` function:

```python
def load_deer_flow_profiles(config_path: str | None = None) -> list[DeerFlowAgentConfig]:
    """Load Deer-flow profiles from YAML config.
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid or empty
    """
```

Now includes:
- Proper YAML error handling with clear messages
- Validation that config contains required sections
- Helpful error messages pointing to fix

### 5. CLI Error Handling

Updated [cli_commands.py](../genai_tk/extra/agents/deer_flow/cli_commands.py) to use validation:

**Both `_run_single_shot()` and `_run_chat_mode()` now:**

```python
# Load and validate profile
try:
    profiles = load_deer_flow_profiles(config_path)
    profile_dict = validate_profile_name(profile_name, profiles)
except DeerFlowError as e:
    console.print(f"[red]Error:[/red] {e}", style="bold")
    raise typer.Exit(1) from e

# Validate overrides
try:
    if mode_override:
        validated_mode = validate_mode(mode_override)
        profile_dict.mode = validated_mode
    
    if extra_mcp:
        validated_mcp = validate_mcp_servers(extra_mcp)
        profile_dict.mcp_servers = list(set(profile_dict.mcp_servers + validated_mcp))
except DeerFlowError as e:
    console.print(f"[red]Error:[/red] {e}", style="bold")
    raise typer.Exit(1) from e
```

### 6. Streamlit Error Handling

Updated [deer_flow_agent.py](../../genai-blueprint/genai_blueprint/webapp/pages/demos/deer_flow_agent.py):

**Profile Loading:**
```python
try:
    from genai_tk.extra.agents.deer_flow.agent import DeerFlowError, load_deer_flow_profiles
    profiles = load_deer_flow_profiles(CONFIG_FILE)
    # ... convert to dicts ...
except (DeerFlowError, FileNotFoundError, ValueError) as e:
    logger.error(f"Failed to load profiles: {e}")
    st.error(f"❌ Error loading Deer-flow profiles: {e}")
    return []
```

**Agent Creation:**
```python
def create_agent_for_profile(profile_dict: dict[str, Any]) -> Any:
    # Validate MCP servers if specified
    mcp_servers = profile_dict.get("mcp_servers", [])
    if mcp_servers:
        try:
            validated_mcp = validate_mcp_servers(mcp_servers)
            profile_dict["mcp_servers"] = validated_mcp
        except Exception as e:
            raise ValueError(f"Invalid MCP servers in profile '{profile_dict['name']}': {e}") from e
    # ... create agent ...
```

## Error Message Examples

### Invalid Profile Name

**CLI:**
```
Error: Profile 'Researcher' not found. Available profiles: Research Assistant, Coder, Web Browser, Chinook DB + Research, Weather & Files, Visual Explainer
```

**Streamlit:**
Displays same message in red error box

### Invalid Mode

**CLI:**
```
Error: Invalid mode 'turbo'. Valid modes: flash, thinking, pro, ultra
```

### Invalid MCP Servers

**CLI:**
```
Error: MCP server(s) not found: invalid1, invalid2. Available servers: math, weather, tech_news
```

### Missing Config File

**CLI/Streamlit:**
```
Error: Deer-flow config not found at /path/to/deerflow.yaml. Please create the config file or specify a valid path.
```

### Invalid YAML

**CLI/Streamlit:**
```
Error: Invalid YAML in /path/to/deerflow.yaml: while scanning a simple key
  in "<unicode string>", line 3, column 1
found unexpected ':'
  in "<unicode string>", line 4, column 5
```

## Testing

### Unit Tests

Created comprehensive test suite in [test_validation.py](../tests/unit_tests/extra/agents/deer_flow/test_validation.py):

```bash
cd /home/tcl/prj/genai-tk
uv run pytest tests/unit_tests/extra/agents/deer_flow/test_validation.py -v
```

**Tests cover:**
- ✅ Profile name validation (success and failure cases)
- ✅ Mode validation (success, failure, case-insensitivity)
- ✅ MCP server validation (empty, valid, invalid)
- ✅ Helper functions (available profiles, modes, servers)
- ✅ Error message content and structure

### Demo Script

Created [deer_flow_validation_demo.py](../examples/deer_flow_validation_demo.py) to demonstrate validation:

```bash
cd /home/tcl/prj/genai-tk
uv run python examples/deer_flow_validation_demo.py
```

## Benefits

### 1. User Experience
- **Clear error messages** - Users know exactly what went wrong
- **Actionable guidance** - Error messages show valid options
- **Consistent behavior** - Same validation in CLI and Streamlit

### 2. Developer Experience
- **Reusable validation** - Functions work in any context
- **Type-safe errors** - Custom exceptions with structured data
- **Easy testing** - Validation functions are independently testable

### 3. Maintainability
- **Single source of truth** - Validation logic in one place
- **Extensible** - Easy to add new validation functions
- **Well-documented** - Clear docstrings with Raises sections

## Usage Examples

### CLI

```bash
# Invalid profile
cli agents deerflow -p "Researcher" "Tell me about AI"
# Error: Profile 'Researcher' not found. Available profiles: Research Assistant, Coder, ...

# Invalid mode
cli agents deerflow -p "Research Assistant" --mode turbo "Tell me about AI"
# Error: Invalid mode 'turbo'. Valid modes: flash, thinking, pro, ultra

# Invalid MCP server
cli agents deerflow -p "Research Assistant" --mcp invalid_server "Tell me about AI"
# Error: MCP server(s) not found: invalid_server. Available servers: math, weather, ...

# List available profiles
cli agents deerflow --list
```

### Streamlit

- Invalid profiles are caught during config loading and shown in error box
- Invalid MCP servers in profile config are caught during agent creation
- Users can only select from valid profiles via dropdown (prevents invalid selection)

### Python API

```python
from genai_tk.extra.agents.deer_flow.agent import (
    load_deer_flow_profiles,
    validate_profile_name,
    validate_mode,
    ProfileNotFoundError,
)

# Load profiles with error handling
try:
    profiles = load_deer_flow_profiles()
except FileNotFoundError as e:
    print(f"Config file missing: {e}")
    exit(1)
except ValueError as e:
    print(f"Invalid config: {e}")
    exit(1)

# Validate user input
try:
    profile = validate_profile_name(user_input, profiles)
    mode = validate_mode(user_mode)
except ProfileNotFoundError as e:
    print(f"Profile error: {e}")
    print(f"Available: {e.available_profiles}")
except InvalidModeError as e:
    print(f"Mode error: {e}")
    print(f"Valid modes: {e.valid_modes}")
```

## Migration Notes

**No breaking changes!**

- Existing code continues to work
- New validation is opt-in (used in CLI and Streamlit)
- Error messages are more helpful but same error types (for config loading)

## Future Enhancements

Potential improvements:

1. **LLM validation** - More comprehensive LLM identifier validation with provider-specific help
2. **Tool validation** - Validate tool configurations before agent creation
3. **Profile schema validation** - Use JSON schema to validate profile structure
4. **Suggestion engine** - Fuzzy matching for profile/mode names ("Did you mean...?")
5. **Configuration wizard** - Interactive CLI/UI for creating profiles

## Files Modified

### genai-tk
- `genai_tk/extra/agents/deer_flow/agent.py` - Added exceptions, validation functions
- `genai_tk/extra/agents/deer_flow/cli_commands.py` - Updated to use validation
- `tests/unit_tests/extra/agents/deer_flow/test_validation.py` - New test file
- `examples/deer_flow_validation_demo.py` - New demo script

### genai-blueprint
- `genai_blueprint/webapp/pages/demos/deer_flow_agent.py` - Updated error handling

## Summary

This improvement adds robust, user-friendly error handling to the deer-flow agent system:

✅ **Custom exception classes** for specific error types  
✅ **Reusable validation functions** for profiles, modes, MCP servers  
✅ **Improved CLI error handling** with clear messages  
✅ **Enhanced Streamlit error handling** with graceful degradation  
✅ **Comprehensive tests** covering all validation scenarios  
✅ **Demo script** showing validation in action  

Users now get **clear, actionable error messages** instead of confusing stack traces!
