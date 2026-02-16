# Deer-flow CLI Fix Summary

## Issues Fixed

### 1. **Chat Mode with Initial Input** ✅ FIXED
**Problem:** When using `--chat` flag with a positional argument, the input text was completely ignored, causing confusion.

**Example:**
```bash
cli agents deerflow -p "Web Browser" --chat -m gpt_oss120@openrouter "get news"
```

**Before:** The command entered interactive mode and ignored "get news" entirely.

**After:** The command now:
1. Processes "get news" as the first message
2. Shows the agent's response
3. Then enters interactive mode for follow-up questions

### 2. **LLM Override Verification** ✅ CONFIRMED WORKING
The LLM override (`-m` / `--llm` flag) was already working correctly. It properly:
- Resolves the model identifier (e.g., `gpt_oss120@openrouter` → `openai/gpt-oss-120b:exacto`)
- Creates the LLM instance with the specified model
- Passes it to the Deer-flow agent
- Displays the active model in the output

### 3. **Tool Groups Behavior** ℹ️ BY DESIGN
The "web_fetch" tool is loaded because the "Web Browser" profile specifies `tool_groups: ["web"]`, which loads all tools in the "web" group from Deer-flow. This is the correct behavior.

**Note:** web_fetch may fail to load if `readabilipy` is not installed, but other web tools (like web_search) will still work.

## Implementation Details

### Changes Made

1. **Modified `_run_chat_mode()` function** ([cli_commands.py](../genai_tk/extra/agents/deer_flow/cli_commands.py))
   - Added `initial_input: Optional[str] = None` parameter
   - Processes initial input before entering the interactive loop

2. **Created `_process_message()` helper function**
   - Extracted common agent execution logic
   - Used by both initial message processing and the interactive loop
   - Handles streaming and non-streaming modes

3. **Updated both CLI entry points**
   - Main entry point in `cli_commands.py`
   - Agents command wrapper in `commands_agents.py`

### Code Flow

```python
# New flow with --chat and initial input:
if chat:
    asyncio.run(
        _run_chat_mode(
            profile_name=profile,
            config_path=config_path,
            llm_override=llm,
            extra_mcp=mcp,
            mode_override=mode,
            stream_enabled=stream,
            initial_input=input_text,  # ← Now passed!
        )
    )
```

Inside `_run_chat_mode()`:
```python
# Process initial input if provided
if initial_input:
    await _process_message(
        user_input=initial_input,
        agent=agent,
        thread_id=thread_id,
        stream_enabled=stream_enabled,
        console=console,
    )

# Then enter interactive loop
while True:
    user_input = await session.prompt_async(">>> ")
    # ... handle commands and process messages
```

## Usage Examples

### Single-Shot Mode (no changes)
```bash
# Process one query and exit
cli agents deerflow -p "Web Browser" -m gpt_oss120@openrouter "get news"
```

### Chat Mode with Initial Message (NEW)
```bash
# Process initial query, then enter interactive mode
cli agents deerflow -p "Web Browser" --chat -m gpt_oss120@openrouter "get news"
# After response, you can continue asking questions:
>>> tell me more
>>> /quit
```

### Chat Mode without Initial Message (unchanged)
```bash
# Enter interactive mode immediately
cli agents deerflow -p "Web Browser" --chat -m gpt_oss120@openrouter
>>> get news
>>> /quit
```

## Testing Results

✅ **Single-shot mode:** Working correctly
```bash
$ cli agents deerflow -p "Web Browser" -m gpt_oss120@openrouter "What is 2+2?"
Profile: Web Browser
LLM: openai/gpt-oss-120b:exacto
# ... (processing)
╭─────────────────────────── 🦌 Deer-flow Response ────────────────────────────╮
│  2 + 2 = 4.                                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

✅ **Chat mode with initial input:** Working correctly
```bash
$ cli agents deerflow -p "Web Browser" --chat -m gpt_oss120@openrouter "What is 3+3?"
Profile: Web Browser
LLM: openai/gpt-oss-120b:exacto
╭──────────────────────────────────── User ────────────────────────────────────╮
│ What is 3+3?                                                                 │
╰──────────────────────────────────────────────────────────────────────────────╯
# ... (processing)
╭────────────────────────────────  Assistant  ─────────────────────────────────╮
│  3 + 3 = 6.                                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
>>> _  # ← Now in interactive mode
```

## Files Modified

1. `/home/tcl/prj/genai-tk/genai_tk/extra/agents/deer_flow/cli_commands.py`
   - Added `_process_message()` helper function
   - Modified `_run_chat_mode()` to accept and process initial input
   - Updated imports to include `Any`

2. `/home/tcl/prj/genai-tk/genai_tk/extra/agents/commands_agents.py`
   - Updated `deerflow` command to pass `initial_input` parameter

## Breaking Changes

None. The changes are backward compatible:
- Existing commands without `--chat` work exactly as before
- Existing commands with `--chat` but no input work as before
- New behavior only applies when both `--chat` and input text are provided
