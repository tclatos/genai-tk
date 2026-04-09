# Deep Agents CLI Integration

## Overview

genai-tk now provides seamless in-process integration with [deepagents-cli](https://github.com/hunterglynn/deepagents-cli), a persistent coding agent framework built on LangGraph. The integration exposes deepagents as the `cli agents deepagent` command group, using genai-tk's **LlmFactory** as the model bridge to bypass deepagents-cli's own model-creation pipeline.

**Key benefits:**
- ✅ Use any LLM configured in `config/agents/deepagent.yaml`
- ✅ Curate the TUI `/model` switcher to show only YAML-defined models
- ✅ Create reusable agent profiles with pre-configured settings
- ✅ Persistent memory, skills, and tool use across sessions
- ✅ Interactive Textual TUI or non-interactive task mode
- ✅ Integrated with genai-tk's CLI structure and config system

---

## Quick Start

### Launch the Interactive TUI

```bash
cli agents deepagent
```

This opens the Textual terminal UI with the default model. Use `/model` to switch models from the curated list in `config/agents/deepagent.yaml`.

### Run a Non-Interactive Task

```bash
cli agents deepagent task "Write a Python script that sorts a list"
```

### Use a Specific Profile

```bash
cli agents deepagent --profile coder --agent mybot
```

### List Configured Profiles

```bash
cli agents deepagent --list-profiles
```

### Resume a Thread

```bash
cli agents deepagent --resume <thread-id>
```

---

## Configuration

All deepagent settings live in `config/agents/deepagent.yaml`.

### Global Settings

```yaml
deepagent:
  default_model: null           # or "fast_model", "gpt41mini@openai", etc.
  default_profile: null         # default profile at launch
  
  auto_approve: false           # skip HITL tool approvals
  enable_memory: true           # persistent agent memory
  enable_skills: true           # load agent skills
  enable_shell: true            # allow shell execution (TUI only)
  
  shell_allow_list: []          # restrict to these commands in non-interactive mode
  sandbox: none                 # or "modal", "daytona", "runloop", "langsmith"
  system_prompt: null           # custom system prompt
  
  switcher_models:              # models shown in the TUI /model switcher
    - gpt41mini@openrouter
    - gpt_oss120@openrouter
```

### Profiles

Named presets override global settings:

```yaml
profiles:
  - name: coder
    description: "Coding assistant with shell access"
    llm: fast_model
    auto_approve: false
    enable_shell: true
    enable_memory: true
    enable_skills: true
    shell_allow_list: []        # empty = no restriction
    sandbox: none
    system_prompt: null         # use deepagents default
    tools: [fetch_url, http_request]

  - name: researcher
    description: "Web search & analysis"
    llm: default
    auto_approve: true
    enable_shell: false
    tools: [web_search, fetch_url, http_request]
    system_prompt: |
      You are a thorough researcher. Always search for current information.
```

### TUI `/model` Switcher

By default, deepagents-cli shows all installed LangChain provider models in the `/model` switcher. The integration narrows this to only models listed in `switcher_models`:

```yaml
switcher_models:
  - default                     # genai-tk tag
  - fast_model                  # genai-tk tag
  - gpt41mini@openrouter        # explicit model ID
  - claude3_haiku@anthropic     # another explicit ID
```

When you press `/model` in the TUI, you'll see these models prefixed with `genai_tk:`, e.g. `genai_tk:default`, `genai_tk:fast_model`.

**Empty list:**  Leave `switcher_models: []` to show all installed LangChain provider models (the deepagents-cli default).

---

## Command Reference

### Main TUI Command

```bash
cli agents deepagent [OPTIONS]
```

**Options:**
- `--profile, -p NAME` — Load a named profile from `deepagent.yaml`
- `--llm, -m ID` — Override model (tag or ID); takes priority over profile
- `--agent, -a NAME` — Agent name (maps to `~/.deepagents/<name>/`)
- `--auto-approve` — Skip HITL tool approval prompts
- `--resume, -r THREAD_ID` — Resume a saved thread (or omit for most recent)
- `--prompt, -q TEXT` — Pre-fill the chat input at startup
- `--list-profiles, -l` — List all configured profiles and exit

**Examples:**
```bash
cli agents deepagent
cli agents deepagent --profile researcher
cli agents deepagent --llm fast_model --agent coding_bot
cli agents deepagent --list-profiles
```

### Task Command (Non-Interactive)

```bash
cli agents deepagent task MESSAGE [OPTIONS]
```

Run a single prompt non-interactively and stream the response to stdout.

**Options:**
- `--profile, -p NAME` — Profile to use
- `--llm, -m ID` — Model override
- `--agent, -a NAME` — Agent name (default: "agent")
- `--auto-approve` — Auto-approve tool requests (default: True)
- `--quiet, -q` — Suppress diagnostic output

**Examples:**
```bash
cli agents deepagent task "Write a Python hello-world"
cli agents deepagent task --profile coder "Fix the failing tests"
cli agents deepagent task -q "Summarize recent AI news" > news.txt
```

**Note:** Shell commands are disabled by default in non-interactive mode. To enable them, set `shell_allow_list: ["cmd1", "cmd2", ...]` in the profile.

### List Agents

```bash
cli agents deepagent list [OPTIONS]
```

List all configured agent directories.

**Options:**
- `--verbose, -v` — Show full paths

### Reset Agent State

```bash
cli agents deepagent reset [OPTIONS]
```

Reset agent memories, skills, threads, or all.

**Options:**
- `--agent, -a NAME` — Agent to reset (default: "agent")
- `--target, -t TARGET` — What to reset: `memories | skills | threads | all`

**Examples:**
```bash
cli agents deepagent reset --agent mybot --target memories
cli agents deepagent reset --target all
```

### Manage Skills

#### List Skills

```bash
cli agents deepagent skills list [--agent NAME]
```

#### Create a New Skill

```bash
cli agents deepagent skills create SKILL_NAME [--agent NAME]
```

#### Show Skill Details

```bash
cli agents deepagent skills info SKILL_NAME [--agent NAME]
```

### Manage Threads

#### List Threads

```bash
cli agents deepagent threads list [--agent NAME]
```

#### Delete a Thread

```bash
cli agents deepagent threads delete THREAD_ID [--agent NAME]
```

---

## Architecture

### In-Process Integration Design

Instead of spawning deepagents-cli as a subprocess, the integration imports and uses its internal APIs directly:

```
genai-tk CLI
    ↓
DeepagentCommands (Typer command group)
    ↓
LlmFactory (model resolution)
    ↓ (BaseChatModel)
create_cli_agent() → (GenaiTkModelAdapter → LlmFactory)
    ↓
run_textual_app() (Textual TUI)
```

**Key components:**

| Module | Purpose |
|--------|---------|
| `models.py` | Pydantic models for `DeepagentProfile` and `DeepagentConfig` |
| `llm_bridge.py` | Translate genai-tk LLM identifiers → LangChain `BaseChatModel` |
| `model_adapter.py` | `GenaiTkModelAdapter` class wraps genai-tk models for deepagents-cli |
| `toml_bridge.py` | Write/update `~/.deepagents/config.toml` with curated model list |
| `cli_commands.py` | Typer command group + async task/TUI runners |

### Model Resolution Pipeline

1. **CLI receives identifier** (tag or explicit ID, e.g. "default" or "gpt41mini@openai")
2. **Priority order:**
   - CLI `--llm` flag
   - Profile `llm` field
   - Global `default_model`
   - Global fallback from `llm.models.default`
3. **LlmFactory.resolve_llm_identifier_safe()** → canonical ID, error
4. **get_llm(llm=id)** → BaseChatModel instance
5. **Pass to create_cli_agent(model=...)** → agent ready to use

### TUI `/model` Switcher

When you press `/model` in the TUI:

1. deepagents-cli calls `get_available_models()` (from `deepagents_cli.model_config`)
2. This reads `~/.deepagents/config.toml` for custom providers
3. At TUI launch, `write_genai_tk_provider()` injects a `[models.providers.genai_tk]` section with:
   - `class_path = "genai_tk.agents.deepagent_cli.model_adapter:GenaiTkModelAdapter"`
   - `api_key_env = "HOME"` (for credential check)
   - `models = [list from deepagent.yaml]`
4. When a `genai_tk` model is selected, deepagents-cli instantiates:
   ```python
   GenaiTkModelAdapter(model="gpt41mini@openrouter")
   ```
5. `GenaiTkModelAdapter.model_post_init()` resolves the ID via LlmFactory
6. All LangChain calls delegate to the underlying model

---

## Examples

### Example 1: Interactive Coding Session

```bash
cli agents deepagent --profile coder --agent coding_bot --prompt "Start by listing all Python files"
```

This launches the TUI with:
- Profile: `coder` (shell enabled, auto_approve=false, memory on)
- Agent name: `coding_bot`
- Initial prompt: "Start by listing all Python files"

Use `/model` to switch between configured models without restarting.

### Example 2: Batch Research Task

```bash
#!/bin/bash
for query in "AI trends 2026" "quantum computing progress" "climate tech innovations"; do
  cli agents deepagent task --profile researcher -q "$query" >> research.txt
  echo "---" >> research.txt
done
```

Runs 3 research tasks sequentially, appending results to `research.txt`.

### Example 3: Profile-Driven Workflow

Create a profile for each use case:

```yaml
profiles:
  - name: code_review
    llm: gpt41mini@openrouter
    system_prompt: |
      You are an expert code reviewer. Check for bugs, style issues, and improvements.
    tools: [fetch_url, http_request]

  - name: documentation
    llm: default
    auto_approve: true
    system_prompt: |
      You are a technical writer. Generate clear, concise documentation.
```

Then:
```bash
cli agents deepagent --profile code_review   # Code review mode
cli agents deepagent --profile documentation # Documentation mode
```

---

## How Models Are Curated in the TUI

### Without Curation (Deepagents Default)

All installed LangChain provider models appear in `/model` switcher:
- openai: gpt-4-turbo, gpt-4o, gpt-4-mini, ...
- anthropic: claude-opus, claude-sonnet, claude-haiku, ...
- google: gemini-2.0-flash, ...
- ... (dozens more)

**Problem:** Too many choices; users get confused about what's actually available in genai-tk.

### With Curation (genai-tk Integration)

Set `switcher_models` in `deepagent.yaml`:

```yaml
switcher_models:
  - gpt41mini@openai
  - claude3_haiku@anthropic
  - gpt41mini@openrouter
```

The TUI `/model` switcher now shows only:
- `genai_tk:gpt41mini@openai`
- `genai_tk:claude3_haiku@anthropic`
- `genai_tk:gpt41mini@openrouter`

When selected, `GenaiTkModelAdapter` resolves each identifier against genai-tk's config, automatically handling credentials and provider-specific details.

---

## Internals: TOML Provider Config

When the TUI launches, `write_genai_tk_provider()` ensures `~/.deepagents/config.toml` contains:

```toml
[models.providers.genai_tk]
class_path = "genai_tk.agents.deepagent_cli.model_adapter:GenaiTkModelAdapter"
api_key_env = "HOME"
models = ["gpt41mini@openai", "claude3_haiku@anthropic", "gpt41mini@openrouter"]
```

This tells deepagents-cli:
1. Use `GenaiTkModelAdapter` to instantiate models from the `genai_tk` provider
2. Check `HOME` env var (always set) for credential validation
3. Show these 3 models in the `/model` switcher

---

## Testing

All integration components have unit test coverage (26 tests):

```bash
cd /home/tcl/prj/genai-tk
uv run pytest tests/unit_tests/agents/deepagent/ -v
```

**Test categories:**
- Config models (`DeepagentProfile`, `DeepagentConfig`)
- Profile resolution and fallback chains
- LLM bridge identifier resolution
- `GenaiTkModelAdapter` instantiation and delegation
- TOML generation and cache clearing
- CLI registration smoke test

---

## Dependencies

The integration adds the following to `pyproject.toml`:

```toml
deepagents-cli = "^0.0.26"
```

deepagents-cli includes:
- `langchain-core`, `langchain` (already in genai-tk)
- `langgraph` (agent framework)
- `textual` (TUI framework)
- `pydantic` (config validation)

No additional dependencies beyond what genai-tk already requires.

---

## Troubleshooting

### "Missing credentials: provider 'genai_tk' is not recognized"

**Cause:** TOML provider config is missing or malformed.

**Solution:** Ensure the TOML has `api_key_env = "HOME"`:
```toml
[models.providers.genai_tk]
api_key_env = "HOME"     # ← must be present
class_path = "genai_tk.agents.deepagent_cli.model_adapter:GenaiTkModelAdapter"
models = ["..."]
```

The integration automatically adds this when the TUI launches, but if you manually edit `~/.deepagents/config.toml`, ensure it's present.

### TUI shows all LangChain models instead of just genai-tk ones

**Cause:** `switcher_models` is empty in `deepagent.yaml`.

**Solution:** Add models:
```yaml
switcher_models:
  - default
  - fast_model
  - gpt41mini@openai
```

### Profile not found

**Cause:** Typo in profile name.

**Solution:** Check `cli agents deepagent --list-profiles` for exact names (case-sensitive).

### Shell commands refused in task mode

**Cause:** Non-interactive mode disables shell by default.

**Solution:** Set `shell_allow_list` in the profile:
```yaml
profiles:
  - name: coder
    shell_allow_list: ["python", "pip", "pytest", "git"]
```

---

## Further Reading

- [deepagents-cli GitHub](https://github.com/hunterglynn/deepagents-cli)
- [genai-tk Configuration Guide](./README.md)
- [Agent Profiles in Agents.md](../AGENTS.md)
