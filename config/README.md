# Configuration Directory Structure

This directory contains all configuration files for the GenAI Toolkit application.

## Overview

The configuration system uses **OmegaConf** with hierarchical YAML files and profile overlays:

1. **Auto-scan**: All `*.yaml` files are automatically discovered and merged
2. **Profile overlays**: Files in `config/profiles/<profile>/` are loaded on top
3. **Overrides**: `config/overrides.yaml` is loaded last (highest priority)

## File Structure

### Root Configuration Files

- **`app_conf.yaml`** - Main entry point; sets profile name and paths
- **`overrides.yaml`** - Local overrides (git-ignored, loaded last)
- **`mcp_servers.yaml`** - Model Context Protocol server configurations
- **`rag.yaml`** - RAG system configuration (chunkers, embeddings stores, retrievers)
- **`workflows.yaml`** - Workflow definitions and profiles

### Subdirectories

#### `agents/` - Agent configurations

- **`langchain/`** - LangChain agent profiles
  - `defaults.yaml` - Default settings
  - `simple.yaml` - Simple and basic agents
  - `deep.yaml` - Deep agents (research, coding, etc.)
  - `browser.yaml` - Browser automation agents
  - `text2sql.yaml` - Text-to-SQL agents

- **`deepagent/`** - Deep agents (dict-keyed)
  - `global.yaml` - Global settings
  - `coder.yaml` - Coding agent profile
  - `fast.yaml` - Fast agent profile
  - `researcher.yaml` - Researcher agent profile

#### `providers/` - LLM and embeddings declarations

- **`llm.yaml`** - LLM model declarations
- **`embeddings.yaml`** - Embeddings model declarations

#### `profiles/` - Profile overlays

Each profile is a directory loaded when `GENAITK_PROFILE=<name>` is set:

- **`local/`** - Default profile (loaded by default). Edit `genai_def.yaml` for local model/path settings.
- **`pytest/`** - Test profile. Fake models, memory caches — loaded automatically by the test suite.
- **`test_unit/`** - Config-manager unit tests. Provides `test_env:` and `prod_env:` context targets.
- **`prod/`** - Example production profile (create as needed)
- **`custom/`** - Custom profiles (create as needed)

Each profile directory can contain any YAML files that override base configuration.

## How Configuration Loading Works

1. **Scan base files**: All `*.yaml` files in `config/` and subdirectories (sorted alphabetically)
   - Files like `agents/langchain/defaults.yaml`, `providers/llm.yaml`, etc.

2. **Load profile overlay**: All `*.yaml` files in `config/profiles/<profile>/` (if profile is selected)
   - By default, `config/profiles/local/` is loaded

3. **Load overrides**: `config/overrides.yaml` is loaded last (wins over all)

## Environment Variables

OmegaConf supports variable interpolation:
- `${oc.env:VARIABLE,default_value}` - Read from environment
- `${paths.config}` - Auto-expanded to config directory path
- `${paths.project}` - Auto-expanded to project root

## Selecting Profiles

### At Runtime

```bash
# Use local profile (default)
python myapp.py

# Use prod profile
GENAITK_PROFILE=prod python myapp.py

# Use custom profile
GENAITK_PROFILE=custom python myapp.py
```

### In Code

```python
from genai_tk.utils.config_mngr import global_config, switch_profile

config = global_config()
config.get("profile")           # Returns the current profile name

# Switch deployment profile (reloads all config files)
switch_profile("prod")          # set GENAITK_PROFILE=prod + reload
switch_profile("pytest")        # switch to test profile

# Activate a context overlay without reloading
global_config().use_context("training_local")
```

## Adding Configuration

1. **For base configuration**: Create/edit files in `config/` root or subdirectories
2. **For environment-specific settings**: Create `config/profiles/<name>/` directory
3. **For local overrides**: Edit `config/overrides.yaml`

## Common Configuration Tasks

### Add a new LLM model

Edit `config/profiles/local/genai_def.yaml`:
```yaml
llm:
  models:
    my_model: my-model@provider
```

### Override settings for production

Create `config/profiles/prod/`:
```
mkdir -p config/profiles/prod
cat > config/profiles/prod/llm.yaml << 'EOF'
llm:
  models:
    default: gpt_4o@openai
EOF
```

## See Also

- [Configuration Documentation](../docs/configuration.md)
- [Agent Configuration](../docs/agents.md)
- [RAG Configuration](../docs/rag.md)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
