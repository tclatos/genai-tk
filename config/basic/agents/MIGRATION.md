# Migration: LangChain Configuration Reorganization

## Changes Made

The single `langchain.yaml` file has been reorganized into a directory-based structure (`langchain/`) with separate YAML files grouped by function.

### Old Structure
```
config/basic/agents/
└── langchain.yaml (monolithic 750+ lines)
```

### New Structure
```
config/basic/agents/
├── langchain/
│   ├── README.md                 # Documentation
│   ├── defaults.yaml             # Global defaults and settings
│   ├── simple.yaml               # Simple React agents (demo profiles)
│   ├── text2sql.yaml             # Text-to-SQL agent
│   ├── browser.yaml              # Browser automation agents
│   └── deep.yaml                 # Advanced Deep agents
└── langchain.yaml.old            # Backup of original file
```

## File Organization

| File | Purpose | Profiles |
|------|---------|----------|
| `defaults.yaml` | Global defaults | - |
| `simple.yaml` | Demo/simple profiles | simple, filesystem, weather, chinook |
| `text2sql.yaml` | Database querying | text2sql |
| `browser.yaml` | Web automation | Browser Agent, Browser Agent Direct |
| `deep.yaml` | Advanced agents | Research, Coding, Data Analysis, Web Research, Documentation Writer, Stock Analysis |

## Benefits

✅ **Better Organization** - Profiles grouped by function, easier to maintain
✅ **Easier Navigation** - Find profiles faster in focused files
✅ **Scalability** - Easy to add new profile files as needs grow
✅ **Separation of Concerns** - Each file has a clear purpose
✅ **Easier Collaboration** - Teams can work on different profile types without conflicts

## Configuration Loading

The config loader automatically discovers and merges all YAML files from the `langchain/` directory. No code changes required.

- Files are loaded automatically when the application starts
- All `langchain_agents:` sections are merged into a single configuration
- Profiles inherit from `defaults` defined in `defaults.yaml`

## Rollback

If you need to revert to the monolithic structure:

```bash
# Restore the old file
mv config/basic/agents/langchain.yaml.old config/basic/agents/langchain.yaml

# Remove the new directory
rm -rf config/basic/agents/langchain/
```

## No Code Changes Needed

The application configuration loader handles directory-based configurations automatically. No updates to application code are required.
