---
name: add-skill
description: How to create a new SKILL.md for a genai-tk project — format, conventions, and wiring to agent profiles.
tags: [skills, agents, documentation]
version: "1.0"
---

# Create a New Skill

Skills are SKILL.md files that give agents **procedural knowledge on demand**.
They are loaded progressively — agents read them when needed, not on every call.

## Quick Method (CLI)

```bash
cli skills create my-skill-name
# Then edit: skills/custom/my-skill-name/SKILL.md
```

## Manual Method

### Step 1: Create the directory

```bash
mkdir -p skills/custom/my-skill-name
```

### Step 2: Create SKILL.md

```markdown
---
name: my-skill-name
description: One sentence — what this skill enables an agent to do.
tags: [tag1, tag2]
version: "1.0"
author: ""
---

# My Skill Name

## When to Use

Describe exactly when an agent should apply this skill. Be specific:
"Use this skill when the user asks to query the knowledge graph with Cypher."

## Workflow

1. First action — be imperative, not passive
2. Second action — include concrete examples (selectors, commands, URLs)
3. Third action — describe expected outcomes

## Code Map

| Concern | Path |
|---------|------|
| Main logic | `<package>/...` |
| Config | `config/...` |
| Tests | `tests/...` |

## Examples

```bash
# Concrete CLI example
cli agents langchain -p my_agent "use my-skill to do X"
```

## References

- Related skill: `skills/custom/related-skill/SKILL.md`
- genai-tk docs: `docs/EXTENDING.md`
```

### Step 3: Validate

```bash
cli skills validate my-skill-name
```

### Step 4: Wire to an agent profile

In `config/agents/langchain.yaml`:
```yaml
langchain_agents:
  my_agent:
    skill_directories:
      - ${paths.project}/skills/custom
```

## Format Rules (skills.sh compatible)

- **Frontmatter** (YAML between `---`) must have `name` and `description`
- **`## When to Use`** — tells the agent when to load this skill
- **`## Workflow`** — numbered steps, imperative voice
- **`## Code Map`** — table of paths the agent needs to navigate
- Keep total length under **300 lines** — shorter skills are loaded more reliably
- **No passive voice** — "Navigate to X", not "X should be navigated to"

## Sharing Your Skill

To share publicly via skills.sh:
1. Push to a GitHub repo as `<repo>/<skill-name>/SKILL.md`
2. Others install with: `cli skills add --skillssh your-org/your-repo`
3. Optionally register at https://www.skills.sh

## Code Map

| Concern | Path |
|---------|------|
| Skill discovery | `genai_tk/main/skills_manager.py` |
| Skill validation | `genai_tk/main/skills_manager.py::validate_skill` |
| Skills CLI | `genai_tk/cli/commands_skills.py` |
| Bundled skills | `skills/genai-tk/`, `skills/copilot/` |
