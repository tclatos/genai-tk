# Skills System

Skills are SKILL.md files — YAML+markdown documents that give agents domain knowledge
on demand. They're loaded progressively (not injected into every prompt), organized by
category (dev/agent/project), and support multiple sources (bundled, community, custom).

---

## Quick start

```bash
# List all skills in your project, grouped by category
cli skills list

# Add a bundled skill
cli skills add getting-started

# Install community skills (from skills.sh)
cli skills add --skillssh langchain-ai/langchain-skills

# Create a new skill
cli skills create my-domain-skill

# Validate all skills
cli skills validate --all
```

---

## Skill categories

Skills are organized by source and purpose:

| Category | Source | Location | Examples |
|----------|--------|----------|----------|
| **Dev** | genai-tk / Copilot | `skills/genai-tk/`, `skills/copilot/` | scaffolding, git workflows, test patterns |
| **Agent** | Community / public | `skills/public/`, `skills/langchain-examples/` | web search, data analysis, coding |
| **Project** | Your project | `skills/custom/` | domain-specific knowledge, internal tools |

Display via `cli skills list`:

```
Dev Skills (genai-tk / Copilot)
  ✓ scaffolding              genai-tk scaffolding best practices
  ✓ git-workflows            Git branching and commit strategies

Agent Skills (Community / Public)
  ✓ web-search               Tavily web search integration
  ✓ data-analysis             Pandas and analysis patterns

Project Skills (Custom)
  ✓ sales-pricing             RFQ pricing models and rules
  ✓ deployment-runbook        Internal ops deployment guide
```

---

## SKILL.md format

### Anatomy

```yaml
---
meta:
  name: "web-search"
  category: "agent"           # dev | agent | project
  tags: ["search", "web", "tavily"]
  min_tokens: 500             # estimated token cost to include
---

# Web Search with Tavily MCP

This skill enables agents to search the web in real-time.

## Setup

```bash
export TAVILY_API_KEY=your_key
```

## Usage

When this skill is loaded, the agent can:
- Search current events
- Find recent articles
- Verify facts against live data

## Example

Agent: "Find the latest developments in AI"
→ Searches the web for recent news
→ Aggregates results
→ Returns summary to user

## Limitations

- Requires TAVILY_API_KEY
- 15-20 second latency per search
- ~1000 free queries/month on free plan
```

### YAML frontmatter fields

| Field | Type | Required | Example |
|-------|------|----------|---------|
| `meta.name` | string | ✓ | "web-search" |
| `meta.category` | string | ✓ | "agent" (or "dev", "project") |
| `meta.tags` | list | | `["search", "tavily", "web"]` |
| `meta.min_tokens` | int | | 500 (for agent prompt planning) |
| `meta.version` | string | | "1.0.0" |
| `meta.author` | string | | "Your Name" |
| `meta.url` | string | | link to source/docs |

### Markdown body

The rest of the file is **plain markdown** — no special syntax required. Use:
- Headers (##, ###) to organize sections
- Code blocks for setup/examples
- Lists for features/limitations
- Links to external resources

---

## Discovery and loading

### Automatic discovery

The skills system auto-discovers SKILL.md files:

```bash
skills/
  genai-tk/          → category: dev
  copilot/           → category: dev
  public/            → category: agent
  langchain-examples/→ category: agent
  custom/            → category: project
```

Use `cli skills list` to see all discovered skills and their category assignments.

### Manual discovery

```python
from genai_tk.main.skills_manager import discover_all_skills

skills = discover_all_skills(project_dir="/path/to/project")

# Filter by category
dev_skills = [s for s in skills if s.category == "dev"]
agent_skills = [s for s in skills if s.category == "agent"]
```

### Filtering at runtime

```bash
# List only agent skills
cli skills list --category agent

# List only project/custom skills
cli skills list --category project
```

---

## Adding skills

### From bundled (genai-tk)

```bash
cli skills add getting-started
cli skills add git-workflows
cli skills add refactoring-patterns
```

See `cli skills list` for full list of bundled skills.

### From community (skills.sh)

Install from the community registry at [skills.sh](https://www.skills.sh):

```bash
# Search for a skill (requires npx)
npx skills.sh search "web search"

# Install a skill
cli skills add --skillssh langchain-ai/langchain-skills
cli skills add --skillssh anthropic-ai/anthropic-skills

# Install with a specific subdirectory
cli skills add --skillssh owner/repo --path specific-skill
```

### From git

```bash
cli skills add --git https://github.com/your-org/my-skills --path web-search
cli skills add --git git@github.com:private/skills.git --path my-skill
```

### Create a new project skill

```bash
cli skills create pricing-models
# Scaffolds: skills/custom/pricing-models/SKILL.md
# Edit to add your domain knowledge
```

Then edit `skills/custom/pricing-models/SKILL.md`:

```yaml
---
meta:
  name: "pricing-models"
  category: "project"
  tags: ["pricing", "sales", "margin"]
---

# RFQ Pricing Models

This skill explains our pricing logic for RFQ responses.

## Margin targets
- Standard products: 15-20%
- Custom orders: 20-25%
- ... (your knowledge here)
```

---

## Agent skill loading

### With LangChain agents

Skills are automatically available to the agent if:
1. The agent profile has `skill_directories:` configured in `config/agents/langchain.yaml`
2. Skills are in those directories

```yaml
langchain_agents:
  research:
    name: "Research"
    skill_directories:
      - ${paths.project}/skills/public
      - ${paths.project}/skills/custom
    available_skills:
      - web-search       # restrict to specific skills (omit to allow all)
      - data-analysis
```

When the agent needs knowledge, the skill content is injected into the prompt.

### With Deer-flow

Deer-flow has its own skill loading via the embedded client:

```yaml
deerflow_agents:
  - name: research
    skill_directories:
      - ${paths.project}/skills
    available_skills:      # optional: restrict which skills are available
      - web-search
      - public/data-analysis
```

### With Copilot / Cursor

Skills are discovered but not automatically injected. Share a skill with your AI assistant:

1. Copy the SKILL.md content
2. Paste into the chat
3. Or reference: "Use the knowledge in `skills/custom/pricing-models/SKILL.md`"

---

## Best practices

### Organize by purpose

- **Dev skills**: scaffolding patterns, code style, testing conventions
- **Agent skills**: external integrations, tools, APIs
- **Project skills**: business logic, domain knowledge, company policies

### Keep skills focused

One skill = one topic. Don't merge multiple domains:

❌ **Too broad**:
```yaml
meta:
  name: "company-knowledge"
  # (100+ lines covering pricing, deployment, sales, HR)
```

✅ **Focused**:
```yaml
meta:
  name: "pricing-models"
  # (focused on pricing logic only)

meta:
  name: "deployment-runbook"
  # (focused on deployment steps only)
```

### Include examples and limitations

Agents work better with concrete examples:

```markdown
## Example

Q: "What margin should we charge for a $5000 custom order?"
A: 20-25%. This is a custom order, so apply the higher margin.

## Limitations
- Does not apply to government contracts
- Check with sales before finalizing complex deals
```

### Use tags for discovery

Tags help agents (and humans) find relevant skills:

```yaml
meta:
  tags: ["pricing", "sales", "margin", "rfq"]
```

### Version your skills

For important domain knowledge:

```yaml
meta:
  name: "pricing-models"
  version: "2.0.0"
  author: "Sales Engineering"
  updated: "2024-01"
```

### Link to external docs

Point to authoritative sources:

```markdown
## Further reading

- [Pricing policy](https://docs.company.com/pricing)
- [Sales playbook](https://confluence.company.com/sales)
```

---

## Validation

### Lint all skills

```bash
cli skills validate --all
```

Checks:
- ✓ Valid YAML frontmatter
- ✓ Required fields (`name`, `category`)
- ✓ Valid category ("dev" | "agent" | "project")
- ✓ No broken markdown syntax
- ✓ Reasonable file size

### Lint a specific skill

```bash
cli skills validate skills/custom/pricing-models/SKILL.md
```

---

## Integration with AI assistants

### Copilot (GitHub)

1. List available skills: `@workspace What skills are available?`
2. Load a skill: `@workspace Read skills/custom/pricing-models/SKILL.md and answer: "What's the margin target?"`
3. Create skill: `@workspace Create a new skill for deployment procedures`

### Cursor / Windsurf

1. Reference directly in chat: "Use the knowledge in `.skills/custom/pricing-models/SKILL.md`"
2. Or ask to read: "@Workspace read the pricing models skill"

### OpenCode / Claude Code

Skills are included in your project's `AGENTS.md` — mention them in your query.

---

## Advanced: Custom skill sources

### Python code discovery

Skills don't have to be static `.md` files. You can generate them dynamically:

```python
def get_dynamic_skill() -> SkillInfo:
    return SkillInfo(
        name="live-api-schema",
        category="project",
        content=generate_api_schema_markdown(),
    )
```

Then register in your agent profile or manually load in code.

### Remote skills

In theory, you can fetch skills from a remote source (API, database). This isn't
yet supported by the CLI, but can be implemented in your agent code:

```python
import httpx
from genai_tk.main.models_skills import SkillInfo

async def load_remote_skills():
    async with httpx.AsyncClient() as client:
        res = await client.get("https://api.example.com/skills")
        # parse and return SkillInfo objects
```

---

## Troubleshooting

**Q: Skill not found when I add it**

```bash
cli skills list --category agent   # verify it's discovered
cli skills validate --all          # check for syntax errors
```

**Q: Agent is ignoring my skill**

Check the agent profile:
- Does `skill_directories:` include the skill's directory?
- Does `available_skills:` (if present) include the skill name?
- Is the skill YAML valid? Run `cli skills validate`

**Q: How do I update a community skill?**

```bash
# Remove and re-add
cli skills remove --source community web-search
cli skills add --skillssh langchain-ai/langchain-skills
```

**Q: Can I have skills that aren't SKILL.md files?**

The CLI expects SKILL.md format. But you can:
- Store extra docs in `skills/custom/<skill>/README.md`
- Load custom skill sources in Python
- Reference files in your agent prompt: "Refer to `docs/pricing.md` for details"

**Q: How do I make a skill private to my project?**

Put it in `skills/custom/` (not `skills/public/`). It won't be shared unless you
commit it to a public Git repo.
