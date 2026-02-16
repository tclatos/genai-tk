# Custom Skills Directory

This directory is for your custom deer-flow skills. Skills are modular packages that extend the agent's capabilities with specialized knowledge, workflows, or tool integrations.

## Creating a New Skill

You can create a new skill in two ways:

### Method 1: Using the skill-creator Skill (Recommended)

The easiest way is to use the built-in `skill-creator` skill to guide you through the process:

```bash
# List available profiles
uv run cli agents deerflow --list

# Run with a profile that has skill-creator enabled
uv run cli agents deerflow -p "Visual Explainer" --chat

# Then in the chat, ask:
# "I want to create a new skill for analyzing code quality"
```

The skill-creator will guide you through:
- Defining the skill's purpose and capabilities
- Creating the SKILL.md file with proper structure
- Adding any necessary scripts or references
- Testing the skill

### Method 2: Manual Creation

1. **Create a directory** for your skill:
   ```bash
   mkdir custom/my-skill-name
   cd custom/my-skill-name
   ```

2. **Create SKILL.md** with metadata and instructions:
   ```markdown
   ---
   name: my-skill-name
   description: Brief description of what this skill does
   ---

   # My Skill Name

   Detailed description and instructions for the agent...

   ## Usage

   This skill should be used when...

   ## Workflow

   1. Step 1...
   2. Step 2...
   ```

3. **(Optional) Add resources**:
   - `scripts/` - Helper scripts the agent can execute
   - `references/` - Reference documents or examples
   - `LICENSE.txt` - If you want to share your skill

4. **Enable the skill** in your profile (`config/agents/deerflow.yaml`):
   ```yaml
   deerflow_agents:
     - name: "My Profile"
       skills:
         - custom/my-skill-name  # Use custom/ prefix for custom skills
         - deep-research          # Public skills don't need prefix
   ```

## Skill Structure

A complete skill directory looks like:

```
my-skill-name/
├── SKILL.md           # Required: Skill metadata and instructions
├── LICENSE.txt        # Optional: License information
├── scripts/           # Optional: Executable scripts
│   └── helper.py
└── references/        # Optional: Reference materials
    └── guide.md
```

## Available Public Skills

Public skills are symlinked from the deer-flow repository. They include:

- **chart-visualization** - Create charts and visualizations
- **data-analysis** - Analyze datasets and generate insights
- **deep-research** - Comprehensive research workflows
- **github-deep-research** - GitHub repository analysis
- **ppt-generation** - Generate PowerPoint presentations
- **skill-creator** - Guide for creating new skills
- **image-generation** - Generate images
- **video-generation** - Generate videos
- And more...

See `/home/tcl/ext_prj/deer-flow/skills/public/` for the complete list.

## Best Practices

1. **Keep it focused** - One skill should do one thing well
2. **Be concise** - Only include information the agent needs
3. **Provide examples** - Show, don't just tell
4. **Test thoroughly** - Run your skill with different queries
5. **Document dependencies** - If your skill needs specific tools or APIs

## Skill Format Details

The SKILL.md file uses YAML frontmatter for metadata:

```markdown
---
name: skill-name          # Required: Unique identifier
description: Brief desc   # Required: One-line description
license: LICENSE.txt      # Optional: License file
---

# Skill Title

Your skill content here...
```

The content should be:
- **Instructional** - Tell the agent what to do, not how LLMs work
- **Actionable** - Provide clear steps and workflows
- **Contextual** - Include information the agent doesn't have

## Testing Your Skill

After creating a skill:

1. **Enable it** in a profile
2. **Run the agent** with that profile
3. **Check logs** for skill loading confirmation:
   ```
   INFO - Enabling skills: custom/my-skill-name
   ```
4. **Test queries** that should trigger your skill

## Troubleshooting

**Skill not loading?**
- Check that SKILL.md exists and has valid YAML frontmatter
- Verify the skill is listed in your profile configuration
- Check logs for error messages

**Skill not being used?**
- Make sure your query matches the skill's description
- Try explicitly mentioning the skill in your query
- Check that required tools/APIs are configured

## Examples

See the public skills directory for examples:
- `/home/tcl/ext_prj/deer-flow/skills/public/skill-creator/` - The skill creator itself
- `/home/tcl/ext_prj/deer-flow/skills/public/data-analysis/` - Data analysis workflow
- `/home/tcl/ext_prj/deer-flow/skills/public/deep-research/` - Research workflow

## More Information

- [Deer-flow Skills Documentation](https://github.com/bytedance/deer-flow/blob/main/skills/README.md)
- [Skill Creator Guide](../public/skill-creator/SKILL.md)
