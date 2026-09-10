# Slidev Atos Corporate Theme (`slidev-theme-atos`)

Executive corporate presentation theme for [Slidev](https://sli.dev), faithfully reflecting standard **Atos PowerPoint & PDF design guidelines**.

Designed for AI engineers, software developers, and consultants creating executive-ready technical presentations in Markdown with automated brand compliance.

---

## 🎨 Visual Brand Identity

- **Color Palette:**
  - Dark Navy: `#00005B` (`--atos-navy`)
  - Corporate Atos Blue: `#0073E6` (`--atos-blue`)
  - Accent Cyan: `#43C7F4` (`--atos-cyan`)
  - Pure White: `#FFFFFF` (`--atos-white`)
  - Body Text Navy: `#161650` (`--atos-text`)
- **Typography:** Arial, Helvetica, sans-serif
- **Slide Ratio:** 16:9 (`1280x720` canvas)
- **Corporate Assets:** Embedded vector Atos logo (`atos-blue.svg`, `atos-white.svg`) and geometric wave curves (`cover-curve.svg`, `section-curve.svg`).

---

## 🚀 How to Use in Any Project

You can use this theme in any project using any of the following approaches:

### Approach 1: Direct Local / Relative Path (Simplest & Zero Setup)

In your presentation's `slides.md` frontmatter, point `theme:` directly to the theme path:

```yaml
---
theme: /home/tcl/prj/slidev-theme-atos   # or relative path: ../slidev-theme-atos
title: My Executive Presentation
presentationDate: 10/09/2026
confidentiality: © Atos Group - for internal use
aspectRatio: 16/9
canvasWidth: 1280
fonts:
  sans: Arial
colorSchema: light
---
```

Run Slidev:
```bash
npx @slidev/cli slides.md --open
```

---

### Approach 2: NPM Package / Link (Standard Package Workflow)

1. In the theme directory:
```bash
cd /home/tcl/prj/slidev-theme-atos
npm link
```

2. In your presentation project:
```bash
npm link slidev-theme-atos
```

3. In `slides.md`:
```yaml
---
theme: slidev-theme-atos
title: My Presentation
---
```

---

### Approach 3: Package.json Dependency

Add the theme to your project's `package.json`:
```json
{
  "dependencies": {
    "slidev-theme-atos": "file:/home/tcl/prj/slidev-theme-atos"
  }
}
```

---

## 📑 Available Layouts

| Layout | Purpose | Key Elements |
|---|---|---|
| `atos-cover` (or `cover`) | Presentation Title Slide | Dark navy background, cyan subtitle, author block, right curve, white logo. |
| `atos-section` (or `section`) | Section Transition Slide | Corporate blue background, navy subtitle, cyan curve graphic, white logo. |
| `atos-default` (or `default`) | Standard Content Slide | Clean white background, blue header, navy text, automatic footer. |
| `atos-two-cols` (or `two-cols`) | 2-Column Comparison | Split layout with `::left::` and `::right::` named slots. |
| `atos-image` (or `image-right`) | Content + Image / Diagram | Text on left, image/media slot on right. |

---

## 💡 Example Slide Deck (`slides.md`)

```md
---
theme: /home/tcl/prj/slidev-theme-atos
title: Agentic AI and Document Intelligence
presentationDate: 10/09/2026
confidentiality: © Atos Group - for internal use
aspectRatio: 16/9
canvasWidth: 1280
fonts:
  sans: Arial
colorSchema: light
---

# Agentic AI and Document Intelligence
## From experimentation to enterprise differentiation

::author::
**Thierry Caminel**  
Global AI Engineering

---
layout: atos-section
---

# Harness Engineering
## Building the execution environment around AI agents

---
layout: atos-default
---

# Why Harness Engineering Matters
## The model alone is not the system

- **Execution Sandbox:** Isolation, permission boundaries, and safety policies
- **Context & Memory:** Graph navigation, semantic indexing, and session memory
- **Skills System:** Progressive disclosure of reusable engineering patterns
- **Observability:** NeMo Relay telemetry, structured logs, and audit trails
- **Deterministic Math:** CodeAct Python runtime eliminating token math errors

---
layout: atos-two-cols
---

<template v-slot:header>

# Architectural Principles
## Layered separation of concerns

</template>

::left::

### Agent Cognitive Layer
- **Planning & Decomposition:** Sub-task synthesis
- **Reasoning:** Step-by-step logic
- **Tool Selection:** Dynamic invocation

::right::

### Harness Execution Layer
- **Sandboxed Execution:** Linux containers
- **Enterprise Connectors:** Ladybug graph DB
- **Observability:** ATOF 0.1 telemetry events

---
layout: atos-image
image: /atos/cover-curve.svg
---

# Multi-Modal Intelligence
## Structured Graph Navigation

- Direct document navigation via hierarchical TOC
- Elimination of chunking fragmentation artifacts
- 100% deterministic precision on financial tables
- Sub-millisecond Cypher query execution
```

---

## 🛠️ Global Footer Behavior

The footer (`global-bottom.vue`) automatically appears on all content slides and is hidden on cover/section slides:
- **Left:** `presentationDate | title | confidentiality`
- **Right:** Slide page counter (`currentPage`) and corporate Atos blue logo.

Configurable in frontmatter:
```yaml
---
title: My Title
presentationDate: 10/09/2026
confidentiality: © Atos Group - for internal use
pageNumbers: true # set to false to hide page numbers
---
```
