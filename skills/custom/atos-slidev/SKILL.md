---
name: atos-slidev
description: Create, style, and publish executive Slidev presentation decks following standard Atos corporate brand guidelines, multi-deck GitHub Pages routing, and automated portal deployment.
---

# Atos Slidev Presentations Skill

Use this skill whenever creating, updating, styling, or deploying **Slidev presentations** with the **Atos corporate brand identity**.

---

## 🎨 Brand Identity & Tokens

- **Color Palette:**
  - Dark Navy: `#00005B` (`--atos-navy`) — Cover backgrounds, section subtitles, table text.
  - Corporate Atos Blue: `#0073E6` (`--atos-blue`) — Primary headings (`h1`), section backgrounds, brand links.
  - Accent Cyan: `#43C7F4` (`--atos-cyan`) — Cover subtitles (`h2`), decorative curves.
  - Pure White: `#FFFFFF` (`--atos-white`) — Slide canvas for default content and two-column layouts.
  - Body Text Navy: `#161650` (`--atos-text`) — Body paragraphs, list items.
- **Typography:** Arial, Helvetica, sans-serif.
- **Slide Dimensions:** 16:9 widescreen format (`canvasWidth: 1280`, `aspectRatio: 16/9`).
- **Official Assets:** Embedded official high-res PNG logos (`atos-blue.png`, `atos-white.png`) and vector curves (`cover-curve.svg`, `section-curve.svg`).

---

## 🏗️ Theme Architecture (`slidev-theme-atos`)

The master theme lives at `/home/tcl/prj/slidev-theme-atos` and is bundled inside each repository under `docs/theme/`.

### 1. Approved Layouts

| Layout Name | Aliases | Use Case | Structure & Slots |
|---|---|---|---|
| `atos-cover` | `cover` | Presentation title slide | Dark navy `#00005B`, large white `h1`, cyan `h2`, `::author::` slot, wave arc graphic, white Atos logo. |
| `atos-section` | `section` | Chapter / Section divider | Corporate blue `#0073E6`, white `h1`, navy `h2`, cyan curve, white Atos logo. |
| `atos-default` | `default` | Standard content slide | White canvas, blue `h1`, navy `h2`, clean bullets, automatic corporate footer. |
| `atos-two-cols` | `two-cols` | Balanced 2-column comparison | `<template v-slot:header>` for title, `::left::` and `::right::` content columns. |
| `atos-image` | `image-right` | Content + media / diagram | Left content area, right image/media slot via `image: <path>` frontmatter or `::image::` slot. |

### 2. Automatic Global Footer (`global-bottom.vue`)
Appears on all standard content slides and is automatically hidden on `atos-cover`, `atos-section`, and full-screen layouts:
- **Left side:** `presentationDate | title | confidentiality`
- **Right side:** Dynamic page numbers (`$slidev.nav.currentPage`) and corporate Atos blue logo.

---

## 📄 Frontmatter Template

Every slide deck markdown file (`docs/slides-<topic>.md`) should begin with this standard headmatter:

```yaml
---
theme: ./theme
title: "Your Presentation Title"
presentationDate: "10/09/2026"
confidentiality: "© Atos Group - for internal use"
aspectRatio: 16/9
canvasWidth: 1280
fonts:
  sans: Arial
colorSchema: light
highlighter: shiki
lineNumbers: true
---

# Your Presentation Title
## Subtitle or Executive Summary

::author::
**Author Name**  
Global AI Engineering

---
layout: atos-section
---

# Section Name
## Key Objectives & Focus Areas

---
layout: atos-default
---

# Content Slide Title
## Subtitle or Context

- **Key Point 1:** Description with bold highlight.
- **Key Point 2:** Description with bold highlight.
- **Key Point 3:** Description with bold highlight.

---
layout: atos-two-cols
---

<template v-slot:header>

# Comparative Architecture
## Left vs. Right Analysis

</template>

::left::

### Cognitive Layer
- Planning & reasoning
- Dynamic tool execution

::right::

### Execution Layer
- Sandboxed runtime
- Single-writer database lock

---
layout: atos-section
---

# Questions & Technical Discussion
## Thank you for your attention
```

---

## 🌐 Multi-Deck Architecture & GitHub Pages Routing

Multiple presentations can coexist in the same repository under `docs/slides-<slug>.md`.

### Naming & Routing Convention

| Source File | Deployed Sub-Route URL | Example |
|---|---|---|
| `docs/slides-benchmark.md` | `https://<user>.github.io/<repo>/benchmark/` | Technical benchmark findings |
| `docs/slides-executive.md` | `https://<user>.github.io/<repo>/executive/` | C-level briefing |
| `docs/slides-architecture.md` | `https://<user>.github.io/<repo>/architecture/` | Deep engineering walkthrough |
| **All Decks Hub** | `https://<user>.github.io/<repo>/` | **Atos Interactive Portal Landing Page** |

### Automated Multi-Deck Builder (`scripts/build_slides.py`)
Scans `docs/` for all `slides*.md` files, compiles each into `dist/<slug>/` with the correct `--base` URL, and synthesizes an executive portal at `dist/index.html`.

Run locally:
```bash
python3 scripts/build_slides.py
```

### GitHub Actions CI/CD (`.github/workflows/deploy-slides.yml`)
Automatically triggered when pushing changes to `docs/slides*.md` or `docs/theme/**`:
1. Checks out repository and sets up Node.js.
2. Executes `python3 scripts/build_slides.py`.
3. Configures GitHub Pages and uploads `dist/` artifact.
4. Deploys to GitHub Pages via `actions/deploy-pages@v4`.

---

## 💻 CLI Commands Cheat Sheet

| Task | Command |
|---|---|
| **Live interactive dev server** | `npx @slidev/cli docs/slides-<name>.md --open` |
| **Build single deck to static SPA** | `npx @slidev/cli build docs/slides-<name>.md --base /<repo>/<name>/ --out dist/<name>` |
| **Build all decks + portal** | `python3 scripts/build_slides.py` |
| **Export deck to PDF** | `npx @slidev/cli export docs/slides-<name>.md` |
| **Backup Slidev project** | `python3 ~/backup_projects.py sync slidev-theme-atos` |
