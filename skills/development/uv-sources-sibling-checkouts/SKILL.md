---
name: uv-sources-sibling-checkouts
description: How to wire uv sources when developing genai-tk/genai-graph and downstream projects (wiki, officeqa, rfq_pricing) as sibling editable checkouts instead of git pins. Covers the four failure modes that cost the most debugging time — override-dependencies silently stripping forwarded extras, [tool.uv.sources] baked into a library conflicting with consumer pins, uv rejecting URL sources declared by non-root packages, and unpublished packages (like prefect-yaml) falling back to incompatible PyPI releases when omitted from consumer dependencies — with the exact error messages as search anchors.
tags: [uv, packaging, dependencies, sources, extras]
version: "1.1"
---

# uv Sources for Sibling-Checkout Development

genai-tk, genai-graph and their downstream projects (wiki, officeqa,
rfq_pricing) are developed as **sibling checkouts**: the consumer's
`[tool.uv.sources]` points the whole dependency graph at local editable
checkouts instead of git. Sources wiring is subtle — three rules prevent the
three failure modes that each cost hours of debugging.

## Rule 1 — Never `override-dependencies` a package whose extras you forward

uv applies an override to *every* requirement for that package — including the
consumer project's own forwarded `genai-tk[<extra>]` extras — and an override
without extras **silently strips them**. Worse, `uv sync --extra harnessing`
still "succeeds" (it considers the already-wrong lockfile consistent), so
nothing warns you until deep agents die at import time.

```toml
# WRONG — consumer pyproject.toml
[tool.uv]
override-dependencies = ["genai-tk @ file:///home/tcl/prj/genai-tk"]
```

Consumer-level `[tool.uv.sources]` already pins the whole graph (including
genai-graph's transitive genai-tk dependency) to the local checkouts — the
override adds nothing and breaks extras forwarding.

### Symptom

```text
ImportError: Optional feature 'harnessing' is not installed
```

…even though `uv sync --extra harnessing` exits 0.

## Rule 2 — Never bake `[tool.uv.sources]` for a dependency in a *library*

uv applies a dependency package's own sources to consumers that resolve that
package from a path or git checkout, so a baked URL (git pin or relative path)
**conflicts with whatever source the consumer chooses** (sibling editable
checkouts, its own git pin, …).

```toml
# WRONG — in genai-graph/pyproject.toml
[tool.uv.sources]
genai-tk = { git = "https://github.com/tclatos/genai-tk", rev = "main" }
```

Instead, keep the dependency declaration bare and carry this repo's own git
reference in a **root-only dependency-group** (consumers never inherit groups):

```toml
# genai-graph/pyproject.toml — the proven shape
[dependency-groups]
# Root-only: carries the git reference for this repo's own `uv sync`;
# consumers never inherit dependency-groups.
toolkit = ["genai-tk @ git+https://github.com/tclatos/genai-tk@main"]
```

…and no `[tool.uv.sources]` entry for genai-tk in the library.

### Symptom

```text
Requirements contain conflicting URLs for genai-tk
```

## Rule 3 — Prefer direct URL requirements over URL sources for git pins

uv rejects URL *sources* declared by non-root packages when the package itself
resolves from git. A direct URL requirement inside package metadata (e.g. an
extra) is honored in every resolution context.

```toml
# genai-tk/pyproject.toml — the proven shape (harnessing extra)
[project.optional-dependencies]
harnessing = [
    # Direct URL requirement, pinned to a tested commit — not [tool.uv.sources].
    "deerflow-harness[tui] @ git+https://github.com/bytedance/deer-flow@<tested-rev>#subdirectory=backend/packages/harness",
]
```

### Symptom

```text
URL dependencies must be expressed as direct requirements or constraints
```

## Rule 4 — List unpublished internal packages in consumer dependencies

uv's `[tool.uv.sources]` only applies to packages that are declared directly in the
consumer project's `[project.dependencies]`, `[project.optional-dependencies]`,
or `[dependency-groups]`. If an unpublished or internal package (such as `prefect-yaml`)
is only a *transitive* dependency of `genai-tk` or `genai-graph`, uv ignores `[tool.uv.sources]`
for that package and queries PyPI. If an obsolete or unrelated package with the same name
exists on PyPI (e.g. legacy `prefect-yaml` requiring `prefect<3`), uv attempts to resolve it and
fails with an unsatisfiable conflict.

```toml
# Consumer pyproject.toml — must declare prefect-yaml in dependencies
[project]
dependencies = [
    "genai-tk",
    "genai-graph",
    "prefect-yaml",
]

[tool.uv.sources]
genai-tk = { path = "../genai-tk", editable = true }
genai-graph = { path = "../genai-graph", editable = true }
prefect-yaml = { path = "../prefect-yaml", editable = true }
```

### Symptom

```text
Because all versions of prefect-yaml depend on prefect>=2.7.1,<3.0.0 and genai-graph==0.1.2 depends on prefect>=3.6.7, we can conclude that genai-graph==0.1.2 and all versions of prefect-yaml are incompatible.
```

## The Correct Consumer Setup

```toml
# downstream project pyproject.toml (wiki/officeqa/rfq_pricing pattern)
[project]
dependencies = [
    "genai-tk",
    "genai-graph",
    "prefect-yaml",
]

[tool.uv.sources]
# Local editable checkouts — consumer-level sources win for the whole graph.
genai-tk = { path = "../genai-tk", editable = true }
genai-graph = { path = "../genai-graph", editable = true }
prefect-yaml = { path = "../prefect-yaml", editable = true }

[project.optional-dependencies]
# Forward genai-tk optional extras — install with: uv sync --extra <name>
harnessing = ["genai-tk[harnessing]"]
docling = ["genai-tk[docling]"]
```

…with no `override-dependencies` anywhere.

## Debug Checklist

1. Read the exact uv/import error — each symptom above maps 1:1 to a rule.
2. `grep -n "override-dependencies" pyproject.toml` — remove any entry for a
   package whose extras you forward.
3. Check every *library* in the graph for `[tool.uv.sources]` on packages that
   consumers resolve differently; move the pin into a root-only
   dependency-group.
4. Express git pins inside extras as direct URL requirements, not sources.
5. If an internal package (like `prefect-yaml`) fails resolution against PyPI,
   ensure it is explicitly listed in consumer `[project.dependencies]` so
   `[tool.uv.sources]` takes effect.
6. Re-lock from scratch (`rm -rf .venv uv.lock && uv sync --extra <name>`) —
   uv may consider a previously-wrong lockfile "consistent".
