"""Rule-based workflow routing models.

Generalizes the markdownize profile pattern one level up: instead of selecting
a converter inside a single flow, an :class:`IngestRouteTable` selects a whole
**workflow** (YAML-defined or ``@workflow``-registered) per source item.

A route table is evaluated for each source item — a file path or a URL string —
and the winning workflow name plus its parameters are returned.

```yaml
ingest_routes:
  default:
    default: markdownize            # fallback workflow for unmatched items
    routes:
      - pathspec: "**/*.{ppt,pptx,odp,doc,docx}"
        workflow: office_via_pdf
        with:
          profile: best
      - pathspec: "https://www.youtube.com/**"
        workflow: youtube_transcript
      - pathspec: "https://**"
        workflow: web_page
```

!! Rule order is priority: the FIRST matching rule wins. Put the most specific
pathspecs first and catch-alls last — silently reordering rules silently
re-routes sources (route-table fingerprints invalidate caches when that happens).
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from prefect_yaml.routing import matches_pattern


class IngestRoutingError(ValueError):
    """Raised when no rule matches an item and the route table has no default."""


class IngestRule(BaseModel):
    """Route one source-item pattern to a workflow."""

    pathspec: str = Field(
        description=(
            "Gitignore-style (gitwildmatch) pattern matched against the item string — "
            "a file path or a URL (e.g. '**/*.pdf' or 'https://www.youtube.com/**'). "
            "Supports brace alternatives: '**/*.{pptx,ppt}'."
        )
    )
    workflow: str = Field(description="Target workflow name, optionally 'workflow/preset'")
    params: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("with", "params"),
        description="Extra parameters forwarded to the workflow (YAML key: 'with:')",
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    def matches(self, item: str) -> bool:
        """Return True when the path or URL string matches this rule's pathspec."""
        return matches_pattern(self.pathspec, item)


class IngestRouteTable(BaseModel):
    """Ordered workflow-routing rules with an optional default fallback.

    !! Rules are evaluated top-down and the first match wins — order IS priority.
    """

    name: str = Field(default="custom", description="Route table name")
    description: str | None = Field(default=None, description="Human-readable purpose of the table")
    default: str | None = Field(default=None, description="Workflow used when no rule matches")
    routes: list[IngestRule] = Field(default_factory=list, description="Ordered rules; first match wins")

    model_config = ConfigDict(extra="forbid")

    def select(self, item: str) -> tuple[str, dict[str, Any]]:
        """Return the ``(workflow, params)`` for the first rule matching *item*.

        Falls back to the ``default`` workflow (empty params) when no rule
        matches; raises when there is no default either.
        """
        for rule in self.routes:
            if rule.matches(item):
                return rule.workflow, dict(rule.params)
        if self.default:
            return self.default, {}
        raise IngestRoutingError(f"No route matches '{item}' and the route table has no default")

    def fingerprint(self) -> str:
        """Return a stable fingerprint of the table, used as a cache code-version."""
        parts = [f"default:{self.default}"] + [
            f"{r.pathspec}->{r.workflow}:{json.dumps(r.params, sort_keys=True)}" for r in self.routes
        ]
        return ";".join(parts)


def is_url(item: str) -> bool:
    """Return True when the source item is an http(s) URL rather than a filesystem path."""
    return item.startswith(("http://", "https://"))
