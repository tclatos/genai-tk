"""Rule-based workflow routing (the Ingest Router).

Pathspec rules select a whole YAML (or ``@workflow``-registered) workflow per
source item — files by path pattern, URLs by URL pattern. See ``models`` for
the rule semantics (first match wins) and ``dispatcher`` for the runtime.
"""

from genai_tk.workflow.routing.dispatcher import ingest_dispatch_flow, ingest_dispatch_step
from genai_tk.workflow.routing.loader import (
    DEFAULT_ROUTES,
    get_ingest_routes,
    list_ingest_routes,
    validate_ingest_routes,
)
from genai_tk.workflow.routing.models import IngestRouteTable, IngestRule, IngestRoutingError, is_url

__all__ = [
    "DEFAULT_ROUTES",
    "IngestRouteTable",
    "IngestRoutingError",
    "IngestRule",
    "get_ingest_routes",
    "ingest_dispatch_flow",
    "ingest_dispatch_step",
    "is_url",
    "list_ingest_routes",
    "validate_ingest_routes",
]
