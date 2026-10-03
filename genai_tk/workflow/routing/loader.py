"""Named ingest-route-table loader.

Route tables resolve by name from two sources, project config taking precedence:

1. The ``ingest_routes`` section of the project configuration (any YAML file
   auto-scanned under ``config/``).
2. Built-in tables packaged with genai-tk in ``default_config/ingest_routes.yaml``.
"""

from __future__ import annotations

import yaml
from loguru import logger

from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.workflow.routing.models import IngestRouteTable

DEFAULT_ROUTES = "default"


def _builtin_tables() -> dict[str, IngestRouteTable]:
    """Load built-in route tables packaged with genai-tk."""
    from importlib.resources import files as _pkg_files

    try:
        src = _pkg_files("genai_tk") / "default_config" / "ingest_routes.yaml"
        yaml_text = src.read_text(encoding="utf-8")
    except Exception:
        from pathlib import Path

        config_path = Path(__file__).parent.parent.parent / "default_config" / "ingest_routes.yaml"
        yaml_text = config_path.read_text(encoding="utf-8")

    data = yaml.safe_load(yaml_text) or {}
    tables: dict[str, IngestRouteTable] = {}
    for table_name, raw in data.get("ingest_routes", {}).items():
        raw = {**raw, "name": table_name} if isinstance(raw, dict) else raw
        tables[table_name] = IngestRouteTable.model_validate(raw)
    return tables


def get_ingest_routes(name: str = DEFAULT_ROUTES) -> IngestRouteTable:
    """Resolve a route table by name from project configuration or built-ins."""
    configured = global_config().section_dict("ingest_routes", IngestRouteTable, inject_name=True)
    if name in configured:
        return configured[name]
    builtin = _builtin_tables()
    if name in builtin:
        return builtin[name]
    available = sorted(set(configured) | set(builtin))
    raise KeyError(f"Unknown ingest route table '{name}'. Available: {available}")


def list_ingest_routes() -> list[str]:
    """Return all available route-table names (project config + built-ins)."""
    return sorted({*_builtin_tables(), *global_config().section_dict("ingest_routes", IngestRouteTable, inject_name=True)})


def validate_ingest_routes() -> dict[str, str]:
    """Validate every configured route table; return name → fingerprint (raises on invalid)."""
    tables = {**_builtin_tables(), **global_config().section_dict("ingest_routes", IngestRouteTable, inject_name=True)}
    for name, table in tables.items():
        for rule in table.routes:
            if not rule.pathspec or not rule.workflow:
                raise ValueError(f"Route table '{name}': rules need both 'pathspec' and 'workflow'")
        logger.debug("Route table '{}' OK: {}", name, table.fingerprint())
    return {name: table.fingerprint() for name, table in tables.items()}
