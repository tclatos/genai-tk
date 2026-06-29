"""Bridge loguru → Python stdlib logging so messages appear in the Prefect UI.

Prefect captures Python :mod:`logging` records emitted during a flow/task run
and persists them in its log store (visible in the Prefect UI log viewer).
Because genai-tk uses :mod:`loguru` (a separate logging library) the rich
application-level messages would otherwise only reach the console.

Usage — call once at the start of any Prefect ``@flow`` or ``@task`` body::

    from genai_tk.utils.prefect_logging import install_loguru_prefect_bridge


    @flow
    def my_flow():
        install_loguru_prefect_bridge()
        ...

The function is **idempotent** — it installs exactly one extra sink per
process lifetime, regardless of how many flows or tasks call it.
"""

from __future__ import annotations

import logging
from typing import Any

# Module-level sentinel: the loguru sink ID returned by logger.add(), or None
# when the bridge has not been installed yet.
_bridge_id: int | None = None

# Mapping from loguru level names to Python stdlib log levels.
# "SUCCESS" is loguru-specific; it maps to INFO since stdlib has no SUCCESS.
_LEVEL_MAP: dict[str, int] = {
    "TRACE": logging.DEBUG,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "SUCCESS": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def install_loguru_prefect_bridge() -> None:
    """Route loguru messages through Python stdlib logging into the Prefect UI.

    Prefect installs its ``PrefectLogHandler`` on the Python root logger when a
    flow/task run context is active.  Any ``logging.getLogger(name).log(...)``
    call therefore reaches Prefect's log store automatically.

    This function adds a loguru *sink* that forwards every loguru record through
    the matching Python :class:`logging.Logger` (keyed by the loguru source
    module name).  The Prefect handler then captures those records and makes
    them visible in the Prefect UI log viewer, alongside Prefect's own engine
    messages.

    Idempotent — safe to call multiple times; the extra sink is installed only
    once per process.
    """
    global _bridge_id
    if _bridge_id is not None:
        return

    from loguru import logger

    def _sink(message: Any) -> None:
        record = message.record
        level_name: str = record["level"].name
        std_level = _LEVEL_MAP.get(level_name, logging.INFO)

        # Use the loguru source module as the Python logger name so the Prefect
        # UI shows the correct origin (e.g. "markdownize_flow", "rfq_merge_step").
        logger_name: str = record["name"] or "genai_tk"

        # Build a stdlib LogRecord from the loguru record fields.
        # We then call logging.root.handle() *directly* — this bypasses all
        # level-threshold checks (root logger defaults to WARNING which would
        # silently drop INFO records before they reach Prefect's handler).
        lr = logging.LogRecord(
            name=logger_name,
            level=std_level,
            pathname=str(record["file"].path),
            lineno=record["line"],
            msg=record["message"],
            args=(),
            exc_info=None,
        )
        # func name for Prefect's log display
        lr.funcName = record["function"]
        logging.root.handle(lr)

    from genai_tk.utils.logger_factory import logging_config

    bridge_level = logging_config().level

    _bridge_id = logger.add(
        _sink,
        # Use a simple format — the sink formats each loguru message object
        # before passing it to _sink; the format string is required but its
        # output is not used (we read record["message"] directly).
        format="{message}",
        level=bridge_level,
    )
