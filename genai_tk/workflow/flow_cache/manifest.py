"""Shared manifest-based incremental processing cache.

Consolidates the per-flow manifest implementations that previously existed
separately in ``ppt2pdf_flow``, ``markdownize_flow``, ``baml_flow``,
``anonymize_flow``, and ``rag_flow``.

Usage
-----

.. code-block:: python

    from pathlib import Path
    from genai_tk.workflow.flow_cache.manifest import ManifestCache

    cache = ManifestCache.load(Path("output/manifest.json"))

    for file in files_to_process:
        content_hash = compute_hash(file)
        if cache.is_fresh(str(file), fingerprint=content_hash):
            continue  # already processed and unchanged

        output = process(file)
        cache.record_success(
            key=str(file),
            fingerprint=content_hash,
            outputs={"output_path": str(output)},
        )

    cache.save(Path("output/manifest.json"))

For BAML extraction or other flows that also depend on a prompt/schema
version, pass ``code_version``:

.. code-block:: python

    cache.is_fresh(str(file), fingerprint=content_hash, code_version=schema_fp)
    cache.record_success(str(file), content_hash, code_version=schema_fp, outputs={...})
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class CacheRecord(BaseModel):
    """A single processed-item record stored in the manifest."""

    key: str
    fingerprint: str
    status: str = "ok"
    code_version: str | None = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ManifestCache(BaseModel):
    """An in-memory manifest cache backed by a JSON file.

    Instances are normally created via :meth:`load` rather than the constructor.
    """

    records: dict[str, CacheRecord] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def is_fresh(
        self,
        key: str,
        *,
        fingerprint: str,
        force: bool = False,
        code_version: str | None = None,
    ) -> bool:
        """Return ``True`` if the item can be skipped (already processed and up-to-date).

        Args:
            key: Unique item key (typically the file path string).
            fingerprint: Content hash or other staleness signal for the item.
            force: When ``True``, always return ``False`` (force reprocessing).
            code_version: Optional secondary fingerprint (e.g., schema or prompt hash).
                          If provided, a mismatch invalidates the cache regardless of
                          the primary ``fingerprint``.

        Returns:
            ``True`` when the item is cached and fresh; ``False`` when processing is needed.
        """
        if force:
            return False
        record = self.records.get(key)
        if record is None:
            return False
        if record.fingerprint != fingerprint:
            return False
        if code_version is not None and record.code_version != code_version:
            return False
        return True

    def get_output(self, key: str, output_field: str) -> Any:
        """Return a specific output value from a cached record, or ``None``."""
        record = self.records.get(key)
        return record.outputs.get(output_field) if record else None

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def record_success(
        self,
        key: str,
        fingerprint: str,
        *,
        outputs: dict[str, Any] | None = None,
        code_version: str | None = None,
    ) -> None:
        """Record a successfully processed item.

        Args:
            key: Unique item key.
            fingerprint: Content hash at time of processing.
            outputs: Optional dict of output metadata (paths, stats, etc.).
            code_version: Optional schema/prompt fingerprint.
        """
        self.records[key] = CacheRecord(
            key=key,
            fingerprint=fingerprint,
            status="ok",
            code_version=code_version,
            outputs=outputs or {},
        )

    def record_failure(self, key: str, fingerprint: str, error: str = "") -> None:
        """Record a failed item so retries can be tracked."""
        self.records[key] = CacheRecord(
            key=key,
            fingerprint=fingerprint,
            status="error",
            outputs={"error": error} if error else {},
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None, *, warn_on_error: bool = True) -> ManifestCache:
        """Load from a JSON file, returning an empty cache if the file does not exist.

        Args:
            path: Path to the manifest JSON file.
            warn_on_error: Log a warning instead of raising if the file is corrupt.

        Returns:
            A populated or fresh :class:`ManifestCache`.
        """
        if path is None or not path.exists():
            return cls()
        try:
            text = path.read_text(encoding="utf-8")
            return cls.model_validate_json(text)
        except Exception as exc:
            if warn_on_error:
                logger.warning("Failed to load manifest from {}: {}. Starting fresh.", path, exc)
                return cls()
            raise

    def save(self, path: Path) -> None:
        """Persist the cache to a JSON file.

        Args:
            path: Destination path (parent directories are created automatically).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        logger.debug("Saved manifest cache ({} records) to {}", len(self.records), path)
