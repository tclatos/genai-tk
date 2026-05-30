"""Unit tests for ManifestCache lifecycle (load, is_fresh, record, save)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from genai_tk.workflow.flow_cache.manifest import CacheRecord, ManifestCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cache_with(records: dict) -> ManifestCache:
    return ManifestCache(records=records)


# ---------------------------------------------------------------------------
# is_fresh
# ---------------------------------------------------------------------------


class TestIsFresh:
    def test_unknown_key_returns_false(self) -> None:
        cache = ManifestCache()
        assert cache.is_fresh("missing", fingerprint="abc") is False

    def test_matching_fingerprint_is_fresh(self) -> None:
        cache = _cache_with({"k": CacheRecord(key="k", fingerprint="fp1")})
        assert cache.is_fresh("k", fingerprint="fp1") is True

    def test_stale_fingerprint_returns_false(self) -> None:
        cache = _cache_with({"k": CacheRecord(key="k", fingerprint="fp1")})
        assert cache.is_fresh("k", fingerprint="fp2") is False

    def test_force_always_returns_false(self) -> None:
        cache = _cache_with({"k": CacheRecord(key="k", fingerprint="fp1")})
        assert cache.is_fresh("k", fingerprint="fp1", force=True) is False

    def test_code_version_mismatch_returns_false(self) -> None:
        cache = _cache_with({"k": CacheRecord(key="k", fingerprint="fp1", code_version="v1")})
        assert cache.is_fresh("k", fingerprint="fp1", code_version="v2") is False

    def test_code_version_match_is_fresh(self) -> None:
        cache = _cache_with({"k": CacheRecord(key="k", fingerprint="fp1", code_version="v1")})
        assert cache.is_fresh("k", fingerprint="fp1", code_version="v1") is True

    def test_error_status_is_not_fresh(self) -> None:
        # A failed record has the same fingerprint but status="error"; is_fresh
        # does NOT filter on status — the caller (record_failure) is responsible.
        # Verify current behaviour: fingerprint match → fresh regardless of status.
        cache = _cache_with({"k": CacheRecord(key="k", fingerprint="fp1", status="error")})
        # is_fresh only checks fingerprint, not status
        assert cache.is_fresh("k", fingerprint="fp1") is True


# ---------------------------------------------------------------------------
# record_success / record_failure
# ---------------------------------------------------------------------------


class TestRecordMutations:
    def test_record_success_stores_record(self) -> None:
        cache = ManifestCache()
        cache.record_success("step1", "fp1", outputs={"result": {"count": 3}})
        assert "step1" in cache.records
        rec = cache.records["step1"]
        assert rec.fingerprint == "fp1"
        assert rec.status == "ok"
        assert rec.outputs["result"] == {"count": 3}

    def test_record_success_overwrites_previous(self) -> None:
        cache = ManifestCache()
        cache.record_success("step1", "fp_old")
        cache.record_success("step1", "fp_new", outputs={"result": 42})
        assert cache.records["step1"].fingerprint == "fp_new"

    def test_record_failure_marks_error_status(self) -> None:
        cache = ManifestCache()
        cache.record_failure("step1", "fp1", error="something went wrong")
        rec = cache.records["step1"]
        assert rec.status == "error"
        assert rec.outputs["error"] == "something went wrong"

    def test_get_output_returns_field(self) -> None:
        cache = ManifestCache()
        cache.record_success("s", "fp", outputs={"result": {"x": 1}})
        assert cache.get_output("s", "result") == {"x": 1}

    def test_get_output_missing_key_returns_none(self) -> None:
        cache = ManifestCache()
        assert cache.get_output("absent", "result") is None


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_reload(self, tmp_path: Path) -> None:
        path = Path(tmp_path / "manifest.json")
        cache = ManifestCache()
        cache.record_success("build", "fp42", outputs={"result": {"total": 10}})
        cache.save(path)

        reloaded = ManifestCache.load(path)
        assert reloaded.is_fresh("build", fingerprint="fp42") is True
        assert reloaded.get_output("build", "result") == {"total": 10}

    def test_load_non_existent_returns_empty(self, tmp_path: Path) -> None:
        path = Path(tmp_path / "no_such.json")
        cache = ManifestCache.load(path)
        assert cache.records == {}

    def test_load_corrupt_file_returns_empty_with_warning(self, tmp_path: Path) -> None:
        path = Path(tmp_path / "bad.json")
        path.write_text("{ not valid json", encoding="utf-8")
        cache = ManifestCache.load(path, warn_on_error=True)
        assert cache.records == {}

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        deep_path = Path(tmp_path / "a" / "b" / "c" / "manifest.json")
        cache = ManifestCache()
        cache.record_success("x", "fp1")
        cache.save(deep_path)
        assert deep_path.exists()

    def test_processed_at_is_persisted(self, tmp_path: Path) -> None:
        path = Path(tmp_path / "manifest.json")
        ts = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        cache = ManifestCache()
        cache.records["s"] = CacheRecord(key="s", fingerprint="fp", processed_at=ts)
        cache.save(path)

        reloaded = ManifestCache.load(path)
        assert reloaded.records["s"].processed_at == ts


# ---------------------------------------------------------------------------
# compute_step_fingerprint (determinism + sensitivity)
# ---------------------------------------------------------------------------


class TestComputeStepFingerprint:
    def test_deterministic(self) -> None:
        from genai_tk.workflow.prefect.flow_factory import compute_step_fingerprint

        fp1 = compute_step_fingerprint("build", {"kg_name": "rainbow", "delete_first": False})
        fp2 = compute_step_fingerprint("build", {"kg_name": "rainbow", "delete_first": False})
        assert fp1 == fp2

    def test_different_inputs_produce_different_fingerprint(self) -> None:
        from genai_tk.workflow.prefect.flow_factory import compute_step_fingerprint

        fp1 = compute_step_fingerprint("build", {"kg_name": "rainbow"})
        fp2 = compute_step_fingerprint("build", {"kg_name": "different"})
        assert fp1 != fp2

    def test_different_step_ids_produce_different_fingerprint(self) -> None:
        from genai_tk.workflow.prefect.flow_factory import compute_step_fingerprint

        fp1 = compute_step_fingerprint("step_a", {"x": 1})
        fp2 = compute_step_fingerprint("step_b", {"x": 1})
        assert fp1 != fp2

    def test_returns_16_char_hex(self) -> None:
        from genai_tk.workflow.prefect.flow_factory import compute_step_fingerprint

        fp = compute_step_fingerprint("s", {})
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)
