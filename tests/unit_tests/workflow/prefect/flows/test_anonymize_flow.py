"""Unit tests for the ``anonymize_files`` Prefect flow.

Uses the real Presidio + Faker anonymization stack (local, no network) on
synthetic PII text.  Guarded by the ``nlp`` optional feature.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from genai_tk.extra.nlp import AnonymizationConfig, PresidioDetectorConfig
    from genai_tk.workflow.prefect.flows.anonymize_flow import (
        anonymize_file_task,
        anonymize_files_flow,
    )
except ImportError:  # pragma: no cover - depends on optional nlp extra
    pytest.skip("nlp feature (presidio) not installed", allow_module_level=True)

from genai_tk.workflow.flow_cache.manifest import ManifestCache

_PII_TEXT = "My name is John Smith. Contact me at john.doe@example.com for details."


def _config(seed: int = 42) -> AnonymizationConfig:
    """Build a deterministic anonymization config."""
    return AnonymizationConfig(
        detector=PresidioDetectorConfig(),
        faker_seed=seed,
    )


# ---------------------------------------------------------------------------
# anonymize_file_task
# ---------------------------------------------------------------------------


@pytest.mark.requires_feature("nlp")
@pytest.mark.fake_models
def test_anonymize_file_task_writes_anonymized_output(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    f = src / "secret.txt"
    f.write_text(_PII_TEXT, encoding="utf-8")

    result = anonymize_file_task(
        source_path=str(f),
        output_dir=str(out),
        root_dir=str(src),
        config=_config(),
        save_mapping=False,
    )

    assert result is not None
    relative_path, source_hash, mapping_path = result
    assert relative_path == "secret.txt"
    assert source_hash  # non-empty hash
    assert mapping_path is None

    written = out / "secret.txt"
    assert written.exists()
    anonymized = written.read_text(encoding="utf-8")
    # PII must be replaced
    assert "john.doe@example.com" not in anonymized
    assert "John Smith" not in anonymized
    assert anonymized != _PII_TEXT
    # mapping sidecar not written
    assert not (out / "secret.txt.mapping.json").exists()


@pytest.mark.requires_feature("nlp")
@pytest.mark.fake_models
def test_anonymize_file_task_save_mapping_writes_sidecar(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    f = src / "secret.md"
    f.write_text(_PII_TEXT, encoding="utf-8")

    result = anonymize_file_task(
        source_path=str(f),
        output_dir=str(out),
        root_dir=str(src),
        config=_config(),
        save_mapping=True,
    )

    assert result is not None
    _, _, mapping_path = result
    assert mapping_path is not None
    sidecar = out / "secret.md.mapping.json"
    assert sidecar.exists()


@pytest.mark.requires_feature("nlp")
@pytest.mark.fake_models
def test_anonymize_file_task_unreadable_returns_none(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = anonymize_file_task(
        source_path=str(tmp_path / "missing.txt"),
        output_dir=str(out),
        root_dir=str(tmp_path),
        config=_config(),
    )
    assert result is None


@pytest.mark.requires_feature("nlp")
@pytest.mark.fake_models
def test_anonymize_file_task_preserves_subdir_structure(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "docs").mkdir(parents=True)
    out = tmp_path / "out"
    f = src / "docs" / "note.txt"
    f.write_text(_PII_TEXT, encoding="utf-8")

    result = anonymize_file_task(
        source_path=str(f),
        output_dir=str(out),
        root_dir=str(src),
        config=_config(),
    )
    assert result is not None
    relative_path, _, _ = result
    assert relative_path == str(Path("docs") / "note.txt")
    assert (out / "docs" / "note.txt").exists()


# ---------------------------------------------------------------------------
# anonymize_files_flow — smoke runs
# ---------------------------------------------------------------------------


@pytest.mark.requires_feature("nlp")
@pytest.mark.fake_models
def test_anonymize_files_flow_anonymizes_and_records_manifest(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text(_PII_TEXT, encoding="utf-8")
    (src / "b.md").write_text("Another note about Jane Doe.", encoding="utf-8")
    out = tmp_path / "out"

    result = anonymize_files_flow(
        base_dir=str(src),
        output_dir=str(out),
        faker_seed=42,
    )

    assert isinstance(result, ManifestCache)
    assert (out / "manifest.json").exists()
    # both files recorded (manifest keys are absolute source paths)
    assert any(k.endswith("a.txt") for k in result.records)
    assert any(k.endswith("b.md") for k in result.records)
    assert (out / "a.txt").exists()
    assert (out / "b.md").exists()
    anonymized = (out / "a.txt").read_text(encoding="utf-8")
    assert "john.doe@example.com" not in anonymized


@pytest.mark.requires_feature("nlp")
@pytest.mark.fake_models
def test_anonymize_files_flow_no_files_returns_empty_cache(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "data.csv").write_text("no,text,here", encoding="utf-8")
    out = tmp_path / "out"

    result = anonymize_files_flow(base_dir=str(src), output_dir=str(out))
    assert isinstance(result, ManifestCache)
    assert result.records == {}
    assert not out.exists()


@pytest.mark.requires_feature("nlp")
@pytest.mark.fake_models
def test_anonymize_files_flow_skips_cached_without_processing(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    f = src / "a.txt"
    f.write_text(_PII_TEXT, encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()

    from genai_tk.utils.hashing import buffer_digest

    cache = ManifestCache.load(out / "manifest.json")
    cache.record_success(
        key=str(f),
        fingerprint=buffer_digest(f.read_bytes()),
        outputs={"output_path": "a.txt"},
    )
    cache.save(out / "manifest.json")

    result = anonymize_files_flow(base_dir=str(src), output_dir=str(out), force=False)
    assert str(f) in result.records
    # no anonymized output file written (only the pre-existing manifest)
    assert not (out / "a.txt").exists()
