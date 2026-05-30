"""Unit tests for BAML Prefect flow helper functions.

These tests cover the pure-Python helpers in baml_flow.py that do not require
a live BAML client or Prefect runtime.  End-to-end flow tests (with stub
task submission) live in tests/integration_tests/test_baml_prefect_flow_integration.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from genai_tk.workflow.prefect.flows.baml_flow import (
    BamlExtractionManifest,
    BamlExtractionManifestEntry,
    _find_existing_manifest,
    _iter_supported_files,
    _save_manifest,
)

# ---------------------------------------------------------------------------
# _iter_supported_files
# ---------------------------------------------------------------------------


class TestIterSupportedFiles:
    def test_yields_markdown_files(self, tmp_path: Path) -> None:
        md = Path(tmp_path / "doc.md")
        md.write_text("# hello")
        result = list(_iter_supported_files([md]))
        assert len(result) == 1
        assert result[0].name == "doc.md"

    def test_yields_pdf_files(self, tmp_path: Path) -> None:
        pdf = Path(tmp_path / "report.pdf")
        pdf.write_bytes(b"%PDF")
        result = list(_iter_supported_files([pdf]))
        assert len(result) == 1

    def test_skips_unsupported_extensions(self, tmp_path: Path) -> None:
        txt = Path(tmp_path / "notes.txt")
        txt.write_text("plain text")
        result = list(_iter_supported_files([txt]))
        assert result == []

    def test_mixed_files_only_yields_supported(self, tmp_path: Path) -> None:
        files = []
        for name in ("a.md", "b.txt", "c.pdf", "d.docx", "e.markdown"):
            p = Path(tmp_path / name)
            p.write_bytes(b"x")
            files.append(p)
        result = list(_iter_supported_files(files))
        names = {r.name for r in result}
        assert names == {"a.md", "c.pdf", "e.markdown"}

    def test_empty_input_returns_empty(self) -> None:
        assert list(_iter_supported_files([])) == []


# ---------------------------------------------------------------------------
# BamlExtractionManifest — serialisation round-trip
# ---------------------------------------------------------------------------


class TestBamlExtractionManifest:
    def test_empty_manifest_serialises(self) -> None:
        manifest = BamlExtractionManifest(function_name="ExtractFoo", config_name="default")
        data = json.loads(manifest.model_dump_json())
        assert data["function_name"] == "ExtractFoo"
        assert data["entries"] == {}

    def test_manifest_with_entry_round_trips(self) -> None:
        entry = BamlExtractionManifestEntry(
            source_hash="abc123",
            output_path="results/doc.json",
            processed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        manifest = BamlExtractionManifest(
            function_name="ExtractBar",
            config_name="default",
            entries={"docs/doc.md": entry},
        )
        recovered = BamlExtractionManifest.model_validate_json(manifest.model_dump_json())
        assert recovered.entries["docs/doc.md"].source_hash == "abc123"
        assert recovered.entries["docs/doc.md"].output_path == "results/doc.json"

    def test_manifest_llm_field_optional(self) -> None:
        manifest = BamlExtractionManifest(function_name="F", config_name="c")
        assert manifest.llm is None
        manifest2 = BamlExtractionManifest(function_name="F", config_name="c", llm="gpt4")
        assert manifest2.llm == "gpt4"


# ---------------------------------------------------------------------------
# _save_manifest / _find_existing_manifest
# ---------------------------------------------------------------------------


class TestManifestIO:
    def _make_manifest(self) -> BamlExtractionManifest:
        return BamlExtractionManifest(
            function_name="ExtractTest",
            config_name="default",
            llm="gpt4",
            entries={
                "doc.md": BamlExtractionManifestEntry(
                    source_hash="deadbeef",
                    output_path="out/doc.json",
                )
            },
        )

    def test_save_creates_json_file(self, tmp_path: Path) -> None:
        manifest = self._make_manifest()
        path = Path(tmp_path / "manifest.json")
        _save_manifest(manifest, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["function_name"] == "ExtractTest"

    def test_find_loads_saved_manifest(self, tmp_path: Path) -> None:
        manifest = self._make_manifest()
        path = Path(tmp_path / "manifest.json")
        _save_manifest(manifest, path)

        loaded, loaded_path = _find_existing_manifest(Path(tmp_path), "ExtractTest", "default", "gpt4")
        assert loaded is not None
        assert loaded.function_name == "ExtractTest"
        assert "doc.md" in loaded.entries
        assert loaded_path == path

    def test_find_returns_none_when_missing(self, tmp_path: Path) -> None:
        loaded, path = _find_existing_manifest(Path(tmp_path), "F", "c", "llm")
        assert loaded is None
        assert path is None

    def test_find_returns_none_on_corrupt_json(self, tmp_path: Path) -> None:
        bad = Path(tmp_path / "manifest.json")
        bad.write_text("not valid json")
        loaded, path = _find_existing_manifest(Path(tmp_path), "F", "c", "llm")
        assert loaded is None
        assert path == bad

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        manifest = self._make_manifest()
        deep_path = Path(tmp_path / "a" / "b" / "manifest.json")
        _save_manifest(manifest, deep_path)
        assert deep_path.exists()
