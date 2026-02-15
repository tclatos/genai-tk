"""Integration-style tests for BAML Prefect flows with deterministic stubs."""

from datetime import datetime, timezone

import pytest
from pydantic import BaseModel
from upath import UPath

import genai_tk.extra.structured.baml_prefect_flow as mod
from genai_tk.extra.structured.baml_prefect_flow import (
    BamlExtractionManifest,
    BamlExtractionManifestEntry,
    baml_single_input_flow,
    baml_structured_extraction_flow,
)


class _FakeFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class _DummyModel(BaseModel):
    value: str


@pytest.mark.integration
@pytest.mark.fake_models
def test_baml_structured_extraction_flow_writes_manifest(tmp_path, monkeypatch) -> None:
    """Ensure structured extraction flow writes manifest under model directory."""
    docs_dir = tmp_path / "docs"
    output_dir = tmp_path / "structured"
    docs_dir.mkdir()
    output_dir.mkdir()

    md_file = docs_dir / "example.md"
    md_file.write_text("# Example", encoding="utf-8")

    def fake_submit(file_info, function_name, config_name, llm, structured_root, root_dir):
        entry = BamlExtractionManifestEntry(
            source_hash=file_info.content_hash,
            output_path="example.json",
            processed_at=datetime.now(timezone.utc),
        )
        return _FakeFuture((str(file_info.path), entry, _DummyModel.__name__))

    monkeypatch.setattr(mod._process_single_file_task, "submit", fake_submit)
    monkeypatch.setattr(mod, "resolve_files", lambda *args, **kwargs: [str(p) for p in docs_dir.iterdir()])

    manifest = baml_structured_extraction_flow.fn(
        root_dir=str(docs_dir),
        output_dir=str(output_dir),
        recursive=False,
        batch_size=1,
        force=False,
        function_name="ExtractDummy",
        config_name="default",
        llm=None,
    )

    assert isinstance(manifest, BamlExtractionManifest)
    assert len(manifest.entries) == 1

    model_dir = UPath(output_dir) / _DummyModel.__name__
    manifest_path = model_dir / "manifest.json"
    assert manifest_path.exists()


@pytest.mark.integration
@pytest.mark.fake_models
def test_baml_single_input_flow_updates_manifest(tmp_path, monkeypatch) -> None:
    """Ensure single input flow updates manifest when output is configured."""
    output_dir = tmp_path / "structured"
    output_dir.mkdir()

    def fake_submit(
        *, input_text, function_name, config_name, llm, output_dir, output_file, input_hash, force, existing_manifest
    ):
        result = _DummyModel(value="ok")
        model_name = _DummyModel.__name__
        relative_output_path = f"{model_name}/{output_file}"
        return _FakeFuture((result, model_name, relative_output_path))

    monkeypatch.setattr(mod._process_single_input_task, "submit", fake_submit)

    result, model_name = baml_single_input_flow.fn(
        input_text="Hello",
        function_name="ExtractDummy",
        config_name="default",
        llm=None,
        output_dir=str(output_dir),
        output_file="result.json",
        force=False,
    )

    assert isinstance(result, BaseModel)
    assert model_name == _DummyModel.__name__

    manifest_path = UPath(output_dir) / _DummyModel.__name__ / "manifest.json"
    assert manifest_path.exists()
