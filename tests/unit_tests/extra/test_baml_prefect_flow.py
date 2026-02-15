"""Tests for the Prefect-based BAML structured extraction flow."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel
from upath import UPath

from genai_tk.extra.structured import baml_prefect_flow as mod


class _DummyConfig:
    """Minimal config stub exposing get_dir_path for tests."""

    def __init__(self, data_root: UPath) -> None:
        self._data_root = data_root

    def get_dir_path(self, key: str, create_if_not_exists: bool = False) -> UPath:
        assert key == "paths.data_root"
        if create_if_not_exists:
            self._data_root.mkdir(parents=True, exist_ok=True)
        return self._data_root


class _DummyModel(BaseModel):
    value: str


def _patch_flow_dependencies(
    tmp_path, monkeypatch
) -> tuple[list[tuple[str, dict[str, Any], str, str | None]], Callable[..., Any]]:
    """Patch global_config and baml_invoke for the flow tests.

    Returns a tuple (calls, fake_baml_invoke) so tests can assert call counts.
    """

    data_root = UPath(tmp_path / "data_root")
    cfg = _DummyConfig(data_root)

    # Patch global_config to return our dummy config pointing to the temp root.
    monkeypatch.setattr(mod, "global_config", lambda: cfg)

    calls: list[tuple[str, dict[str, Any], str, str | None]] = []

    async def fake_baml_invoke(
        function_name: str, params: dict[str, Any], config_name: str, llm: str | None
    ) -> _DummyModel:
        calls.append((function_name, params, config_name, llm))
        return _DummyModel(value="ok")

    monkeypatch.setattr(mod, "baml_invoke", fake_baml_invoke)

    return calls, fake_baml_invoke


@pytest.mark.unit
def test_flow_writes_json_and_manifest_under_model_dir(tmp_path, monkeypatch) -> None:
    """The flow should write JSON outputs and manifest under structured/<model_name>/."""

    calls, _ = _patch_flow_dependencies(tmp_path, monkeypatch)

    # Create a single markdown file to process.
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    md_file = docs_dir / "example.md"
    md_file.write_text("# Example", encoding="utf-8")

    manifest = mod.baml_structured_extraction_flow(
        root_dir=str(docs_dir),
        output_dir="${paths.data_root}/structured",
        recursive=False,
        batch_size=2,
        force=False,
        function_name="ExtractDummy",
        config_name="default",
        llm=None,
    )

    # Verify that BAML was invoked once and manifest contains model_name.
    assert len(calls) == 1
    assert manifest.model_name == _DummyModel.__name__

    data_root = UPath(tmp_path / "data_root")
    structured_root = data_root / "structured"
    model_dir = structured_root / _DummyModel.__name__

    # Manifest must be stored under structured/<model_name>/.
    manifest_path = model_dir / "manifest.json"
    assert manifest_path.exists()

    # At least one JSON output must be present in the model directory.
    json_files = list(model_dir.glob("*.json"))
    assert json_files, "Expected at least one JSON file in the model directory"


@pytest.mark.unit
def test_flow_uses_manifest_to_skip_unchanged_files(tmp_path, monkeypatch) -> None:
    """A second run on unchanged files should not re-invoke BAML when force is False."""

    calls, _ = _patch_flow_dependencies(tmp_path, monkeypatch)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    md_file = docs_dir / "example.md"
    md_file.write_text("# Example", encoding="utf-8")

    # First run populates manifest and outputs.
    manifest1 = mod.baml_structured_extraction_flow(
        root_dir=str(docs_dir),
        output_dir="${paths.data_root}/structured",
        recursive=False,
        batch_size=2,
        force=False,
        function_name="ExtractDummy",
        config_name="default",
        llm=None,
    )

    assert manifest1.model_name == _DummyModel.__name__
    assert len(calls) == 1

    # Second run on the same content should use the manifest to skip.
    manifest2 = mod.baml_structured_extraction_flow(
        root_dir=str(docs_dir),
        output_dir="${paths.data_root}/structured",
        recursive=False,
        batch_size=2,
        force=False,
        function_name="ExtractDummy",
        config_name="default",
        llm=None,
    )

    # No additional BAML invocations expected.
    assert len(calls) == 1
    assert manifest2.model_name == _DummyModel.__name__

    data_root = UPath(tmp_path / "data_root")
    structured_root = data_root / "structured"
    model_dir = structured_root / _DummyModel.__name__

    # Ensure the manifest still lives under the model directory after rerun.
    manifest_path = model_dir / "manifest.json"
    assert manifest_path.exists()
