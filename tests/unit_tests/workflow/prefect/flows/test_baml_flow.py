"""Unit tests for the BAML Prefect flows.

Covers the helpers not already exercised by ``tests/unit_tests/extra`` and runs
the flows through the Prefect test harness with ``.submit`` stubbed out (BAML
invocation hits an LLM API — a true external boundary).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

try:
    from genai_tk.workflow.prefect.flows.baml_flow import (
        BamlExtractionManifest,
        _chunked,
        _compute_hash,
        _FileToProcess,
        _prepare_files,
        _process_single_file_task,
        _process_single_input_task,
        baml_single_input_flow,
        baml_structured_extraction_flow,
    )
except ImportError:  # pragma: no cover - depends on optional baml extra
    pytest.skip("baml feature not installed", allow_module_level=True)

import genai_tk.workflow.prefect.flows.baml_flow as baml_module
from genai_tk.workflow.flow_cache.manifest import ManifestCache


class _FakeFuture:
    """Stand-in for a Prefect future returned by ``task.submit``."""

    def __init__(self, result: Any) -> None:
        self._result = result

    def result(self) -> Any:
        return self._result


class _DummyModel(BaseModel):
    """Sample Pydantic model returned by the stubbed BAML task."""

    value: str


# ---------------------------------------------------------------------------
# _compute_hash
# ---------------------------------------------------------------------------


def test_compute_hash_is_deterministic() -> None:
    assert _compute_hash(b"abc") == _compute_hash(b"abc")


def test_compute_hash_differs_for_different_content() -> None:
    assert _compute_hash(b"abc") != _compute_hash(b"abd")


# ---------------------------------------------------------------------------
# _chunked
# ---------------------------------------------------------------------------


def test_chunked_splits_into_batches() -> None:
    assert list(_chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_chunked_zero_size_returns_single_batch() -> None:
    assert list(_chunked([1, 2, 3], 0)) == [[1, 2, 3]]


def test_chunked_empty() -> None:
    assert list(_chunked([], 3)) == []


# ---------------------------------------------------------------------------
# _prepare_files
# ---------------------------------------------------------------------------


def test_prepare_files_new_markdown_queued(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# hello", encoding="utf-8")
    cache = ManifestCache()

    to_process, skipped = _prepare_files([md], cache, schema_fp=None, force=False)
    assert len(to_process) == 1
    assert skipped == 0
    item = to_process[0]
    assert item.is_pdf is False
    assert item.content_bytes is None
    assert item.content_text == "# hello"


def test_prepare_files_pdf_sets_bytes_and_empty_text(tmp_path: Path) -> None:
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    cache = ManifestCache()

    to_process, _ = _prepare_files([pdf], cache, schema_fp=None, force=False)
    assert len(to_process) == 1
    item = to_process[0]
    assert item.is_pdf is True
    assert item.content_bytes == b"%PDF-1.4 fake"
    assert item.content_text == ""


def test_prepare_files_fresh_file_skipped(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# hello", encoding="utf-8")
    cache = ManifestCache()
    cache.record_success(
        key=str(md),
        fingerprint=_compute_hash(md.read_bytes()),
        code_version=None,
        outputs={"output_path": "doc.json"},
    )

    to_process, skipped = _prepare_files([md], cache, schema_fp=None, force=False)
    assert to_process == []
    assert skipped == 1


def test_prepare_files_schema_version_mismatch_reprocesses(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# hello", encoding="utf-8")
    cache = ManifestCache()
    cache.record_success(
        key=str(md),
        fingerprint=_compute_hash(md.read_bytes()),
        code_version="v1",
        outputs={"output_path": "doc.json"},
    )

    # same content fingerprint but different code_version → not fresh
    to_process, skipped = _prepare_files([md], cache, schema_fp="v2", force=False)
    assert len(to_process) == 1
    assert skipped == 0


def test_prepare_files_force_reprocesses_fresh(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# hello", encoding="utf-8")
    cache = ManifestCache()
    cache.record_success(
        key=str(md),
        fingerprint=_compute_hash(md.read_bytes()),
        code_version=None,
        outputs={},
    )

    to_process, skipped = _prepare_files([md], cache, schema_fp=None, force=True)
    assert len(to_process) == 1
    assert skipped == 0


def test_prepare_files_skips_unreadable(tmp_path: Path) -> None:
    cache = ManifestCache()
    to_process, skipped = _prepare_files([tmp_path / "missing.md"], cache, schema_fp=None, force=False)
    assert to_process == []
    assert skipped == 0


# ---------------------------------------------------------------------------
# _process_single_file_task — task body with faked baml_invoke
# ---------------------------------------------------------------------------
#
# ``baml_invoke`` wraps the BAML client → LLM API (a true external boundary).
# Faking it lets the real task logic (model detection, output writing, PDF
# branching) run end-to-end without a configured BAML project.


def _fake_baml_invoke_returning(result: Any) -> Any:
    """Return an async function substituting ``baml_invoke`` that yields *result*."""

    async def _invoke(function_name, params, config_name, llm):  # noqa: ANN001
        return result

    return _invoke


@pytest.mark.requires_feature("baml")
@pytest.mark.fake_models
def test_process_single_file_task_writes_model_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    src.mkdir()
    md = src / "example.md"
    md.write_text("# Example", encoding="utf-8")
    out = tmp_path / "out"

    monkeypatch.setattr(baml_module, "baml_invoke", _fake_baml_invoke_returning(_DummyModel(value="ok")))

    file_info = _FileToProcess(path=md, content_hash="h", content_text="# Example")
    source, relative_output, model_name = asyncio.run(
        _process_single_file_task.fn(file_info, "ExtractDummy", "default", None, str(out), str(src))
    )

    assert source == str(md)
    assert relative_output == "example.json"
    assert model_name == _DummyModel.__name__
    assert (out / _DummyModel.__name__ / "example.json").exists()


@pytest.mark.requires_feature("baml")
@pytest.mark.fake_models
def test_process_single_file_task_non_model_result_uses_function_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    md = src / "example.md"
    md.write_text("# Example", encoding="utf-8")
    out = tmp_path / "out"

    monkeypatch.setattr(baml_module, "baml_invoke", _fake_baml_invoke_returning({"k": "v"}))

    file_info = _FileToProcess(path=md, content_hash="h", content_text="# Example")
    source, relative_output, model_name = asyncio.run(
        _process_single_file_task.fn(file_info, "ExtractDummy", "default", None, str(out), str(src))
    )

    assert model_name is None
    # non-BaseModel results land under structured/<function_name>/
    assert (out / "ExtractDummy" / "example.json").exists()


@pytest.mark.requires_feature("baml")
@pytest.mark.fake_models
def test_process_single_file_task_baml_error_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from genai_tk.extra.structured.exceptions import StructuredOutputError

    src = tmp_path / "src"
    src.mkdir()
    md = src / "example.md"
    md.write_text("# Example", encoding="utf-8")
    out = tmp_path / "out"

    async def _raise(function_name, params, config_name, llm):  # noqa: ANN001
        raise StructuredOutputError("boom")

    monkeypatch.setattr(baml_module, "baml_invoke", _raise)

    file_info = _FileToProcess(path=md, content_hash="h", content_text="# Example")
    with pytest.raises(StructuredOutputError):
        asyncio.run(_process_single_file_task.fn(file_info, "ExtractDummy", "default", None, str(out), str(src)))


# ---------------------------------------------------------------------------
# _process_single_input_task — task body with faked baml_invoke
# ---------------------------------------------------------------------------


@pytest.mark.requires_feature("baml")
@pytest.mark.fake_models
def test_process_single_input_task_no_output_returns_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(baml_module, "baml_invoke", _fake_baml_invoke_returning(_DummyModel(value="ok")))

    result, model_name, output_path = asyncio.run(
        _process_single_input_task.fn(
            input_text="Hello",
            function_name="ExtractDummy",
            config_name="default",
            llm=None,
            output_dir=None,
            output_file=None,
            input_hash="h",
            force=False,
            existing_manifest=None,
        )
    )
    assert isinstance(result, _DummyModel)
    assert model_name == _DummyModel.__name__
    assert output_path is None


@pytest.mark.requires_feature("baml")
@pytest.mark.fake_models
def test_process_single_input_task_with_output_writes_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    monkeypatch.setattr(baml_module, "baml_invoke", _fake_baml_invoke_returning(_DummyModel(value="ok")))

    result, model_name, output_path = asyncio.run(
        _process_single_input_task.fn(
            input_text="Hello",
            function_name="ExtractDummy",
            config_name="default",
            llm=None,
            output_dir=str(out),
            output_file="result.json",
            input_hash="h",
            force=False,
            existing_manifest=None,
        )
    )
    assert isinstance(result, _DummyModel)
    assert model_name == _DummyModel.__name__
    assert output_path == f"{_DummyModel.__name__}/result.json"
    assert (out / _DummyModel.__name__ / "result.json").exists()


@pytest.mark.requires_feature("baml")
@pytest.mark.fake_models
def test_process_single_input_task_skips_via_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    out.mkdir()
    # pre-write an existing result that the manifest points to
    model_dir = out / _DummyModel.__name__
    model_dir.mkdir(parents=True)
    (model_dir / "result.json").write_text('{"value": "cached"}', encoding="utf-8")

    manifest = BamlExtractionManifest(
        function_name="ExtractDummy",
        config_name="default",
        model_name=_DummyModel.__name__,
        entries={
            "input:h": baml_module.BamlExtractionManifestEntry(
                source_hash="h", output_path=f"{_DummyModel.__name__}/result.json"
            )
        },
    )

    # baml_invoke must NOT be called for a cached entry
    async def _raise(*args, **kwargs):  # noqa: ANN002
        raise AssertionError("should not invoke BAML for a cached input")

    monkeypatch.setattr(baml_module, "baml_invoke", _raise)

    result, model_name, output_path = asyncio.run(
        _process_single_input_task.fn(
            input_text="Hello",
            function_name="ExtractDummy",
            config_name="default",
            llm=None,
            output_dir=str(out),
            output_file="result.json",
            input_hash="h",
            force=False,
            existing_manifest=manifest,
        )
    )
    assert output_path == f"{_DummyModel.__name__}/result.json"


@pytest.mark.requires_feature("baml")
@pytest.mark.fake_models
def test_baml_structured_extraction_flow_no_files_returns_empty_manifest(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"

    manifest = baml_structured_extraction_flow(
        base_dir=str(src),
        output_dir=str(out),
        function_name="ExtractDummy",
        llm="default",
    )
    assert isinstance(manifest, BamlExtractionManifest)
    assert manifest.function_name == "ExtractDummy"
    assert manifest.entries == {}


@pytest.mark.requires_feature("baml")
@pytest.mark.fake_models
def test_baml_structured_extraction_flow_writes_manifest_with_stubbed_submit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    md = src / "example.md"
    md.write_text("# Example", encoding="utf-8")

    def fake_submit(file_info, function_name, config_name, llm, structured_root, root_dir):  # noqa: ANN001
        return _FakeFuture((str(file_info.path), "example.json", _DummyModel.__name__))

    monkeypatch.setattr(baml_module._process_single_file_task, "submit", fake_submit)

    manifest = baml_structured_extraction_flow(
        base_dir=str(src),
        output_dir=str(out),
        batch_size=1,
        function_name="ExtractDummy",
        llm="default",
    )

    assert isinstance(manifest, BamlExtractionManifest)
    assert len(manifest.entries) == 1
    # the flow persists the manifest; the per-file JSON is written by the (stubbed) task
    assert (out / "manifest.json").exists()


@pytest.mark.requires_feature("baml")
@pytest.mark.fake_models
def test_baml_structured_extraction_flow_all_cached_returns_without_invoking_baml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    md = src / "example.md"
    md.write_text("# Example", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()

    cache = ManifestCache.load(out / "manifest.json")
    cache.record_success(
        key=str(md),
        fingerprint=_compute_hash(md.read_bytes()),
        code_version=None,
        outputs={"output_path": "example.json"},
    )
    cache.save(out / "manifest.json")

    def _boom(*args, **kwargs):  # noqa: ANN002
        raise AssertionError("should not invoke BAML for a cached file")

    monkeypatch.setattr(baml_module._process_single_file_task, "submit", _boom)

    manifest = baml_structured_extraction_flow(
        base_dir=str(src),
        output_dir=str(out),
        function_name="ExtractDummy",
        llm="default",
    )
    assert str(md) in manifest.entries


# ---------------------------------------------------------------------------
# baml_single_input_flow — harness runs
# ---------------------------------------------------------------------------


@pytest.mark.requires_feature("baml")
@pytest.mark.fake_models
def test_baml_single_input_flow_no_output_returns_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_submit(
        *, input_text, function_name, config_name, llm, output_dir, output_file, input_hash, force, existing_manifest
    ):  # noqa: ANN001
        return _FakeFuture((_DummyModel(value="ok"), _DummyModel.__name__, None))

    monkeypatch.setattr(baml_module._process_single_input_task, "submit", fake_submit)

    result, model_name, resolved_llm, relative_output_path = baml_single_input_flow(
        input_text="Hello",
        function_name="ExtractDummy",
        llm="default",
    )
    assert isinstance(result, BaseModel)
    assert model_name == _DummyModel.__name__
    assert resolved_llm is None
    assert relative_output_path is None


@pytest.mark.requires_feature("baml")
@pytest.mark.fake_models
def test_baml_single_input_flow_with_output_saves_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "structured"
    out.mkdir()

    def fake_submit(
        *, input_text, function_name, config_name, llm, output_dir, output_file, input_hash, force, existing_manifest
    ):  # noqa: ANN001
        model_name = _DummyModel.__name__
        return _FakeFuture((_DummyModel(value="ok"), model_name, f"{model_name}/{output_file}"))

    monkeypatch.setattr(baml_module._process_single_input_task, "submit", fake_submit)

    result, model_name, _, relative_output_path = baml_single_input_flow(
        input_text="Hello",
        function_name="ExtractDummy",
        llm="default",
        output_dir=str(out),
        output_file="result.json",
    )
    assert isinstance(result, _DummyModel)
    assert model_name == _DummyModel.__name__
    assert relative_output_path == f"{_DummyModel.__name__}/result.json"
    # manifest written under the model directory
    assert (out / _DummyModel.__name__ / "manifest.json").exists()
