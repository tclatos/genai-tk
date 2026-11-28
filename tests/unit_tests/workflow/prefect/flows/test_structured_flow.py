"""Unit tests for the LinkML & Pydantic structured Prefect flows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

try:
    from genai_tk.extra.structured.linkml_contract import LinkMLContract
    from genai_tk.workflow.prefect.flows.linkml_flow import (
        _chunked,
        _compute_hash,
        _prepare_files,
        compute_code_version,
        linkml_extraction_flow,
        linkml_single_input_flow,
    )
except ImportError:
    pytest.skip("linkml feature not installed", allow_module_level=True)

import genai_tk.workflow.prefect.flows.linkml_flow as flow_module
from genai_tk.workflow.flow_cache.manifest import ManifestCache
from tests.generated.resume import Resume

RESUME_SCHEMA = Path(__file__).parent.parent.parent.parent.parent / "linkml_src" / "resume.yaml"
PROMPT_FILE = Path(__file__).parent.parent.parent.parent.parent / "prompts" / "resume.md"


# ---------------------------------------------------------------------------
# _compute_hash & _chunked
# ---------------------------------------------------------------------------


def test_compute_hash_deterministic() -> None:
    assert _compute_hash(b"hello") == _compute_hash(b"hello")
    assert _compute_hash(b"hello") != _compute_hash(b"world")


def test_chunked_batches() -> None:
    items = [1, 2, 3, 4, 5]
    assert list(_chunked(items, 2)) == [[1, 2], [3, 4], [5]]
    assert list(_chunked(items, 0)) == [items]


# ---------------------------------------------------------------------------
# _prepare_files
# ---------------------------------------------------------------------------


def test_prepare_files_markdown_and_pdf(tmp_path: Path) -> None:
    md_file = tmp_path / "doc.md"
    md_file.write_text("Resume of John", encoding="utf-8")

    pdf_file = tmp_path / "resume.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy")

    cache = ManifestCache()
    to_process, skipped = _prepare_files([md_file, pdf_file], cache, code_version="v1", force=False)

    assert len(to_process) == 2
    assert skipped == 0

    item_md = next(f for f in to_process if f.path == md_file)
    assert not item_md.is_pdf
    assert item_md.content_text == "Resume of John"

    item_pdf = next(f for f in to_process if f.path == pdf_file)
    assert item_pdf.is_pdf
    assert item_pdf.content_bytes == b"%PDF-1.4 dummy"


def test_prepare_files_fresh_file_skipped(tmp_path: Path) -> None:
    md_file = tmp_path / "doc.md"
    md_file.write_text("Fresh content", encoding="utf-8")

    cache = ManifestCache()
    content_hash = _compute_hash(md_file.read_bytes())
    cache.record_success(key=str(md_file), fingerprint=content_hash, code_version="v1")

    # Fresh -> skip
    to_process, skipped = _prepare_files([md_file], cache, code_version="v1", force=False)
    assert len(to_process) == 0
    assert skipped == 1

    # Force -> re-process
    to_process_forced, skipped_forced = _prepare_files([md_file], cache, code_version="v1", force=True)
    assert len(to_process_forced) == 1
    assert skipped_forced == 0


def test_compute_code_version() -> None:
    contract = LinkMLContract(RESUME_SCHEMA)
    cv1 = compute_code_version(contract, prompt="Prompt text", llm="default")
    cv2 = compute_code_version(contract, prompt="Prompt text", llm="default")
    assert cv1 == cv2

    cv_diff_prompt = compute_code_version(contract, prompt="Different prompt", llm="default")
    assert cv1 != cv_diff_prompt


# ---------------------------------------------------------------------------
# Flow execution tests (with processor stubbed)
# ---------------------------------------------------------------------------


def test_structured_single_input_flow_execution_and_caching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample_resume = Resume(
        name="John Smith",
        email="john@example.com",
        experience=["Senior Engineer"],
        skills=["Python", "PostgreSQL"],
    )

    async def mock_aextract(self: Any, document_id: str, content: Any, is_pdf: bool = False) -> Resume:
        return sample_resume

    monkeypatch.setattr(flow_module.LinkMLProcessor, "aextract", mock_aextract)

    out_dir = tmp_path / "out"
    input_text = "John Smith; SW engineer; john@example.com"

    # Run 1: initial execution
    res1, model_name1, out_path1 = linkml_single_input_flow(
        input_text=input_text,
        schema_path=str(RESUME_SCHEMA),
        root_class="Resume",
        generated_module="tests.generated.resume",
        output_dir=str(out_dir),
        output_file="result.json",
        force=False,
    )

    assert isinstance(res1, Resume)
    assert res1.name == "John Smith"
    assert model_name1 == "Resume"
    assert out_path1 == "Resume/result.json"

    saved_file = out_dir / "Resume" / "result.json"
    assert saved_file.exists()

    manifest_file = out_dir / "manifest.json"
    assert manifest_file.exists()

    # Run 2: cache hit
    res2, model_name2, out_path2 = linkml_single_input_flow(
        input_text=input_text,
        schema_path=str(RESUME_SCHEMA),
        root_class="Resume",
        generated_module="tests.generated.resume",
        output_dir=str(out_dir),
        output_file="result.json",
        force=False,
    )
    assert res2.name == "John Smith"
    assert out_path2 == "Resume/result.json"


def test_structured_extraction_flow_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    doc1 = src_dir / "candidate1.md"
    doc1.write_text("Candidate 1 resume text", encoding="utf-8")

    out_dir = tmp_path / "out"

    sample_resume = Resume(
        name="Candidate One",
        email="c1@example.com",
        experience=["Dev"],
        skills=["Linux"],
    )

    async def mock_aextract(self: Any, document_id: str, content: Any, is_pdf: bool = False) -> Resume:
        return sample_resume

    monkeypatch.setattr(flow_module.LinkMLProcessor, "aextract", mock_aextract)

    manifest = linkml_extraction_flow(
        base_dir=str(src_dir),
        output_dir=str(out_dir),
        schema_path=str(RESUME_SCHEMA),
        root_class="Resume",
        generated_module="tests.generated.resume",
        prompt=str(PROMPT_FILE),
        force=False,
    )

    assert len(manifest.entries) == 1
    assert manifest.model_name == "Resume"

    saved_json = out_dir / "Resume" / "candidate1.json"
    assert saved_json.exists()

    # Re-run: should skip unchanged file
    manifest2 = linkml_extraction_flow(
        base_dir=str(src_dir),
        output_dir=str(out_dir),
        schema_path=str(RESUME_SCHEMA),
        root_class="Resume",
        generated_module="tests.generated.resume",
        prompt=str(PROMPT_FILE),
        force=False,
    )
    assert len(manifest2.entries) == 1
