"""Unit tests for the ``ppt2pdf`` Prefect flow.

LibreOffice is invoked via :mod:`subprocess` — a true external boundary — so
``subprocess.run`` is the only thing faked.  The real ``_convert_with_libreoffice``
command-building and output-existence logic is exercised end-to-end.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import genai_tk.workflow.prefect.flows.ppt2pdf_flow as ppt2pdf_module
from genai_tk.workflow.flow_cache.manifest import ManifestCache
from genai_tk.workflow.prefect.flows.ppt2pdf_flow import (
    SUPPORTED_EXTENSIONS,
    _chunked,
    _convert_with_libreoffice,
    _FileToProcess,
    _is_ppt_compatible,
    _prepare_files,
    _process_single_file_task,
    _TaskResult,
    ppt2pdf_flow,
)

# ---------------------------------------------------------------------------
# _is_ppt_compatible
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [(s, True) for s in SUPPORTED_EXTENSIONS] + [(".pdf", False), (".docx", False), (".txt", False)],
)
def test_is_ppt_compatible(suffix: str, expected: bool) -> None:
    assert _is_ppt_compatible(Path(f"file{suffix}")) is expected


# ---------------------------------------------------------------------------
# _chunked
# ---------------------------------------------------------------------------


def test_chunked_splits_into_batches() -> None:
    assert list(_chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_chunked_zero_or_negative_size_returns_single_batch() -> None:
    assert list(_chunked([1, 2, 3], 0)) == [[1, 2, 3]]
    assert list(_chunked([1, 2, 3], -1)) == [[1, 2, 3]]


def test_chunked_empty() -> None:
    assert list(_chunked([], 2)) == []


# ---------------------------------------------------------------------------
# _prepare_files
# ---------------------------------------------------------------------------


def test_prepare_files_new_file_queued(tmp_path: Path) -> None:
    ppt = tmp_path / "deck.pptx"
    ppt.write_bytes(b"fake ppt")
    cache = ManifestCache()
    to_process, skipped = _prepare_files([ppt], cache, force=False)
    assert len(to_process) == 1
    assert skipped == 0
    assert to_process[0].path == ppt


def test_prepare_files_fresh_file_skipped(tmp_path: Path) -> None:
    ppt = tmp_path / "deck.pptx"
    ppt.write_bytes(b"fake ppt")
    cache = ManifestCache()
    # record a success with the current fingerprint so the file is "fresh"
    from genai_tk.utils.hashing import buffer_digest

    cache.record_success(key=str(ppt), fingerprint=buffer_digest(ppt.read_bytes()), outputs={"output_path": "x.pdf"})
    to_process, skipped = _prepare_files([ppt], cache, force=False)
    assert to_process == []
    assert skipped == 1


def test_prepare_files_force_reprocesses_fresh(tmp_path: Path) -> None:
    ppt = tmp_path / "deck.pptx"
    ppt.write_bytes(b"fake ppt")
    cache = ManifestCache()
    from genai_tk.utils.hashing import buffer_digest

    cache.record_success(key=str(ppt), fingerprint=buffer_digest(ppt.read_bytes()), outputs={})
    to_process, skipped = _prepare_files([ppt], cache, force=True)
    assert len(to_process) == 1
    assert skipped == 0


def test_prepare_files_skips_unreadable(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pptx"
    cache = ManifestCache()
    to_process, skipped = _prepare_files([missing], cache, force=False)
    assert to_process == []
    assert skipped == 0


# ---------------------------------------------------------------------------
# _convert_with_libreoffice (subprocess.run faked)
# ---------------------------------------------------------------------------


def _fake_run_writing_pdf(output_dir: Path, stem: str, returncode: int = 0, stderr: str = "") -> MagicMock:
    """Return a MagicMock substituting subprocess.run that writes the expected PDF."""

    def _run(cmd, capture_output=False, text=False, timeout=None):  # noqa: ANN001
        # Simulate LibreOffice writing the PDF output file
        (output_dir / f"{stem}.pdf").write_bytes(b"%PDF-1.4")
        m = MagicMock()
        m.returncode = returncode
        m.stdout = ""
        m.stderr = stderr
        return m

    mock = MagicMock(side_effect=_run)
    return mock


def test_convert_with_libreoffice_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "deck.pptx"
    src.write_bytes(b"fake")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(ppt2pdf_module.subprocess, "run", _fake_run_writing_pdf(out_dir, "deck"))

    result = _convert_with_libreoffice(src, out_dir)
    assert result == out_dir / "deck.pdf"
    assert result.exists()


def test_convert_with_libreoffice_nonzero_return_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "deck.pptx"
    src.write_bytes(b"fake")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        ppt2pdf_module.subprocess, "run", _fake_run_writing_pdf(out_dir, "deck", returncode=1, stderr="boom")
    )

    with pytest.raises(RuntimeError, match="LibreOffice conversion failed"):
        _convert_with_libreoffice(src, out_dir)


def test_convert_with_libreoffice_missing_output_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "deck.pptx"
    src.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    # run "succeeds" but never writes the PDF file
    def _run(cmd, capture_output=False, text=False, timeout=None):  # noqa: ANN001
        m = MagicMock()
        m.returncode = 0
        m.stdout = ""
        m.stderr = ""
        return m

    monkeypatch.setattr(ppt2pdf_module.subprocess, "run", _run)
    with pytest.raises(RuntimeError, match="Expected output file not found"):
        _convert_with_libreoffice(src, out_dir)


def test_convert_with_libreoffice_timeout_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "deck.pptx"
    src.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    def _run(cmd, capture_output=False, text=False, timeout=None):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(ppt2pdf_module.subprocess, "run", _run)
    with pytest.raises(RuntimeError, match="timed out"):
        _convert_with_libreoffice(src, out_dir)


# ---------------------------------------------------------------------------
# _process_single_file_task
# ---------------------------------------------------------------------------


@pytest.mark.fake_models
def test_process_single_file_task_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    src.mkdir()
    ppt = src / "deck.pptx"
    ppt.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(ppt2pdf_module.subprocess, "run", _fake_run_writing_pdf(out_dir, "deck"))

    file_info = _FileToProcess(path=ppt, content_hash="h")
    result = _process_single_file_task(file_info, output_dir=str(out_dir), root_dir=str(src))

    assert isinstance(result, _TaskResult)
    assert result.success is True
    assert result.source_path == str(ppt)
    assert result.output_path == "deck.pdf"


@pytest.mark.fake_models
def test_process_single_file_task_failure_returns_failed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    ppt = src / "deck.pptx"
    ppt.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    def _run(cmd, capture_output=False, text=False, timeout=None):  # noqa: ANN001
        m = MagicMock()
        m.returncode = 2
        m.stdout = ""
        m.stderr = "cannot open"
        return m

    monkeypatch.setattr(ppt2pdf_module.subprocess, "run", _run)

    file_info = _FileToProcess(path=ppt, content_hash="h")
    result = _process_single_file_task(file_info, output_dir=str(out_dir), root_dir=str(src))

    assert result.success is False
    assert result.source_path == str(ppt)
    assert "LibreOffice" in (result.error or "")


# ---------------------------------------------------------------------------
# ppt2pdf_flow — smoke runs
# ---------------------------------------------------------------------------


@pytest.mark.fake_models
def test_ppt2pdf_flow_no_files_returns_empty_cache(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "notes.txt").write_text("not a ppt")

    result = ppt2pdf_flow(base_dir=str(src), output_dir=str(tmp_path / "out"))
    assert isinstance(result, ManifestCache)
    assert result.records == {}


@pytest.mark.fake_models
def test_ppt2pdf_flow_nonexistent_base_dir(tmp_path: Path) -> None:
    result = ppt2pdf_flow(base_dir=str(tmp_path / "missing"), output_dir=str(tmp_path / "out"))
    assert isinstance(result, ManifestCache)
    assert result.records == {}


@pytest.mark.fake_models
def test_ppt2pdf_flow_converts_and_records_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    src.mkdir()
    ppt = src / "deck.pptx"
    ppt.write_bytes(b"fake ppt")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(ppt2pdf_module.subprocess, "run", _fake_run_writing_pdf(out_dir, "deck"))

    result = ppt2pdf_flow(base_dir=str(src), output_dir=str(out_dir), batch_size=1)

    assert isinstance(result, ManifestCache)
    assert str(ppt) in result.records
    assert (out_dir / "deck.pdf").exists()
    # manifest persisted to disk
    assert (out_dir / "manifest.json").exists()


@pytest.mark.fake_models
def test_ppt2pdf_flow_all_files_cached_returns_without_converting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    ppt = src / "deck.pptx"
    ppt.write_bytes(b"fake ppt")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    from genai_tk.utils.hashing import buffer_digest

    cache = ManifestCache.load(out_dir / "manifest.json")
    cache.record_success(key=str(ppt), fingerprint=buffer_digest(ppt.read_bytes()), outputs={"output_path": "deck.pdf"})
    cache.save(out_dir / "manifest.json")

    # If the flow tries to convert, this raises (proving it was skipped)
    def _boom(*args, **kwargs):  # noqa: ANN002
        raise AssertionError("should not convert a cached file")

    monkeypatch.setattr(ppt2pdf_module.subprocess, "run", _boom)

    result = ppt2pdf_flow(base_dir=str(src), output_dir=str(out_dir), force=False)
    assert str(ppt) in result.records
