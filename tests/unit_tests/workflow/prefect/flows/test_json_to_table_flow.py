"""Unit tests for the ``json_to_table`` Prefect flow.

Covers the helper functions, the individual ``@task`` callables, and smoke runs
of :func:`json_to_table_flow` producing CSV and Excel output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from genai_tk.workflow.prefect.flows.json_to_table_flow import (
    _flatten_value,
    _record_from_json,
    build_dataframe_task,
    json_to_table_flow,
    load_json_files_task,
    write_table_task,
)


class PersonRow(BaseModel):
    """Sample Pydantic model used for the model-validation code path."""

    name: str
    age: int = 0
    skills: list[str] = Field(default_factory=list)


_MODEL_PATH = "tests.unit_tests.workflow.prefect.flows.test_json_to_table_flow.PersonRow"


# ---------------------------------------------------------------------------
# _flatten_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("hello", "hello"),
        (42, 42),
        (3.14, 3.14),
        (True, True),
        (["a", "b", "c"], "a; b; c"),
        ({"k": "v"}, '{"k": "v"}'),
        ([1, 2], "1; 2"),
    ],
)
def test_flatten_value(value: Any, expected: Any) -> None:
    assert _flatten_value(value) == expected


def test_flatten_value_dict_is_json_serialised() -> None:
    result = _flatten_value({"nested": [1, 2]})
    assert json.loads(result) == {"nested": [1, 2]}


# ---------------------------------------------------------------------------
# _record_from_json
# ---------------------------------------------------------------------------


def test_record_from_json_no_keys_no_model_returns_all_fields() -> None:
    data = {"name": "Ada", "age": 30, "skills": ["python"]}
    record = _record_from_json(data, keys=None, model_cls=None)
    assert record == {"name": "Ada", "age": 30, "skills": "python"}


def test_record_from_json_with_keys_selects_and_orders() -> None:
    data = {"name": "Ada", "age": 30, "skills": ["python"], "extra": "ignored"}
    record = _record_from_json(data, keys=["skills", "name"], model_cls=None)
    assert list(record.keys()) == ["skills", "name"]
    assert record["skills"] == "python"
    assert record["name"] == "Ada"


def test_record_from_json_with_model_validates_and_fills_defaults() -> None:
    data = {"name": "Ada", "skills": ["python"]}
    record = _record_from_json(data, keys=None, model_cls=PersonRow)
    # age default filled by the model
    assert record["age"] == 0
    assert record["name"] == "Ada"
    assert record["skills"] == "python"


def test_record_from_json_with_keys_after_model_validation() -> None:
    data = {"name": "Ada", "skills": ["python", "go"]}
    record = _record_from_json(data, keys=["name"], model_cls=PersonRow)
    assert record == {"name": "Ada"}


def test_record_from_json_non_pydantic_model_skips_validation() -> None:
    class NotAModel:
        pass

    data = {"name": "Ada", "age": 30}
    record = _record_from_json(data, keys=None, model_cls=NotAModel)
    # validation skipped; data returned as-is (flattened)
    assert record == {"name": "Ada", "age": 30}


# ---------------------------------------------------------------------------
# load_json_files_task
# ---------------------------------------------------------------------------


@pytest.mark.fake_models
def test_load_json_files_task_finds_json(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("ignore")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.json").write_text("{}")

    files = load_json_files_task(str(tmp_path), None)
    names = sorted(p.name for p in files)
    assert names == ["a.json", "b.json", "c.json"]


@pytest.mark.fake_models
def test_load_json_files_task_empty_dir(tmp_path: Path) -> None:
    assert load_json_files_task(str(tmp_path), None) == []


# ---------------------------------------------------------------------------
# build_dataframe_task
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.fake_models
def test_build_dataframe_task_no_model_no_keys(tmp_path: Path) -> None:
    f1 = _write_json(tmp_path / "a.json", {"name": "Ada", "age": 30})
    f2 = _write_json(tmp_path / "b.json", {"name": "Bo", "age": 25})

    rows = build_dataframe_task([f1, f2], model_dotted_path=None, keys=None)
    assert len(rows) == 2
    assert rows[0]["name"] == "Ada"
    assert rows[0]["age"] == 30
    assert rows[0]["_source"] == str(f1)
    assert rows[1]["name"] == "Bo"


@pytest.mark.fake_models
def test_build_dataframe_task_with_model_and_keys(tmp_path: Path) -> None:
    f1 = _write_json(tmp_path / "a.json", {"name": "Ada", "skills": ["python"]})
    f2 = _write_json(tmp_path / "b.json", {"name": "Bo", "skills": ["go"]})

    rows = build_dataframe_task([f1, f2], model_dotted_path=_MODEL_PATH, keys=["name", "age"])
    assert len(rows) == 2
    assert {r["name"] for r in rows} == {"Ada", "Bo"}
    # age filled via model default
    assert all(r["age"] == 0 for r in rows)


@pytest.mark.fake_models
def test_build_dataframe_task_skips_corrupt_json(tmp_path: Path) -> None:
    good = _write_json(tmp_path / "good.json", {"name": "Ada"})
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")

    rows = build_dataframe_task([good, tmp_path / "bad.json"], model_dotted_path=None, keys=None)
    assert len(rows) == 1
    assert rows[0]["name"] == "Ada"


@pytest.mark.fake_models
def test_build_dataframe_task_bad_model_path_raises(tmp_path: Path) -> None:
    f1 = _write_json(tmp_path / "a.json", {"name": "Ada"})
    with pytest.raises(RuntimeError, match="Cannot import model"):
        build_dataframe_task([f1], model_dotted_path="nonexistent.module.Foo", keys=None)


# ---------------------------------------------------------------------------
# write_table_task
# ---------------------------------------------------------------------------


@pytest.mark.fake_models
def test_write_table_task_csv(tmp_path: Path) -> None:
    rows = [{"name": "Ada", "age": 30}, {"name": "Bo", "age": 25}]
    out = tmp_path / "nested" / "out.csv"
    result = write_table_task(rows, str(out), "Sheet1")
    assert Path(result).exists()
    text = Path(result).read_text(encoding="utf-8")
    assert "name,age" in text
    assert "Ada,30" in text


@pytest.mark.fake_models
def test_write_table_task_xlsx(tmp_path: Path) -> None:
    rows = [{"name": "Ada", "age": 30}]
    out = tmp_path / "out.xlsx"
    result = write_table_task(rows, str(out), "Data")
    assert Path(result).exists()
    assert Path(result).suffix == ".xlsx"


@pytest.mark.fake_models
def test_write_table_task_unsupported_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported output format"):
        write_table_task([{"a": 1}], str(tmp_path / "out.txt"), "Sheet1")


# ---------------------------------------------------------------------------
# json_to_table_flow — smoke runs
# ---------------------------------------------------------------------------


@pytest.mark.fake_models
def test_json_to_table_flow_writes_csv(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_json(src / "a.json", {"name": "Ada", "age": 30})
    _write_json(src / "b.json", {"name": "Bo", "age": 25})

    out = tmp_path / "result.csv"
    result = json_to_table_flow(input_dir=str(src), output_file=str(out))
    assert Path(result).exists()
    text = Path(result).read_text(encoding="utf-8")
    assert "Ada" in text and "Bo" in text


@pytest.mark.fake_models
def test_json_to_table_flow_with_keys_subset(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_json(src / "a.json", {"name": "Ada", "age": 30, "skills": ["python"]})

    out = tmp_path / "result.csv"
    result = json_to_table_flow(input_dir=str(src), output_file=str(out), keys=["name"])
    text = Path(result).read_text(encoding="utf-8")
    assert "name" in text
    assert "Ada" in text
    assert "skills" not in text


@pytest.mark.fake_models
def test_json_to_table_flow_no_files_returns_path_without_writing(tmp_path: Path) -> None:
    src = tmp_path / "empty"
    src.mkdir()
    out = tmp_path / "result.csv"
    result = json_to_table_flow(input_dir=str(src), output_file=str(out))
    assert result == str(out)
    assert not Path(out).exists()


@pytest.mark.fake_models
def test_json_to_table_flow_xlsx_output(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_json(src / "a.json", {"name": "Ada", "age": 30})

    out = tmp_path / "result.xlsx"
    result = json_to_table_flow(input_dir=str(src), output_file=str(out))
    assert Path(result).exists()
    assert Path(result).suffix == ".xlsx"
