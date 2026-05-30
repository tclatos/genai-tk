"""Prefect flow to convert JSON extraction outputs to tabular CSV/Excel format.

Reads JSON files produced by ``baml_extract`` (or any JSON files whose top-level
keys match a Pydantic model), optionally sub-selects a list of columns, and
writes the result as a CSV or Excel file.

Typical usage::

    # CSV output
    uv run cli workflow run json_to_table \\
        --set input_dir=./data/structured/Resume \\
        --set output_file=./data/results.csv \\
        --set model=myapp.baml_client.types.Resume \\
        --set keys='["name","skills","years_experience"]'

    # Excel output (auto-detected by file extension)
    uv run cli workflow run json_to_table \\
        --set input_dir=./data/structured/Resume \\
        --set output_file=./data/results.xlsx \\
        --set model=myapp.baml_client.types.Resume
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from prefect import flow, task

from genai_tk.utils.file_patterns import resolve_config_path, resolve_files
from genai_tk.utils.import_utils import import_model

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _flatten_value(value: Any) -> str | int | float | bool | None:
    """Flatten a nested value to a scalar for tabular output.

    Lists are joined as ``; ``-separated strings.
    Dicts are serialised as JSON.
    Everything else is returned as-is.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return value


def _record_from_json(
    data: dict[str, Any],
    keys: list[str] | None,
    model_cls: type | None,
) -> dict[str, Any]:
    """Build a flat row dict from a JSON data dict.

    Args:
        data: Parsed JSON object.
        keys: Columns to include (in order).  ``None`` = all keys from *data*.
        model_cls: Pydantic model class used to validate/cast the data.
            When provided the data is first validated through the model, which
            fills defaults and coerces types.  Pass ``None`` to skip validation.

    Returns:
        Flat dict suitable for a DataFrame row.
    """
    if model_cls is not None:
        from pydantic import BaseModel

        if issubclass(model_cls, BaseModel):
            instance = model_cls.model_validate(data)
            data = instance.model_dump()
        else:
            logger.warning("Class {} is not a Pydantic BaseModel; skipping validation", model_cls.__name__)

    if keys:
        return {k: _flatten_value(data.get(k)) for k in keys}
    return {k: _flatten_value(v) for k, v in data.items()}


# ---------------------------------------------------------------------------
# Prefect tasks
# ---------------------------------------------------------------------------


@task
def load_json_files_task(input_dir: str, pathspecs: list[str] | None) -> list[Path]:
    """Discover JSON files in input_dir matching pathspecs.

    Args:
        input_dir: Root directory to scan.
        pathspecs: Gitwildmatch patterns; defaults to ``["**/*.json"]``.

    Returns:
        Sorted list of JSON file paths.
    """
    effective_specs = pathspecs or ["**/*.json"]
    files = resolve_files(input_dir, pathspecs=effective_specs)
    logger.info("Found {} JSON file(s) in {}", len(files), input_dir)
    return files


@task
def build_dataframe_task(
    json_files: list[Path],
    model_dotted_path: str | None,
    keys: list[str] | None,
) -> list[dict[str, Any]]:
    """Parse JSON files and build a list of row dicts.

    Args:
        json_files: List of JSON file paths to read.
        model_dotted_path: Dotted Python path to a Pydantic model class for
            validation/coercion, e.g. ``myapp.baml_client.types.Resume``.
            Pass ``None`` to skip model validation.
        keys: Columns to extract; ``None`` = all keys from the first record.

    Returns:
        List of row dicts (one per input file).
    """
    model_cls: type | None = None
    if model_dotted_path:
        try:
            model_cls = import_model(model_dotted_path)
            logger.debug("Loaded model class: {}", model_dotted_path)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(f"Cannot import model '{model_dotted_path}': {exc}") from exc

    rows: list[dict[str, Any]] = []
    for path in json_files:
        try:
            raw = path.read_text(encoding="utf-8")
            data: dict[str, Any] = json.loads(raw)
        except Exception as exc:
            logger.error("Skipping {} — cannot read/parse JSON: {}", path, exc)
            continue

        try:
            row = _record_from_json(data, keys, model_cls)
            row["_source"] = str(path)
            rows.append(row)
        except Exception as exc:
            logger.error("Skipping {} — error building row: {}", path, exc)
            continue

    logger.info("Built {} row(s) from {} file(s)", len(rows), len(json_files))
    return rows


@task
def write_table_task(
    rows: list[dict[str, Any]],
    output_file: str,
    sheet_name: str,
) -> str:
    """Write rows to a CSV or Excel file using pandas.

    The output format is inferred from the file extension:
    - ``.csv`` → CSV (UTF-8, comma separator)
    - ``.xlsx`` / ``.xls`` → Excel (openpyxl engine)

    Args:
        rows: List of row dicts to write.
        output_file: Destination file path.
        sheet_name: Excel sheet name (ignored for CSV).

    Returns:
        Resolved output file path.
    """
    import pandas as pd

    resolved = resolve_config_path(output_file)
    out_path = Path(resolved)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    suffix = out_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(out_path, index=False, encoding="utf-8")
        logger.success("Wrote CSV with {} rows to {}", len(df), out_path)
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(out_path, index=False, sheet_name=sheet_name, engine="openpyxl")
        logger.success("Wrote Excel ({} rows, sheet '{}') to {}", len(df), sheet_name, out_path)
    else:
        raise ValueError(f"Unsupported output format: '{suffix}'. Use .csv or .xlsx")

    return str(out_path)


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------


@flow(name="json_to_table")
def json_to_table_flow(
    input_dir: str,
    output_file: str,
    *,
    model: str | None = None,
    keys: list[str] | None = None,
    pathspecs: list[str] | None = None,
    sheet_name: str = "Sheet1",
    force: bool = False,
) -> str:
    """Convert JSON files in a directory to a single CSV or Excel table.

    Reads every JSON file under *input_dir* (matched by *pathspecs*), validates
    each record against a Pydantic model (optional), selects *keys* as columns,
    and writes the result to *output_file*.

    The output format is determined by the file extension:
    - ``.csv``  → UTF-8 CSV
    - ``.xlsx`` / ``.xls`` → Excel workbook

    Args:
        input_dir: Directory containing JSON files.  Supports ``${paths.*}`` vars.
        output_file: Destination file path (``.csv`` or ``.xlsx``).
        model: Dotted Python path to a Pydantic model class used for
            validation/coercion, e.g. ``myapp.baml_client.types.Resume``.
            When omitted all keys from the JSON objects are used as-is.
        keys: List of field names to include as columns (in order).
            When omitted all fields from the JSON are included.
        pathspecs: Gitwildmatch patterns to filter JSON files; default ``["**/*.json"]``.
        sheet_name: Excel sheet name (ignored for CSV output).
        force: Currently unused; reserved for future cache-bypass support.

    Returns:
        Resolved path of the written output file.
    """
    resolved_input = resolve_config_path(input_dir)

    json_files = load_json_files_task(resolved_input, pathspecs)

    if not json_files:
        logger.warning("No JSON files found in '{}'; nothing to write", resolved_input)
        return resolve_config_path(output_file)

    rows = build_dataframe_task(json_files, model, keys)

    if not rows:
        logger.warning("No rows produced; output file not written")
        return resolve_config_path(output_file)

    out = write_table_task(rows, output_file, sheet_name)
    return out
