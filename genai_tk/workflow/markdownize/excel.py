"""Spreadsheet-to-Markdown conversion via ``md-spreadsheet-parser``."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _grid_cell(value: Any) -> str:
    """Stringify a raw openpyxl cell value, mapping None to an empty string."""
    return "" if value is None else str(value)


def _drop_empty_rows_and_cols(grid: list[list[str]]) -> list[list[str]]:
    """Remove fully-blank rows/columns and pad ragged rows to a common width."""
    rows = [row for row in grid if any(cell.strip() for cell in row)]
    if not rows:
        return []
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    keep_cols = [i for i in range(width) if any(row[i].strip() for row in rows)]
    return [[row[i] for i in keep_cols] for row in rows]


def _split_leading_title(grid: list[list[str]]) -> tuple[str | None, list[list[str]]]:
    """Rescue a title row (a single filled cell above a wider header row) from the header."""
    first_row = grid[0]
    filled = [cell for cell in first_row if cell.strip()]
    if len(grid) > 1 and len(filled) == 1 and sum(bool(cell.strip()) for cell in grid[1]) > 1:
        return filled[0], grid[1:]
    return None, grid


def _excel_to_markdown_md_parser(path: Path) -> str:
    """Convert an .xlsx/.xls file to Markdown via ``md-spreadsheet-parser``.

    One section per worksheet: empty rows/columns are dropped, a leading title
    row is promoted to a heading, and merged header cells are forward-filled.
    """
    import openpyxl
    from md_spreadsheet_parser import ExcelParsingSchema, parse_excel

    schema = ExcelParsingSchema(header_rows=1, fill_merged_headers=True)
    workbook = openpyxl.load_workbook(path, data_only=True)
    parts: list[str] = []

    for worksheet in workbook.worksheets:
        grid = [[_grid_cell(cell) for cell in row] for row in worksheet.iter_rows(values_only=True)]
        grid = _drop_empty_rows_and_cols(grid)
        if not grid:
            continue

        title, grid = _split_leading_title(grid)
        heading = f"## {worksheet.title}" + (f"\n\n### {title}" if title else "")
        table = parse_excel(grid, schema)
        parts.append(f"{heading}\n\n{table.to_markdown()}\n")

    return "\n".join(parts)
