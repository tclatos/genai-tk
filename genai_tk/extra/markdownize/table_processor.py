"""HTML table analysis and lossless conversion to Markdown."""

from __future__ import annotations

import re
import warnings

import markdownify
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning, Tag

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

_HTML_TABLE_PATTERN = re.compile(r"<table(?:\s+[^>]*)?>(.*?)</table>", re.DOTALL | re.IGNORECASE)
_MISTRAL_TABLE_LINK_PATTERN = re.compile(
    r"(?:!?\[(?P<alt>[^\]]*)\]\((?P<url>[^\s\)\"\']+\.html?)(?:\s+[\"'][^\"']*[\"'])?\))",
    re.IGNORECASE,
)


def is_markdown_table(html_or_tag: str | Tag) -> bool:
    """Determine whether an HTML table can be losslessly converted to a standard Markdown table.

    A table cannot be represented losslessly in Markdown if:
    1. Any cell (<td> or <th>) uses `rowspan` > 1.
    2. Any cell (<td> or <th>) uses `colspan` > 1.
    3. Any cell contains a nested <table>.

    Args:
        html_or_tag: Raw HTML string or BeautifulSoup Tag representing the table.

    Returns:
        True if the table is a simple grid convertible to standard Markdown table without loss.
    """
    if isinstance(html_or_tag, Tag):
        soup_table = html_or_tag if html_or_tag.name == "table" else html_or_tag.find("table")
    else:
        soup = BeautifulSoup(html_or_tag, "html.parser")
        soup_table = soup.find("table")

    if soup_table is None:
        return False

    # Check for nested tables
    nested_tables = soup_table.find_all("table")
    if nested_tables:
        return False

    # Check cells for rowspan or colspan > 1
    for cell in soup_table.find_all(["td", "th"]):
        rowspan = cell.get("rowspan")
        if rowspan is not None:
            try:
                if int(str(rowspan).strip()) > 1:
                    return False
            except ValueError:
                return False

        colspan = cell.get("colspan")
        if colspan is not None:
            try:
                if int(str(colspan).strip()) > 1:
                    return False
            except ValueError:
                return False

    return True


def get_table_dimensions(html_or_tag: str | Tag) -> tuple[int, int]:
    """Calculate the dimensions (rows, cols) of an HTML table.

    Args:
        html_or_tag: Raw HTML string or BeautifulSoup Tag representing the table.

    Returns:
        Tuple of (row_count, col_count).
    """
    if isinstance(html_or_tag, Tag):
        soup_table = html_or_tag if html_or_tag.name == "table" else html_or_tag.find("table")
    else:
        soup = BeautifulSoup(html_or_tag, "html.parser")
        soup_table = soup.find("table")

    if soup_table is None:
        return 0, 0

    rows = soup_table.find_all("tr")
    row_count = len(rows)
    col_count = 0

    for row in rows:
        cells = row.find_all(["td", "th"], recursive=False)
        curr_cols = 0
        for cell in cells:
            colspan = cell.get("colspan", 1)
            try:
                curr_cols += int(str(colspan).strip())
            except ValueError:
                curr_cols += 1
        col_count = max(col_count, curr_cols)

    return row_count, col_count


def expand_html_table_to_grid(table_tag: Tag) -> list[list[str]] | None:
    """Expand an HTML table with rowspan and colspan into a rectangular 2D matrix.

    Returns None if the table cannot be cleanly flattened (e.g. nested tables).

    Args:
        table_tag: BeautifulSoup Tag representing the <table>.

    Returns:
        2D list of strings representing the grid cells, or None.
    """
    # Check for nested tables - cannot be safely flattened
    nested_tables = table_tag.find_all("table")
    if nested_tables:
        return None

    rows = table_tag.find_all("tr")
    if not rows:
        return None

    grid: list[list[str | None]] = []

    for r_idx, tr in enumerate(rows):
        while len(grid) <= r_idx:
            grid.append([])

        c_idx = 0
        cells = tr.find_all(["td", "th"], recursive=False)
        for cell in cells:
            # Advance past already occupied cells in this row
            while c_idx < len(grid[r_idx]) and grid[r_idx][c_idx] is not None:
                c_idx += 1

            # Read rowspan and colspan
            try:
                rowspan = max(1, min(500, int(str(cell.get("rowspan", 1)).strip())))
            except (ValueError, TypeError):
                rowspan = 1
            try:
                colspan = max(1, min(500, int(str(cell.get("colspan", 1)).strip())))
            except (ValueError, TypeError):
                colspan = 1

            # Extract cell content and format for Markdown table cell (process children to avoid td/th pipe formatting)
            inner_html = "".join(str(c) for c in cell.children).strip()
            if inner_html:
                cell_md = markdownify.markdownify(inner_html, strip=["style", "script"]).strip()
            else:
                cell_md = ""
            cell_md = re.sub(r"[\r\n]+", " ", cell_md).strip()
            # Escape unescaped pipe characters
            cell_md = re.sub(r"(?<!\\)\|", r"\|", cell_md)

            # Fill the rectangular region in the grid
            for dr in range(rowspan):
                target_r = r_idx + dr
                while len(grid) <= target_r:
                    grid.append([])
                for dc in range(colspan):
                    target_c = c_idx + dc
                    while len(grid[target_r]) <= target_c:
                        grid[target_r].append(None)
                    grid[target_r][target_c] = cell_md

            c_idx += colspan

    max_cols = max((len(r) for r in grid), default=0)
    if max_cols == 0 or len(grid) == 0:
        return None

    normalized_grid: list[list[str]] = []
    for r in grid:
        row_cells = [(r[c] if c < len(r) and r[c] is not None else "") for c in range(max_cols)]
        normalized_grid.append(row_cells)

    return normalized_grid


def format_grid_to_markdown(grid: list[list[str]]) -> str:
    """Format a 2D matrix of cell strings into a standard Markdown pipe table."""
    if not grid or not grid[0]:
        return ""
    header_row = grid[0]
    separator_row = ["---"] * len(header_row)

    lines = [
        "| " + " | ".join(header_row) + " |",
        "| " + " | ".join(separator_row) + " |",
    ]
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def convert_html_table(table_html: str, *, table_expanded: bool = True) -> str:
    """Convert an HTML table to Markdown if convertible, else return HTML with dimension comments.

    When `table_expanded=True`, tables with `rowspan` and `colspan` are expanded into a
    rectangular 2D grid of Markdown cells. If expanding is not possible (e.g. nested tables),
    the structured HTML table is retained.

    Args:
        table_html: Complete <table>...</table> HTML string.
        table_expanded: Whether to expand rowspan/colspan into rectangular Markdown tables.

    Returns:
        Converted Markdown string or dimension-annotated HTML table string.
    """
    soup = BeautifulSoup(table_html, "html.parser")
    table_tag = soup.find("table")
    if not table_tag:
        return table_html

    if is_markdown_table(table_tag):
        # Convert simple grid to markdown using markdownify
        md_text = markdownify.markdownify(str(table_tag), strip=["style", "script"]).strip()
        return md_text

    if table_expanded:
        grid = expand_html_table_to_grid(table_tag)
        if grid:
            return format_grid_to_markdown(grid)

    # Complex / nested table that cannot be flattened: retain HTML
    rows, cols = get_table_dimensions(table_tag)
    comment = f"<!-- Table: {rows}x{cols} -->"
    return f"{comment}\n{str(table_tag).strip()}\n"


def process_markdown_tables(markdown: str, *, table_expanded: bool = True) -> str:
    """Find and process all HTML tables in a Markdown document.

    - If a table can be converted to Markdown (simple or via span expansion), converts it.
    - If it has complex nested layouts, keeps it as HTML and adds a dimension comment.
    - Removes standalone Mistral HTML table links (e.g. `[table-0.html](table-0.html)`).

    Args:
        markdown: Full Markdown text.
        table_expanded: Whether to expand rowspan/colspan into rectangular Markdown tables.

    Returns:
        Markdown text with processed tables and cleaned table links.
    """
    if not markdown:
        return ""

    def _replace_table(match: re.Match) -> str:
        table_html = match.group(0)
        return convert_html_table(table_html, table_expanded=table_expanded)

    # Replace HTML tables
    result = _HTML_TABLE_PATTERN.sub(_replace_table, markdown)

    # Remove standalone Mistral HTML table links
    result = _MISTRAL_TABLE_LINK_PATTERN.sub("", result)

    return result
