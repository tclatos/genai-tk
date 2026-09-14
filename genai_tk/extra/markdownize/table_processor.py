"""HTML table analysis and lossless conversion to Markdown."""

from __future__ import annotations

import re

import markdownify
from bs4 import BeautifulSoup, Tag

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


def convert_html_table(table_html: str) -> str:
    """Convert an HTML table to Markdown if lossless, else return HTML with dimension comments.

    Args:
        table_html: Complete <table>...</table> HTML string.

    Returns:
        Converted Markdown string or dimension-annotated HTML table string.
    """
    soup = BeautifulSoup(table_html, "html.parser")
    table_tag = soup.find("table")
    if not table_tag:
        return table_html

    if is_markdown_table(table_tag):
        # Convert to markdown using markdownify
        md_text = markdownify.markdownify(str(table_tag), strip=["style", "script"]).strip()
        return md_text
    else:
        rows, cols = get_table_dimensions(table_tag)
        comment = f"<!-- Table: {rows}x{cols} -->"
        return f"{comment}\n{str(table_tag).strip()}\n"


def process_markdown_tables(markdown: str) -> str:
    """Find and process all HTML tables in a Markdown document.

    - If a table can be converted to Markdown without loss, converts it.
    - If it has complex layouts (rowspan/colspan/nested), keeps it as HTML and adds a dimension comment.
    - Removes standalone Mistral HTML table links (e.g. `[table-0.html](table-0.html)`).

    Args:
        markdown: Full Markdown text.

    Returns:
        Markdown text with processed tables and cleaned table links.
    """
    if not markdown:
        return ""

    def _replace_table(match: re.Match) -> str:
        table_html = match.group(0)
        return convert_html_table(table_html)

    # Replace HTML tables
    result = _HTML_TABLE_PATTERN.sub(_replace_table, markdown)

    # Remove standalone Mistral HTML table links
    result = _MISTRAL_TABLE_LINK_PATTERN.sub("", result)

    return result
