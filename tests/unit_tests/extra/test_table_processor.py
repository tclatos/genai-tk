"""Unit tests for HTML table processing and Markdown conversion."""

import pytest

from genai_tk.extra.markdownize.table_processor import (
    convert_html_table,
    find_html_table_spans,
    get_table_dimensions,
    is_markdown_table,
    process_markdown_tables,
)


@pytest.mark.unit
def test_is_markdown_table_simple():
    simple_html = """
    <table>
        <thead>
            <tr><th>Name</th><th>Age</th><th>City</th></tr>
        </thead>
        <tbody>
            <tr><td>Alice</td><td>30</td><td>Paris</td></tr>
            <tr><td>Bob</td><td>25</td><td>London</td></tr>
        </tbody>
    </table>
    """
    assert is_markdown_table(simple_html) is True
    rows, cols = get_table_dimensions(simple_html)
    assert rows == 3
    assert cols == 3


@pytest.mark.unit
def test_is_markdown_table_with_rowspan():
    rowspan_html = """
    <table>
        <tr><td rowspan="2">Merged Row</td><td>Cell 1</td></tr>
        <tr><td>Cell 2</td></tr>
    </table>
    """
    assert is_markdown_table(rowspan_html) is False
    rows, cols = get_table_dimensions(rowspan_html)
    assert rows == 2
    assert cols == 2


@pytest.mark.unit
def test_is_markdown_table_with_colspan():
    colspan_html = """
    <table>
        <tr><th colspan="2">Merged Header</th></tr>
        <tr><td>Cell 1</td><td>Cell 2</td></tr>
    </table>
    """
    assert is_markdown_table(colspan_html) is False
    rows, cols = get_table_dimensions(colspan_html)
    assert rows == 2
    assert cols == 2


@pytest.mark.unit
def test_is_markdown_table_nested():
    nested_html = """
    <table>
        <tr>
            <td>Outer Cell</td>
            <td>
                <table><tr><td>Inner Cell</td></tr></table>
            </td>
        </tr>
    </table>
    """
    assert is_markdown_table(nested_html) is False


@pytest.mark.unit
def test_convert_html_table_simple():
    simple_html = "<table><tr><th>Col A</th><th>Col B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    converted = convert_html_table(simple_html)
    assert "| Col A" in converted
    assert "| Col B" in converted
    assert "<table" not in converted


@pytest.mark.unit
def test_convert_html_table_complex_expanded_to_markdown():
    complex_html = (
        '<table><tr><th colspan="2">Header</th><th>Value</th></tr>'
        '<tr><td rowspan="2">Group</td><td>A</td><td>1</td></tr>'
        "<tr><td>B</td><td>2</td></tr></table>"
    )
    converted = convert_html_table(complex_html, table_expanded=True)
    assert "| Header | Header | Value |" in converted
    assert "| Group | A | 1 |" in converted
    assert "| Group | B | 2 |" in converted
    assert "<table" not in converted


@pytest.mark.unit
def test_convert_html_table_complex_retains_html_when_not_expanded():
    complex_html = '<table><tr><th colspan="2">Header</th></tr><tr><td>1</td><td>2</td></tr></table>'
    converted = convert_html_table(complex_html, table_expanded=False)
    assert "<!-- Table: 2x2 -->" in converted
    assert "<table" in converted


@pytest.mark.unit
def test_convert_html_table_nested_retains_html():
    nested_html = "<table><tr><td>Outer</td><td><table><tr><td>Inner</td></tr></table></td></tr></table>"
    converted = convert_html_table(nested_html, table_expanded=True)
    assert "<!-- Table:" in converted
    assert "<table" in converted


@pytest.mark.unit
def test_convert_html_table_escapes_pipes_in_cells():
    html = '<table><tr><th>A</th><th>B</th></tr><tr><td>x|y</td><td>2</td></tr></table>'
    converted = convert_html_table(html)
    data_row = converted.splitlines()[-1]
    assert data_row == "| x\\|y | 2 |"


@pytest.mark.unit
def test_find_html_table_spans_handles_nesting_and_siblings():
    nested = "<table><tr><td>x<table><tr><td>y</td></tr></table></td></tr></table>"
    plain = "<table><tr><td>z</td></tr></table>"
    text = f"a{nested}b{plain}"
    spans = find_html_table_spans(text)
    assert len(spans) == 2
    first, second = spans
    assert text[first[0] : first[1]] == nested
    assert text[second[0] : second[1]] == plain


@pytest.mark.unit
def test_find_html_table_spans_unclosed():
    assert find_html_table_spans("a <table> unclosed markup") == []


@pytest.mark.unit
def test_process_markdown_tables_nested_no_stray_markup():
    nested = "<table><tr><td>Outer</td><td><table><tr><td>Inner</td></tr></table></td></tr></table>"
    processed = process_markdown_tables(f"Before\n{nested}\nAfter", table_expanded=True)
    assert "Before" in processed
    assert "After" in processed
    assert "<!-- Table:" in processed
    # The full nested table is retained exactly once, without stray closing fragments
    assert processed.count("<table") == 2
    assert processed.count("</table>") == 2
    assert processed.count("</td>") == 3


@pytest.mark.unit
def test_process_markdown_tables_mixed_nested_and_simple():
    md = (
        "<table><tr><td>A<table><tr><td>B</td></tr></table></td></tr></table>\n\n"
        "text between\n\n"
        "<table><tr><th>C</th></tr><tr><td>1</td></tr></table>"
    )
    processed = process_markdown_tables(md, table_expanded=True)
    # Nested table retained as HTML
    assert "<table" in processed
    assert processed.count("<table") == 2
    # Simple table between the nested one and the text is still converted
    assert "text between" in processed
    assert "| C |" in processed
    assert "| 1 |" in processed


@pytest.mark.unit
def test_process_markdown_tables():
    md = (
        "# Document Header\n\n"
        "Here is table 1:\n"
        "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>\n\n"
        "[table_1.html](table_1.html)\n\n"
        "And complex table 2:\n"
        '<table><tr><td rowspan="2">X</td><td>Y</td></tr><tr><td>Z</td></tr></table>\n'
    )
    processed = process_markdown_tables(md, table_expanded=True)
    # Simple table converted
    assert "| A | B |" in processed or ("| A" in processed and "| B" in processed)
    # Table link removed
    assert "table_1.html" not in processed
    # Complex table expanded to rectangular markdown table
    assert "| X | Y |" in processed
    assert "| X | Z |" in processed
    assert "<table" not in processed
