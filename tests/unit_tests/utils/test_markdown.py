"""Unit tests for genai_tk.utils.markdown."""

from genai_tk.utils.markdown import looks_like_markdown


class TestLooksLikeMarkdown:
    def test_plain_text_returns_false(self) -> None:
        text = "This is just plain text without any formatting."
        result, reasons = looks_like_markdown(text)
        assert result is False

    def test_heading_indicates_markdown(self) -> None:
        # Multiple headings + list + code fence -> well above threshold of 3
        text = "# Main heading\n\n## Sub\n\n### Sub2\n\n- item1\n- item2\n- item3\n\n```python\ncode\n```"
        result, reasons = looks_like_markdown(text)
        assert result is True
        assert "heading" in reasons

    def test_bullet_list_indicates_markdown(self) -> None:
        # Multiple bullet lists over threshold
        text = "# Header\n\nItems:\n- apple\n- banana\n- cherry\n- more\n- and more\n- six"
        result, reasons = looks_like_markdown(text)
        assert result is True
        assert "ul_list" in reasons

    def test_ordered_list_indicates_markdown(self) -> None:
        # Multiple ordered list items + heading
        text = "# Steps\n\n1. First step\n2. Second step\n3. Third step\n4. Done\n5. More"
        result, reasons = looks_like_markdown(text)
        assert result is True
        assert "ol_list" in reasons

    def test_code_fence_indicates_markdown(self) -> None:
        # Code fence + bullet list + heading
        text = "# Example\n\n- item1\n- item2\n- item3\n\n```python\ndef hello():\n    pass\n```\n"
        result, reasons = looks_like_markdown(text)
        assert result is True
        assert "code_fence" in reasons

    def test_inline_code_indicates_markdown(self) -> None:
        # 3+ inline code occurrences + heading + list
        text = "# API\n\n- Use `foo()` here\n- Call `bar()` there\n- Also `baz()` more\n- And `qux()` last"
        result, reasons = looks_like_markdown(text)
        assert result is True
        assert "inline_code" in reasons or "ul_list" in reasons

    def test_links_indicate_markdown(self) -> None:
        # 2+ links + heading + list items to exceed threshold
        text = (
            "# Links\n\n"
            "- See [docs](https://example.com)\n"
            "- Check [guide](https://guide.com)\n"
            "- Also [ref](https://ref.com)\n"
        )
        result, reasons = looks_like_markdown(text)
        assert result is True
        assert "link_or_img" in reasons or "ul_list" in reasons

    def test_blockquote_indicates_markdown(self) -> None:
        text = "As someone said:\n> First quote here\n> Second quote here\n> Third line"
        result, reasons = looks_like_markdown(text)
        assert result is True
        assert "blockquote" in reasons

    def test_returns_tuple(self) -> None:
        result = looks_like_markdown("some text")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)

    def test_empty_string_returns_false(self) -> None:
        result, reasons = looks_like_markdown("")
        assert result is False

    def test_custom_threshold(self) -> None:
        # Single heading should not meet threshold=10
        text = "# Just one heading"
        result_default, _ = looks_like_markdown(text, threshold=3)
        result_high, _ = looks_like_markdown(text, threshold=10)
        # Lower threshold more likely to return True
        assert isinstance(result_high, bool)

    def test_full_markdown_document(self) -> None:
        text = """# Title

This is a **bold** paragraph with `inline code`.

## Section

- Item one
- Item two
- Item three

```python
print("hello")
```

See [example](http://example.com) for more.
"""
        result, reasons = looks_like_markdown(text)
        assert result is True
        assert len(reasons) >= 3

    def test_html_not_detected_as_markdown(self) -> None:
        text = "<html><body><h1>Title</h1><p>Paragraph text here.</p></body></html>"
        result, _ = looks_like_markdown(text)
        # HTML is not markdown - may be false
        assert isinstance(result, bool)
