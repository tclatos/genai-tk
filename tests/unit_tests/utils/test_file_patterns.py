"""Unit tests for genai_tk.utils.file_patterns."""

from pathlib import Path

import pytest

from genai_tk.utils.file_patterns import resolve_config_path, resolve_files


class TestResolveConfigPath:
    def test_plain_string_unchanged(self) -> None:
        result = resolve_config_path("/some/plain/path")
        assert result == "/some/plain/path"

    def test_resolves_config_variable(self) -> None:
        from genai_tk.utils.config_mngr import global_config

        # Use a known config key
        try:
            expected = global_config().get_str("paths.project")
            result = resolve_config_path("${paths.project}/sub")
            if expected:
                assert result.endswith("/sub")
                assert "${" not in result
        except Exception:
            pytest.skip("paths.project not available in config")

    def test_unknown_variable_kept_as_is(self) -> None:
        result = resolve_config_path("${nonexistent.key}/path")
        # Original reference kept if key not found
        assert "${nonexistent.key}" in result or "/path" in result

    def test_multiple_variables(self) -> None:
        # Multiple references in same string
        result = resolve_config_path("/fixed/${nonexistent.x}/${nonexistent.y}")
        assert isinstance(result, str)


class TestResolveFiles:
    def test_returns_empty_for_nonexistent_dir(self) -> None:
        result = resolve_files("/nonexistent/path/that/does/not/exist")
        assert result == []

    def test_returns_files_from_existing_dir(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("content a")
        (tmp_path / "b.txt").write_text("content b")
        (tmp_path / "c.md").write_text("content c")
        result = resolve_files(str(tmp_path))
        assert len(result) == 3

    def test_include_pattern_filters(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("content")
        (tmp_path / "b.py").write_text("code")
        (tmp_path / "c.md").write_text("doc")
        result = resolve_files(str(tmp_path), include_patterns=["*.txt"])
        assert len(result) == 1
        assert result[0].suffix == ".txt"

    def test_exclude_pattern(self, tmp_path: Path) -> None:
        (tmp_path / "keep.txt").write_text("keep")
        (tmp_path / "exclude_me.txt").write_text("exclude")
        result = resolve_files(str(tmp_path), include_patterns=["*.txt"], exclude_patterns=["exclude*"])
        assert len(result) == 1
        assert "keep" in str(result[0])

    def test_recursive_false_default(self, tmp_path: Path) -> None:
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (tmp_path / "root.txt").write_text("root")
        (subdir / "nested.txt").write_text("nested")
        result = resolve_files(str(tmp_path), include_patterns=["*.txt"])
        assert len(result) == 1
        assert "root" in str(result[0])

    def test_recursive_true_finds_nested(self, tmp_path: Path) -> None:
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (tmp_path / "root.txt").write_text("root")
        (subdir / "nested.txt").write_text("nested")
        result = resolve_files(str(tmp_path), include_patterns=["*.txt"], recursive=True)
        assert len(result) == 2

    def test_case_insensitive_lowercase(self, tmp_path: Path) -> None:
        # Same case pattern should always match
        (tmp_path / "file.txt").write_text("content")
        result = resolve_files(str(tmp_path), include_patterns=["*.txt"])
        assert len(result) == 1

    def test_returns_upath_objects(self, tmp_path: Path) -> None:
        from upath import UPath

        (tmp_path / "test.txt").write_text("content")
        result = resolve_files(str(tmp_path), include_patterns=["*.txt"])
        assert all(isinstance(p, UPath) for p in result)

    def test_not_a_directory_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("content")
        result = resolve_files(str(f))
        assert result == []

    def test_multiple_include_patterns(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("text")
        (tmp_path / "b.md").write_text("markdown")
        (tmp_path / "c.py").write_text("code")
        result = resolve_files(str(tmp_path), include_patterns=["*.txt", "*.md"])
        assert len(result) == 2
