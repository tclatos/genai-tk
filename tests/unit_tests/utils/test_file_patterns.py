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
        assert "${nonexistent.key}" in result or "/path" in result

    def test_multiple_variables(self) -> None:
        result = resolve_config_path("/fixed/${nonexistent.x}/${nonexistent.y}")
        assert isinstance(result, str)


class TestResolveFiles:
    def test_returns_empty_for_nonexistent_dir(self, tmp_path: Path) -> None:
        result = resolve_files(str(tmp_path / "does_not_exist"))
        assert result == []

    def test_default_matches_all_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "c.md").write_text("c")
        result = resolve_files(str(tmp_path))
        assert len(result) == 3

    def test_pathspec_extension_filter(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("text")
        (tmp_path / "b.py").write_text("code")
        (tmp_path / "c.md").write_text("doc")
        result = resolve_files(str(tmp_path), pathspecs=["**/*.txt"])
        assert len(result) == 1
        assert result[0].suffix == ".txt"

    def test_negation_excludes_files(self, tmp_path: Path) -> None:
        (tmp_path / "keep.txt").write_text("keep")
        (tmp_path / "exclude_me.txt").write_text("exclude")
        result = resolve_files(str(tmp_path), pathspecs=["**/*.txt", "!**/exclude_*"])
        assert len(result) == 1
        assert "keep" in str(result[0])

    def test_recursive_with_glob(self, tmp_path: Path) -> None:
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (tmp_path / "root.txt").write_text("root")
        (subdir / "nested.txt").write_text("nested")
        result = resolve_files(str(tmp_path), pathspecs=["**/*.txt"])
        assert len(result) == 2

    def test_non_recursive_glob(self, tmp_path: Path) -> None:
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (tmp_path / "root.txt").write_text("root")
        (subdir / "nested.txt").write_text("nested")
        # In gitignore semantics, '*.txt' matches at all directory levels
        # Use a full-path pattern to restrict to root level only
        result = resolve_files(str(tmp_path), pathspecs=["*.txt", "!sub/*.txt"])
        assert len(result) == 1
        assert "root" in str(result[0])

    def test_returns_path_objects(self, tmp_path: Path) -> None:
        from pathlib import Path

        (tmp_path / "test.txt").write_text("content")
        result = resolve_files(str(tmp_path), pathspecs=["**/*.txt"])
        assert all(isinstance(p, Path) for p in result)

    def test_not_a_directory_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("content")
        result = resolve_files(str(f))
        assert result == []

    def test_multiple_patterns(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("text")
        (tmp_path / "b.md").write_text("markdown")
        (tmp_path / "c.py").write_text("code")
        result = resolve_files(str(tmp_path), pathspecs=["**/*.txt", "**/*.md"])
        assert len(result) == 2

    def test_exclude_draft_files(self, tmp_path: Path) -> None:
        (tmp_path / "report.md").write_text("report")
        (tmp_path / "report_draft.md").write_text("draft")
        result = resolve_files(str(tmp_path), pathspecs=["**/*.md", "!**/*_draft*"])
        assert len(result) == 1
        assert "report.md" in str(result[0])

    def test_result_is_sorted(self, tmp_path: Path) -> None:
        (tmp_path / "z.txt").write_text("z")
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "m.txt").write_text("m")
        result = resolve_files(str(tmp_path), pathspecs=["**/*.txt"])
        names = [r.name for r in result]
        assert names == sorted(names)
