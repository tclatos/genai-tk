"""Unit tests for genai_tk.utils.hashing."""

from pathlib import Path

import pytest

from genai_tk.utils.hashing import buffer_digest, file_digest


class TestBufferDigest:
    def test_default_algorithm_returns_string(self) -> None:
        result = buffer_digest(b"hello world")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_xxh3_64_algorithm(self) -> None:
        result = buffer_digest(b"hello", algorithm="xxh3_64")
        assert isinstance(result, str)

    def test_xxh3_128_algorithm(self) -> None:
        result = buffer_digest(b"hello", algorithm="xxh3_128")
        assert isinstance(result, str)

    def test_sha256_algorithm(self) -> None:
        result = buffer_digest(b"hello", algorithm="sha256")
        assert len(result) == 64  # SHA-256 hex = 64 chars

    def test_md5_algorithm(self) -> None:
        result = buffer_digest(b"hello", algorithm="md5")
        assert len(result) == 32  # MD5 hex = 32 chars

    def test_empty_bytes(self) -> None:
        result = buffer_digest(b"", algorithm="sha256")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_deterministic_same_input(self) -> None:
        data = b"test data"
        assert buffer_digest(data) == buffer_digest(data)

    def test_different_inputs_differ(self) -> None:
        assert buffer_digest(b"foo") != buffer_digest(b"bar")

    def test_invalid_algorithm_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            buffer_digest(b"hello", algorithm="nonexistent")  # type: ignore

    def test_xxh3_64_shorter_than_xxh3_128(self) -> None:
        data = b"some data"
        result_64 = buffer_digest(data, algorithm="xxh3_64")
        result_128 = buffer_digest(data, algorithm="xxh3_128")
        assert len(result_64) < len(result_128)


class TestFileDigest:
    def test_file_returns_string(self, tmp_path: Path) -> None:
        f = tmp_path / "test.bin"
        f.write_bytes(b"file content")
        result = file_digest(f)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_file_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "test.bin"
        f.write_bytes(b"some bytes")
        assert file_digest(f) == file_digest(f)

    def test_file_different_from_other(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"AAA")
        f2.write_bytes(b"BBB")
        assert file_digest(f1) != file_digest(f2)

    def test_file_sha256(self, tmp_path: Path) -> None:
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello")
        result = file_digest(f, algorithm="sha256")
        assert len(result) == 64

    def test_file_matches_buffer_digest(self, tmp_path: Path) -> None:
        data = b"exact bytes"
        f = tmp_path / "test.bin"
        f.write_bytes(data)
        assert file_digest(f) == buffer_digest(data)

    def test_large_file(self, tmp_path: Path) -> None:
        # 100KB of data to test chunked reading
        f = tmp_path / "large.bin"
        f.write_bytes(b"x" * 100_000)
        result = file_digest(f)
        assert isinstance(result, str)

    def test_all_algorithms_on_same_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.bin"
        f.write_bytes(b"test")
        results = {algo: file_digest(f, algorithm=algo) for algo in ("xxh3_64", "xxh3_128", "sha256", "md5")}  # type: ignore
        # All must be strings and different
        values = list(results.values())
        assert all(isinstance(v, str) for v in values)
        assert len(set(values)) == 4  # All unique
