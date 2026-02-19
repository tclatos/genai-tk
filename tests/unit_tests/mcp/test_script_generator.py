"""Unit tests for genai_tk.mcp.script_generator."""

import tempfile
from pathlib import Path

from genai_tk.mcp.script_generator import generate_server_script, write_server_script


class TestGenerateServerScript:
    def test_contains_server_name(self) -> None:
        code = generate_server_script("my-server")
        assert "my-server" in code

    def test_contains_main_guard(self) -> None:
        code = generate_server_script("x")
        assert '__name__ == "__main__"' in code

    def test_calls_serve(self) -> None:
        code = generate_server_script("x")
        assert "serve(" in code

    def test_config_none_by_default(self) -> None:
        code = generate_server_script("x")
        assert "_CONFIG_PATH: str | None = None" in code

    def test_config_path_embedded(self) -> None:
        code = generate_server_script("x", config_path="/tmp/servers.yaml")
        assert "'/tmp/servers.yaml'" in code

    def test_has_docstring(self) -> None:
        code = generate_server_script("myserver")
        assert '"""' in code


class TestWriteServerScript:
    def test_file_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "server_test.py"
            result = write_server_script("test", output=out)
            assert result == out
            assert out.exists()

    def test_default_output_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            import os

            original_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                result = write_server_script("my_server")
                assert result.name == "server_my_server.py"
            finally:
                os.chdir(original_cwd)
