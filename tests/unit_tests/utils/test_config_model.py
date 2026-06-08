"""Unit tests for ConfigModel base class."""

import textwrap
from pathlib import Path

import pytest

from genai_tk.config_mgmt.config_model import ConfigModel


class SampleConfig(ConfigModel):
    host: str = "localhost"
    port: int = 8080
    name: str = ""


class NestedConfig(ConfigModel):
    title: str = ""
    items: list[str] = []


# ---------------------------------------------------------------------------
# from_yaml — string source
# ---------------------------------------------------------------------------


class TestFromYamlString:
    def test_basic_yaml_string(self):
        cfg = SampleConfig.from_yaml("host: 0.0.0.0\nport: 9090")
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 9090

    def test_defaults_when_empty(self):
        cfg = SampleConfig.from_yaml("{}")
        assert cfg.host == "localhost"
        assert cfg.port == 8080

    def test_top_key_extraction(self):
        yaml_str = textwrap.dedent("""\
            server:
              host: 10.0.0.1
              port: 443
        """)
        cfg = SampleConfig.from_yaml(yaml_str, top_key="server")
        assert cfg.host == "10.0.0.1"
        assert cfg.port == 443

    def test_resolve_false_skips_global_config(self):
        # Should not raise even if global config is broken
        cfg = SampleConfig.from_yaml("host: test\nport: 1234", resolve=False)
        assert cfg.host == "test"
        assert cfg.port == 1234

    def test_nested_model(self):
        yaml_str = "title: My List\nitems:\n  - alpha\n  - beta"
        cfg = NestedConfig.from_yaml(yaml_str)
        assert cfg.title == "My List"
        assert cfg.items == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# from_yaml — Path source
# ---------------------------------------------------------------------------


class TestFromYamlPath:
    def test_load_from_file(self, tmp_path: Path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("host: filehost\nport: 7777\n")
        cfg = SampleConfig.from_yaml(yaml_file, resolve=False)
        assert cfg.host == "filehost"
        assert cfg.port == 7777

    def test_top_key_from_file(self, tmp_path: Path):
        yaml_file = tmp_path / "nested.yaml"
        yaml_file.write_text("wrapper:\n  host: inner\n  port: 5555\n")
        cfg = SampleConfig.from_yaml(yaml_file, top_key="wrapper", resolve=False)
        assert cfg.host == "inner"
        assert cfg.port == 5555


# ---------------------------------------------------------------------------
# from_yaml — dict source
# ---------------------------------------------------------------------------


class TestFromYamlDict:
    def test_dict_passthrough(self):
        cfg = SampleConfig.from_yaml({"host": "dicthost", "port": 3333})
        assert cfg.host == "dicthost"
        assert cfg.port == 3333

    def test_dict_with_top_key(self):
        data = {"section": {"host": "sec", "port": 2222}}
        cfg = SampleConfig.from_yaml(data, top_key="section")
        assert cfg.host == "sec"

    def test_invalid_source_type_raises(self):
        with pytest.raises(TypeError, match="source must be"):
            SampleConfig.from_yaml(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# from_yaml_dict — dict-keyed factory
# ---------------------------------------------------------------------------


class TestFromYamlDict_DictKeyed:
    def test_basic_dict_keyed(self):
        yaml_str = textwrap.dedent("""\
            servers:
              alpha:
                host: a.example.com
                port: 1001
              beta:
                host: b.example.com
                port: 1002
        """)
        items = SampleConfig.from_yaml_dict(yaml_str, top_key="servers")
        assert len(items) == 2
        assert items["alpha"].host == "a.example.com"
        assert items["alpha"].port == 1001
        assert items["beta"].host == "b.example.com"
        assert items["beta"].name == "beta"  # injected

    def test_inject_name_disabled(self):
        yaml_str = textwrap.dedent("""\
            items:
              one:
                host: h1
                port: 1
        """)
        items = SampleConfig.from_yaml_dict(yaml_str, top_key="items", inject_name=False)
        assert items["one"].name == ""  # default, not injected

    def test_empty_source_returns_empty_dict(self):
        items = SampleConfig.from_yaml_dict("{}")
        assert items == {}

    def test_from_path(self, tmp_path: Path):
        yaml_file = tmp_path / "multi.yaml"
        yaml_file.write_text("entries:\n  x:\n    host: xh\n    port: 10\n  y:\n    host: yh\n    port: 20\n")
        items = SampleConfig.from_yaml_dict(yaml_file, top_key="entries", resolve=False)
        assert len(items) == 2
        assert items["x"].host == "xh"


# ---------------------------------------------------------------------------
# from_config — global config accessor
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_from_config_delegates_to_section(self, monkeypatch):
        """Verify from_config calls global_config().section()."""
        from unittest.mock import MagicMock, patch

        mock_gc = MagicMock()
        mock_gc.section.return_value = SampleConfig(host="mocked", port=42)

        with patch("genai_tk.config_mgmt.config_mngr.global_config", return_value=mock_gc):
            cfg = SampleConfig.from_config("my_key")

        mock_gc.section.assert_called_once_with("my_key", SampleConfig)
        assert cfg.host == "mocked"
        assert cfg.port == 42
