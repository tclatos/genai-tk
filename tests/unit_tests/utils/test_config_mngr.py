"""Tests for the Config class and its configuration management features."""

import os
import tempfile
from pathlib import Path
from unittest import TestCase

from genai_tk.utils.config_exceptions import (
    ConfigFileNotFoundError,
    ConfigKeyNotFoundError,
    ConfigParseError,
    ConfigTypeError,
    ConfigValidationError,
)
from genai_tk.utils.config_mngr import OmegaConfig, global_config, global_config_reload, switch_profile


class TestOmegaConfig(TestCase):
    """Test cases for OmegaConfig configuration management."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Temporary directory for tests that need isolated config files
        self.temp_dir = tempfile.TemporaryDirectory()

        # Switch to the test_unit profile and activate the test_env context
        switch_profile("test_unit")
        global_config().use_context("test_env")
        self.config = global_config()

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
        # Restore pytest profile so other tests are not affected
        switch_profile("pytest")

    def test_create_config(self) -> None:
        """Test configuration loading and creation."""
        self.assertIsInstance(self.config, OmegaConfig)
        self.assertEqual(self.config.active_context, "test_env")
        self.assertIn("test_env", self.config.root)
        self.assertIn("prod_env", self.config.root)

    def test_get_string_value(self) -> None:
        """Test getting string configuration values."""
        model = self.config.get("llm.models.default")
        self.assertEqual(model, "gpt-3.5-turbo")

        # Test with default value
        missing_value = self.config.get("nonexistent.key", "default_value")
        self.assertEqual(missing_value, "default_value")

    def test_get_int_value(self) -> None:
        """Test getting integer configuration values."""
        max_tokens = self.config.get("llm.max_tokens")
        self.assertEqual(max_tokens, 1000)

        port = self.config.get("db.port")
        self.assertEqual(port, 5432)

    def test_get_float_value(self) -> None:
        """Test getting float configuration values."""
        temperature = self.config.get("llm.temperature")
        self.assertEqual(temperature, 0.7)

    def test_get_bool_value(self) -> None:
        """Test getting boolean configuration values."""
        enable_caching = self.config.get("features.enable_caching")
        self.assertTrue(enable_caching)

        enable_logging = self.config.get("features.enable_logging")
        self.assertFalse(enable_logging)

    def test_get_str_method(self) -> None:
        """Test get_str type-safe method."""
        model = self.config.get_str("llm.models.default")
        self.assertEqual(model, "gpt-3.5-turbo")

        with self.assertRaises(ConfigTypeError):
            self.config.get_str("llm.max_tokens")

    def test_get_bool_method(self) -> None:
        """Test get_bool type-safe method."""
        enable_caching = self.config.get_bool("features.enable_caching")
        self.assertTrue(enable_caching)

        # Test string boolean conversion with runtime override
        self.config.set("test_bool", "true")
        bool_value = self.config.get_bool("test_bool")
        self.assertTrue(bool_value)

    def test_get_list_method(self) -> None:
        """Test get_list type-safe method."""
        commands = self.config.get_list("cli.commands")
        self.assertEqual(commands, ["test.module:register_commands", "test.module2:register_commands"])

        with self.assertRaises(ConfigTypeError):
            self.config.get_list("llm.models.default")

    def test_get_list_with_value_type(self) -> None:
        """Test get_list with value type validation."""
        # Test with string type validation
        self.config.set("test_string_list", ["item1", "item2", "item3"])
        string_list = self.config.get_list("test_string_list", value_type=str)
        self.assertEqual(string_list, ["item1", "item2", "item3"])

        # Test with integer type validation
        self.config.set("test_int_list", [1, 2, 3, 4, 5])
        int_list = self.config.get_list("test_int_list", value_type=int)
        self.assertEqual(int_list, [1, 2, 3, 4, 5])

        # Test with float type validation
        self.config.set("test_float_list", [1.1, 2.2, 3.3])
        float_list = self.config.get_list("test_float_list", value_type=float)
        self.assertEqual(float_list, [1.1, 2.2, 3.3])

        # Test type validation failure
        self.config.set("test_mixed_list", ["string", 123, 3.14])
        with self.assertRaises(ConfigTypeError):
            self.config.get_list("test_mixed_list", value_type=str)

        # Test with empty list
        self.config.set("test_empty_list", [])
        empty_list = self.config.get_list("test_empty_list", value_type=str)
        self.assertEqual(empty_list, [])

        # Test with default value and type validation
        default_list = self.config.get_list("nonexistent.list", default=["default1", "default2"], value_type=str)
        self.assertEqual(default_list, ["default1", "default2"])

    def test_get_dict_method(self) -> None:
        """Test get_dict type-safe method."""
        db_config = self.config.get_dict("db")
        expected = {"host": "localhost", "port": 5432, "name": "test_db"}
        self.assertEqual(db_config, expected)

        # Test with expected keys validation
        db_config_validated = self.config.get_dict("db", expected_keys=["host", "port", "name"])
        self.assertEqual(db_config_validated, expected)

        with self.assertRaises(ConfigValidationError):
            self.config.get_dict("db", expected_keys=["host", "missing_key"])

    def test_use_context(self) -> None:
        """Test switching between configuration contexts."""
        # Initially in test_env
        self.assertEqual(self.config.get("llm.models.default"), "gpt-3.5-turbo")
        self.assertEqual(self.config.get("llm.max_tokens"), 1000)

        # Switch to prod_env
        self.config.use_context("prod_env")
        self.assertEqual(self.config.get("llm.models.default"), "gpt-4")
        self.assertEqual(self.config.get("llm.max_tokens"), 2000)

        # Test switching to non-existent config
        with self.assertRaises(ConfigKeyNotFoundError):
            self.config.use_context("nonexistent_env")

    def test_set_runtime_override(self) -> None:
        """Test setting runtime configuration overrides."""
        original_model = self.config.get("llm.models.default")
        self.assertEqual(original_model, "gpt-3.5-turbo")

        # Set override
        self.config.set("llm.models.default", "custom-model")
        self.assertEqual(self.config.get("llm.models.default"), "custom-model")

        # Test setting nested override
        self.config.set("new.nested.value", "test")
        self.assertEqual(self.config.get("new.nested.value"), "test")

    def test_get_dir_path(self) -> None:
        """Test getting directory paths."""
        # Test with existing directory
        with tempfile.TemporaryDirectory() as temp_dir:
            self.config.set("test_dir", temp_dir)
            dir_path = self.config.get_dir_path("test_dir")
            self.assertEqual(str(dir_path), temp_dir)

        # Test creating non-existent directory
        temp_path = Path(self.temp_dir.name) / "new_dir"
        self.config.set("new_dir", str(temp_path))
        dir_path = self.config.get_dir_path("new_dir", create_if_not_exists=True)
        self.assertTrue(dir_path.exists())
        self.assertTrue(dir_path.is_dir())

    def test_get_file_path(self) -> None:
        """Test getting file paths."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write("test content")
            temp_path = temp_file.name

        try:
            self.config.set("test_file", temp_path)
            file_path = self.config.get_file_path("test_file")
            self.assertEqual(str(file_path), temp_path)

            # Test non-existent file
            self.config.set("missing_file", "/nonexistent/path/file.txt")
            with self.assertRaises(ConfigFileNotFoundError):
                self.config.get_file_path("missing_file")
        finally:
            os.unlink(temp_path)

    def test_merge_with_additional_config(self) -> None:
        """Test merging additional configuration files."""
        additional_config = """                                                                            
additional_env:                                                                                            
  new_setting: "from_additional"                                                                           
  llm:                                                                                                     
    provider: "openai"                                                                                     
"""
        additional_path = Path(self.temp_dir.name) / "additional.yaml"
        additional_path.write_text(additional_config)

        # Merge additional config
        self.config.merge_with(str(additional_path))
        self.assertIn("additional_env", self.config.root)
        self.assertEqual(self.config.get("additional_env.new_setting"), "from_additional")

    def test_dsn_generation(self) -> None:
        """Test DSN URL generation for database connections."""
        test_dsn = "postgresql://user:pass@localhost:5432/testdb"
        self.config.set("test_dsn", test_dsn)

        dsn = self.config.get_dsn("test_dsn")
        self.assertEqual(dsn, test_dsn)

        # Test with driver override
        dsn_with_driver = self.config.get_dsn("test_dsn", driver="asyncpg")
        self.assertEqual(dsn_with_driver, "postgresql+asyncpg://user:pass@localhost:5432/testdb")

    def test_environment_variable_interpolation(self) -> None:
        """Test OmegaConf environment variable interpolation."""
        os.environ["TEST_VAR"] = "interpolated_value"

        # Use existing config to test interpolation
        self.config.set("test_interpolation", "${oc.env:TEST_VAR}")
        value = self.config.get("test_interpolation")
        self.assertEqual(value, "interpolated_value")

    def test_global_config_singleton(self) -> None:
        """Test the global config singleton."""
        config1 = global_config()
        config2 = global_config()

        # Should be the same instance
        self.assertIs(config1, config2)

        # Test reload creates a new instance
        global_config_reload()
        config3 = global_config()
        self.assertIsNot(config1, config3)
        # tearDown will restore the pytest profile

    def test_invalid_config_file(self) -> None:
        """Test handling of invalid configuration files."""
        invalid_path = Path(self.temp_dir.name) / "invalid.yaml"
        invalid_path.write_text("invalid: yaml: content: [")

        with self.assertRaises(ConfigParseError):
            OmegaConfig.create(invalid_path)

    def test_missing_config_section(self) -> None:
        """Test handling of missing configuration sections."""
        with self.assertRaises(ConfigKeyNotFoundError):
            self.config.get("missing_section.key")

    def test_nested_access(self) -> None:
        """Test deeply nested configuration access."""
        # Use runtime configuration to test nested access
        self.config.set("deep.nested.structure.value", 42)
        self.config.set("deep.nested.structure.list", [1, 2, 3])
        self.config.set("deep.nested.structure.dict", {"inner": "test"})

        # Test deep access
        value = self.config.get("deep.nested.structure.value")
        self.assertEqual(value, 42)

        nested_list = self.config.get_list("deep.nested.structure.list")
        self.assertEqual(nested_list, [1, 2, 3])

        nested_dict = self.config.get_dict("deep.nested.structure.dict")
        self.assertEqual(nested_dict, {"inner": "test"})

    def test_env_pseudo_key(self) -> None:
        """Test :env pseudo-key for setting environment variables."""
        # Clean up any existing test variables
        for var in ["TEST_ENV_VAR1", "TEST_ENV_VAR2", "TEST_ENV_VAR3"]:
            if var in os.environ:
                del os.environ[var]

        env_config = """
default_config: test_env
paths:
  project: /tmp/test_project
  data: /tmp/test_data

:env:
  TEST_ENV_VAR1: "simple_value"
  TEST_ENV_VAR2: "value_with_spaces"
  TEST_ENV_VAR3: "${paths.project}/subdir"

test_env:
  setting: "test"
"""
        env_config_path = Path(self.temp_dir.name) / "env_config.yaml"
        env_config_path.write_text(env_config)

        # Create config and verify environment variables were set
        config = OmegaConfig.create(env_config_path)

        self.assertEqual(os.environ.get("TEST_ENV_VAR1"), "simple_value")
        self.assertEqual(os.environ.get("TEST_ENV_VAR2"), "value_with_spaces")
        self.assertEqual(os.environ.get("TEST_ENV_VAR3"), "/tmp/test_project/subdir")

        # Verify :env key was removed from config
        self.assertNotIn(":env", config.root)

        # Clean up
        for var in ["TEST_ENV_VAR1", "TEST_ENV_VAR2", "TEST_ENV_VAR3"]:
            if var in os.environ:
                del os.environ[var]

    def test_env_pseudo_key_in_merged_file(self) -> None:
        """Test :env pseudo-key in a configuration file matched by :merge: patterns."""
        for var in ["MERGED_ENV_VAR1"]:
            if var in os.environ:
                del os.environ[var]

        # Create a subdirectory with the merged config
        extra_dir = Path(self.temp_dir.name) / "extra"
        extra_dir.mkdir()
        merged_config_path = extra_dir / "merged_env.yaml"
        merged_config_path.write_text(
            ':env:\n  MERGED_ENV_VAR1: "from_merged"\n\ntest_env:\n  merged_setting: "from_merged_file"\n'
        )

        # Base config uses pathspec pattern to include the merged file
        base_config = (
            "default_config: test_env\n"
            "paths:\n  project: /tmp/test_project\n\n"
            ":merge:\n  - extra/*.yaml\n\n"
            'test_env:\n  base_setting: "from_base"\n'
        )
        base_config_path = Path(self.temp_dir.name) / "app_conf.yaml"
        base_config_path.write_text(base_config)

        try:
            config = OmegaConfig.create(base_config_path)
            self.assertEqual(os.environ.get("MERGED_ENV_VAR1"), "from_merged")
            self.assertEqual(config.get("test_env.merged_setting"), "from_merged_file")
            self.assertEqual(config.get("test_env.base_setting"), "from_base")
            self.assertNotIn(":env", config.root)
        finally:
            for var in ["MERGED_ENV_VAR1"]:
                if var in os.environ:
                    del os.environ[var]

    def test_merge_pseudo_key_with_colon(self) -> None:
        """Test :merge pseudo-key with pathspec patterns includes matched files."""
        extra_dir = Path(self.temp_dir.name) / "extra"
        extra_dir.mkdir()
        additional_path = extra_dir / "additional.yaml"
        additional_path.write_text(
            'test_env:\n  additional_value: "from_additional"\n  nested:\n    key: "nested_value"\n'
        )

        base_config = 'default_config: test_env\n:merge:\n  - extra/*.yaml\n\ntest_env:\n  base_value: "from_base"\n'
        base_config_path = Path(self.temp_dir.name) / "app_conf.yaml"
        base_config_path.write_text(base_config)

        config = OmegaConfig.create(base_config_path)

        self.assertEqual(config.get("test_env.base_value"), "from_base")
        self.assertEqual(config.get("test_env.additional_value"), "from_additional")
        self.assertEqual(config.get("test_env.nested.key"), "nested_value")
        self.assertNotIn(":merge", config.root)

    def test_merge_with_method_supports_env_and_merge(self) -> None:
        """Test that merge_with method processes :env and :merge pseudo-keys."""
        # Clean up test variables
        for var in ["MERGE_WITH_VAR"]:
            if var in os.environ:
                del os.environ[var]

        # Start with base config
        base_config = """
default_config: test_env
paths:
  project: /tmp/project

test_env:
  original: "value"
"""
        base_config_path = Path(self.temp_dir.name) / "merge_with_base.yaml"
        base_config_path.write_text(base_config)
        config = OmegaConfig.create(base_config_path)

        # Create a file to merge that has :env
        merge_with_config = """
:env:
  MERGE_WITH_VAR: "${paths.project}/data"

test_env:
  added: "from_merge_with"
"""
        merge_with_path = Path(self.temp_dir.name) / "merge_with_file.yaml"
        merge_with_path.write_text(merge_with_config)

        # Use merge_with method
        config.merge_with(str(merge_with_path))

        # Verify env var was set
        self.assertEqual(os.environ.get("MERGE_WITH_VAR"), "/tmp/project/data")

        # Verify config was merged
        self.assertEqual(config.get("test_env.original"), "value")
        self.assertEqual(config.get("test_env.added"), "from_merge_with")

        # Verify :env key was removed
        self.assertNotIn(":env", config.root)

        # Clean up
        if "MERGE_WITH_VAR" in os.environ:
            del os.environ["MERGE_WITH_VAR"]

    def test_env_in_both_main_and_merged_files(self) -> None:
        """Test :env pseudo-key in both main file and a file matched by :merge:, including within config sections.

        This mimics the real-world scenario where app_conf.yaml has :env at root level
        and a merged file has :env within a config section.
        """
        for var in ["MAIN_ENV_VAR", "MERGED_ENV_VAR", "SECTION_ENV_VAR"]:
            if var in os.environ:
                del os.environ[var]

        # Create merged config
        extra_dir = Path(self.temp_dir.name) / "extra"
        extra_dir.mkdir()
        merged_config_path = extra_dir / "merged_overrides.yaml"
        merged_config_path.write_text(
            ':env:\n  MERGED_ENV_VAR: "from_merged"\n\n'
            'test_section:\n  :env:\n    SECTION_ENV_VAR: "from_section_in_merged"\n'
            '  override_value: "from_merged"\n  nested:\n    key: "nested_value"\n\n'
            'other_section:\n  value: "other"\n'
        )

        main_config = (
            "default_config: test_section\n\n"
            "paths:\n  project: /tmp/test_project\n  data: /tmp/test_project/data\n\n"
            ':env:\n  MAIN_ENV_VAR: "from_main_root"\n\n'
            ":merge:\n  - extra/*.yaml\n\n"
            'test_section:\n  base_value: "from_main"\n'
        )
        main_config_path = Path(self.temp_dir.name) / "app_conf.yaml"
        main_config_path.write_text(main_config)

        try:
            config = OmegaConfig.create(main_config_path)

            self.assertEqual(os.environ.get("MAIN_ENV_VAR"), "from_main_root")
            self.assertEqual(os.environ.get("MERGED_ENV_VAR"), "from_merged")
            self.assertEqual(os.environ.get("SECTION_ENV_VAR"), "from_section_in_merged")
            self.assertEqual(config.get("test_section.base_value"), "from_main")
            self.assertEqual(config.get("test_section.override_value"), "from_merged")
            self.assertEqual(config.get("test_section.nested.key"), "nested_value")
            self.assertEqual(config.get("other_section.value"), "other")
            self.assertNotIn(":env", config.root)
            test_section = config.root.get("test_section")
            if test_section:
                self.assertNotIn(":env", test_section)
        finally:
            for var in ["MAIN_ENV_VAR", "MERGED_ENV_VAR", "SECTION_ENV_VAR"]:
                if var in os.environ:
                    del os.environ[var]


if __name__ == "__main__":
    import unittest

    unittest.main()
