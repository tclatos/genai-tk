"""Tests for the Config class and its configuration management features."""

import os
import tempfile
from pathlib import Path
from unittest import TestCase

from genai_tk.utils.config_mngr import OmegaConfig, global_config, global_config_reload


class TestOmegaConfig(TestCase):
    """Test cases for OmegaConfig configuration management."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a temporary directory for test config files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "test_config.yaml"

        # Create test configuration content
        self.test_config_content = """                                                                     
default_config: test_env                                                                                   
paths:                                                                                                     
  data_root: /tmp/test_data                                                                                
  project: /tmp/test_project                                                                               
  models: /tmp/models                                                                                      
test_env:                                                                                                  
  llm:
    models:                                                                                                     
      default: "gpt-3.5-turbo"                                                                         
    max_tokens: 1000                                                                                       
    temperature: 0.7                                                                                       
  features:                                                                                                
    enable_caching: true                                                                                   
    enable_logging: false                                                                                  
  db:                                                                                                      
    host: "localhost"                                                                                      
    port: 5432                                                                                             
    name: "test_db"                                                                                        
prod_env:                                                                                                  
  llm: 
    models:                                                                                                    
      default: "gpt-4"                                                                                 
    max_tokens: 2000                                                                                       
    temperature: 0.1                                                                                       
  features:                                                                                                
    enable_caching: true                                                                                   
    enable_logging: true                                                                                   
  db:                                                                                                      
    host: "prod.example.com"                                                                               
    port: 5432                                                                                             
    name: "prod_db"                                                                                   
cli:                                                                                                       
  commands:                                                                                                
    - test.module:register_commands                                                                        
    - test.module2:register_commands                                                                      
ui:                                                                                                        
  app_name: "Test App"                                                                                     
  pages_dir: /tmp/pages                                                                                    
"""

        # Write test config to temporary file
        self.config_path.write_text(self.test_config_content)

        # Set environment variable for testing
        os.environ["BLUEPRINT_CONFIG"] = "test_env"

        # Create test instance
        self.config = OmegaConfig.create(self.config_path)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
        # Clean up environment
        if "BLUEPRINT_CONFIG" in os.environ:
            del os.environ["BLUEPRINT_CONFIG"]

    def test_create_config(self) -> None:
        """Test configuration loading and creation."""
        self.assertIsInstance(self.config, OmegaConfig)
        self.assertEqual(self.config.selected_config, "test_env")
        self.assertIn("test_env", self.config.root)

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

        with self.assertRaises(TypeError):
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

        with self.assertRaises(TypeError):
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
        with self.assertRaises(TypeError):
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

        with self.assertRaises(KeyError):
            self.config.get_dict("db", expected_keys=["host", "missing_key"])

    def test_select_config(self) -> None:
        """Test switching between configuration environments."""
        # Initially in test_env
        self.assertEqual(self.config.get("llm.models.default"), "gpt-3.5-turbo")
        self.assertEqual(self.config.get("llm.max_tokens"), 1000)

        # Switch to prod_env
        self.config.select_config("prod_env")
        self.assertEqual(self.config.get("llm.models.default"), "gpt-4")
        self.assertEqual(self.config.get("llm.max_tokens"), 2000)

        # Test switching to non-existent config
        with self.assertRaises(ValueError):
            self.config.select_config("nonexistent_env")

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
            with self.assertRaises(FileNotFoundError):
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

        # Test reload
        global_config_reload()
        config3 = global_config()
        self.assertIsNot(config1, config3)

    def test_invalid_config_file(self) -> None:
        """Test handling of invalid configuration files."""
        invalid_path = Path(self.temp_dir.name) / "invalid.yaml"
        invalid_path.write_text("invalid: yaml: content: [")

        with self.assertRaises(Exception):
            OmegaConfig.create(invalid_path)

    def test_missing_config_section(self) -> None:
        """Test handling of missing configuration sections."""
        with self.assertRaises(ValueError):
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
        """Test :env pseudo-key in a merged configuration file."""
        # Clean up any existing test variables
        for var in ["MERGED_ENV_VAR1", "MERGED_ENV_VAR2"]:
            if var in os.environ:
                del os.environ[var]

        # Create base config
        base_config = """
default_config: test_env
paths:
  project: /tmp/test_project
  config: /tmp/test_config

:merge:
  - ${paths.config}/merged_env.yaml

test_env:
  base_setting: "from_base"
"""
        base_config_path = Path(self.temp_dir.name) / "base_config.yaml"
        base_config_path.write_text(base_config)

        # Create config directory
        config_dir = Path(self.temp_dir.name) / "config"
        config_dir.mkdir(exist_ok=True)

        # Create merged config with :env
        merged_config = """
:env:
  MERGED_ENV_VAR1: "from_merged"
  MERGED_ENV_VAR2: "${paths.project}/merged_subdir"

test_env:
  merged_setting: "from_merged_file"
"""
        # Write to the location expected by interpolation
        merged_config_path = Path("/tmp/test_config/merged_env.yaml")
        merged_config_path.parent.mkdir(parents=True, exist_ok=True)
        merged_config_path.write_text(merged_config)

        try:
            # Create config and verify environment variables from merged file were set
            config = OmegaConfig.create(base_config_path)

            self.assertEqual(os.environ.get("MERGED_ENV_VAR1"), "from_merged")
            self.assertEqual(os.environ.get("MERGED_ENV_VAR2"), "/tmp/test_project/merged_subdir")

            # Verify merged content is present
            self.assertEqual(config.get("test_env.merged_setting"), "from_merged_file")
            self.assertEqual(config.get("test_env.base_setting"), "from_base")

            # Verify :env key was removed from config
            self.assertNotIn(":env", config.root)
        finally:
            # Clean up
            if merged_config_path.exists():
                merged_config_path.unlink()
            if merged_config_path.parent.exists():
                merged_config_path.parent.rmdir()
            for var in ["MERGED_ENV_VAR1", "MERGED_ENV_VAR2"]:
                if var in os.environ:
                    del os.environ[var]

    def test_merge_pseudo_key_with_colon(self) -> None:
        """Test :merge pseudo-key (new format with colon)."""
        # Create base config with :merge
        base_config = """
default_config: test_env
paths:
  config: /tmp/test_config

:merge:
  - ${paths.config}/additional.yaml

test_env:
  base_value: "from_base"
"""
        base_config_path = Path(self.temp_dir.name) / "merge_test.yaml"
        base_config_path.write_text(base_config)

        # Create additional config file
        additional_config = """
test_env:
  additional_value: "from_additional"
  nested:
    key: "nested_value"
"""
        additional_path = Path("/tmp/test_config/additional.yaml")
        additional_path.parent.mkdir(parents=True, exist_ok=True)
        additional_path.write_text(additional_config)

        try:
            config = OmegaConfig.create(base_config_path)

            # Verify both configs were merged
            self.assertEqual(config.get("test_env.base_value"), "from_base")
            self.assertEqual(config.get("test_env.additional_value"), "from_additional")
            self.assertEqual(config.get("test_env.nested.key"), "nested_value")

            # Verify :merge key was removed
            self.assertNotIn(":merge", config.root)
        finally:
            if additional_path.exists():
                additional_path.unlink()
            if additional_path.parent.exists():
                additional_path.parent.rmdir()

    def test_deprecated_merge_key_warning(self) -> None:
        """Test that old 'merge' key shows deprecation warning."""
        # Create config with old 'merge' key (without colon)
        old_config = """
default_config: test_env
paths:
  config: /tmp/test_config

merge:
  - ${paths.config}/old_merge.yaml

test_env:
  value: "test"
"""
        old_config_path = Path(self.temp_dir.name) / "old_merge.yaml"
        old_config_path.write_text(old_config)

        # Create file to merge
        merge_file = """
test_env:
  merged_value: "merged"
"""
        merge_path = Path("/tmp/test_config/old_merge.yaml")
        merge_path.parent.mkdir(parents=True, exist_ok=True)
        merge_path.write_text(merge_file)

        try:
            # This should work but log a warning
            config = OmegaConfig.create(old_config_path)

            # Verify merge still works
            self.assertEqual(config.get("test_env.merged_value"), "merged")

            # Verify both old 'merge' and new ':merge' keys were removed
            self.assertNotIn("merge", config.root)
            self.assertNotIn(":merge", config.root)
        finally:
            if merge_path.exists():
                merge_path.unlink()
            if merge_path.parent.exists():
                merge_path.parent.rmdir()

    def test_recursive_merge_in_merged_file(self) -> None:
        """Test that :merge pseudo-key works recursively in merged files."""
        # Create base config
        base_config = """
default_config: test_env
paths:
  config: /tmp/test_config

:merge:
  - ${paths.config}/level1.yaml

test_env:
  level0: "base"
"""
        base_config_path = Path(self.temp_dir.name) / "recursive_base.yaml"
        base_config_path.write_text(base_config)

        # Create level1 config that also has :merge
        level1_config = """
:merge:
  - ${paths.config}/level2.yaml

test_env:
  level1: "first_merge"
"""
        level1_path = Path("/tmp/test_config/level1.yaml")
        level1_path.parent.mkdir(parents=True, exist_ok=True)
        level1_path.write_text(level1_config)

        # Create level2 config
        level2_config = """
test_env:
  level2: "second_merge"
"""
        level2_path = Path("/tmp/test_config/level2.yaml")
        level2_path.write_text(level2_config)

        try:
            config = OmegaConfig.create(base_config_path)

            # Verify all levels were merged
            self.assertEqual(config.get("test_env.level0"), "base")
            self.assertEqual(config.get("test_env.level1"), "first_merge")
            self.assertEqual(config.get("test_env.level2"), "second_merge")

            # Verify :merge key was removed
            self.assertNotIn(":merge", config.root)
        finally:
            for path in [level1_path, level2_path]:
                if path.exists():
                    path.unlink()
            if level1_path.parent.exists():
                level1_path.parent.rmdir()

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
        """Test :env pseudo-key in both main file and merged file, including within config sections.

        This mimics the real-world scenario from genai-blueprint where:
        - app_conf.yaml has :env at root level
        - overrides.yaml has :env within a config section (e.g., ekg_local)
        """
        # Clean up test variables
        for var in ["MAIN_ENV_VAR", "MERGED_ENV_VAR", "SECTION_ENV_VAR"]:
            if var in os.environ:
                del os.environ[var]

        # Create main config with :env at root and merge directive
        main_config = """
default_config: test_section

paths:
  project: /tmp/test_project
  config: /tmp/test_config
  data: ${paths.project}/data

:env:
  MAIN_ENV_VAR: "from_main_root"

:merge:
  - ${paths.config}/merged_overrides.yaml

test_section:
  base_value: "from_main"
"""
        main_config_path = Path(self.temp_dir.name) / "main_with_env.yaml"
        main_config_path.write_text(main_config)

        # Create merged config with :env at root AND within a section
        merged_config = """
:env:
  MERGED_ENV_VAR: "${paths.data}/merged"

test_section:
  :env:
    SECTION_ENV_VAR: "from_section_in_merged"
  override_value: "from_merged"
  nested:
    key: "nested_value"

other_section:
  value: "other"
"""
        merged_config_path = Path("/tmp/test_config/merged_overrides.yaml")
        merged_config_path.parent.mkdir(parents=True, exist_ok=True)
        merged_config_path.write_text(merged_config)

        try:
            # Create config - this should process :env from both files
            config = OmegaConfig.create(main_config_path)

            # Verify all environment variables were set
            self.assertEqual(os.environ.get("MAIN_ENV_VAR"), "from_main_root")
            self.assertEqual(os.environ.get("MERGED_ENV_VAR"), "/tmp/test_project/data/merged")
            self.assertEqual(os.environ.get("SECTION_ENV_VAR"), "from_section_in_merged")

            # Verify merged content is present
            self.assertEqual(config.get("test_section.base_value"), "from_main")
            self.assertEqual(config.get("test_section.override_value"), "from_merged")
            self.assertEqual(config.get("test_section.nested.key"), "nested_value")
            self.assertEqual(config.get("other_section.value"), "other")

            # Verify :env keys were removed from root
            self.assertNotIn(":env", config.root)

            # Verify :env key was removed from within section
            test_section = config.root.get("test_section")
            if test_section:
                self.assertNotIn(":env", test_section)
        finally:
            # Clean up
            if merged_config_path.exists():
                merged_config_path.unlink()
            if merged_config_path.parent.exists():
                merged_config_path.parent.rmdir()
            for var in ["MAIN_ENV_VAR", "MERGED_ENV_VAR", "SECTION_ENV_VAR"]:
                if var in os.environ:
                    del os.environ[var]


if __name__ == "__main__":
    import unittest

    unittest.main()
