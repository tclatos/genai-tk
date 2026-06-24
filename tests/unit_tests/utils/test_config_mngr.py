"""Tests for the OmegaConfig configuration management class."""

import os

import pytest

from genai_tk.config_mgmt.config_exceptions import (
    ConfigFileNotFoundError,
    ConfigKeyNotFoundError,
    ConfigParseError,
    ConfigTypeError,
    ConfigValidationError,
)
from genai_tk.config_mgmt.config_mngr import OmegaConfig, global_config, global_config_reload, switch_profile

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_env_config(tmp_path):
    """Activate the test_unit profile, scoped to this test, and restore pytest afterwards."""
    switch_profile("test_unit")
    global_config().use_context("test_env")
    cfg = global_config()
    yield cfg
    switch_profile("pytest")


# ---------------------------------------------------------------------------
# Basic get / type-safe accessors
# ---------------------------------------------------------------------------


def test_config_is_omega_config_instance(test_env_config) -> None:
    """Active config is an OmegaConfig."""
    assert isinstance(test_env_config, OmegaConfig)


def test_config_has_expected_contexts(test_env_config) -> None:
    """test_unit profile exposes test_env and prod_env contexts."""
    assert "test_env" in test_env_config.root
    assert "prod_env" in test_env_config.root


def test_get_string_value(test_env_config) -> None:
    """Reading a string config key returns the expected string."""
    assert test_env_config.get("llm.models.default") == "gpt-3.5-turbo"


def test_get_missing_key_returns_default(test_env_config) -> None:
    """Reading a missing key with a default returns that default."""
    assert test_env_config.get("nonexistent.key", "default_value") == "default_value"


def test_get_int_value(test_env_config) -> None:
    """Reading an integer config key returns an int."""
    assert test_env_config.get("llm.max_tokens") == 1000


def test_get_nested_int_value(test_env_config) -> None:
    """Reading a nested integer config key returns an int."""
    assert test_env_config.get("db.port") == 5432


def test_get_float_value(test_env_config) -> None:
    """Reading a float config key returns a float."""
    assert test_env_config.get("llm.temperature") == 0.7


def test_get_bool_true_value(test_env_config) -> None:
    """Reading a true boolean config key returns True."""
    assert test_env_config.get("features.enable_caching") is True


def test_get_bool_false_value(test_env_config) -> None:
    """Reading a false boolean config key returns False."""
    assert test_env_config.get("features.enable_logging") is False


def test_get_str_returns_string(test_env_config) -> None:
    """get_str returns the string value for a string key."""
    assert test_env_config.get_str("llm.models.default") == "gpt-3.5-turbo"


def test_get_str_raises_on_int_key(test_env_config) -> None:
    """get_str raises ConfigTypeError when the stored value is not a string."""
    with pytest.raises(ConfigTypeError):
        test_env_config.get_str("llm.max_tokens")


def test_get_bool_returns_bool(test_env_config) -> None:
    """get_bool returns the boolean value for a boolean key."""
    assert test_env_config.get_bool("features.enable_caching") is True


def test_get_bool_parses_string_true(test_env_config) -> None:
    """get_bool parses the string 'true' as True."""
    test_env_config.set("test_bool", "true")
    assert test_env_config.get_bool("test_bool") is True


def test_get_list_returns_list(test_env_config) -> None:
    """get_list returns the list value for a list key."""
    commands = test_env_config.get_list("cli.commands")
    assert commands == ["test.module:register_commands", "test.module2:register_commands"]


def test_get_list_raises_on_scalar_key(test_env_config) -> None:
    """get_list raises ConfigTypeError when the key holds a scalar."""
    with pytest.raises(ConfigTypeError):
        test_env_config.get_list("llm.models.default")


@pytest.mark.parametrize(
    "values,vtype",
    [
        (["item1", "item2"], str),
        ([1, 2, 3], int),
        ([1.1, 2.2], float),
    ],
)
def test_get_list_with_value_type(test_env_config, values, vtype) -> None:
    """get_list validates element types when value_type is provided."""
    test_env_config.set("typed_list", values)
    assert test_env_config.get_list("typed_list", value_type=vtype) == values


def test_get_list_raises_on_type_mismatch(test_env_config) -> None:
    """get_list raises ConfigTypeError when elements don't match value_type."""
    test_env_config.set("mixed_list", ["string", 123])
    with pytest.raises(ConfigTypeError):
        test_env_config.get_list("mixed_list", value_type=str)


def test_get_list_empty_list(test_env_config) -> None:
    """get_list returns an empty list for an empty list key."""
    test_env_config.set("empty_list", [])
    assert test_env_config.get_list("empty_list", value_type=str) == []


def test_get_list_default_value(test_env_config) -> None:
    """get_list returns the default when the key is missing."""
    result = test_env_config.get_list("nonexistent.list", default=["a", "b"], value_type=str)
    assert result == ["a", "b"]


def test_get_dict_returns_dict(test_env_config) -> None:
    """get_dict returns the dict value for a dict key."""
    db_config = test_env_config.get_dict("db")
    assert db_config == {"host": "localhost", "port": 5432, "name": "test_db"}


def test_get_dict_validates_expected_keys(test_env_config) -> None:
    """get_dict passes when all expected_keys are present."""
    result = test_env_config.get_dict("db", expected_keys=["host", "port", "name"])
    assert result["host"] == "localhost"


def test_get_dict_raises_on_missing_expected_key(test_env_config) -> None:
    """get_dict raises ConfigValidationError when an expected_key is absent."""
    with pytest.raises(ConfigValidationError):
        test_env_config.get_dict("db", expected_keys=["host", "missing_key"])


# ---------------------------------------------------------------------------
# Context switching
# ---------------------------------------------------------------------------


def test_use_context_switches_values(test_env_config) -> None:
    """use_context('prod_env') makes prod_env keys accessible."""
    test_env_config.use_context("prod_env")
    assert test_env_config.get("llm.models.default") == "gpt-4"
    assert test_env_config.get("llm.max_tokens") == 2000


def test_use_context_raises_on_nonexistent(test_env_config) -> None:
    """use_context raises ConfigKeyNotFoundError for an unknown context."""
    with pytest.raises(ConfigKeyNotFoundError):
        test_env_config.use_context("nonexistent_env")


# ---------------------------------------------------------------------------
# Runtime overrides
# ---------------------------------------------------------------------------


def test_set_overrides_value(test_env_config) -> None:
    """set() overrides an existing config value at runtime."""
    test_env_config.set("llm.models.default", "custom-model")
    assert test_env_config.get("llm.models.default") == "custom-model"


def test_set_creates_nested_key(test_env_config) -> None:
    """set() creates a new nested key when it doesn't exist."""
    test_env_config.set("new.nested.value", "test")
    assert test_env_config.get("new.nested.value") == "test"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def test_get_dir_path_returns_path(test_env_config, tmp_path) -> None:
    """get_dir_path returns a Path object for an existing directory."""
    test_env_config.set("test_dir", str(tmp_path))
    result = test_env_config.get_dir_path("test_dir")
    assert result == tmp_path


def test_get_dir_path_creates_directory(test_env_config, tmp_path) -> None:
    """get_dir_path creates the directory when create_if_not_exists=True."""
    new_dir = tmp_path / "new_subdir"
    test_env_config.set("new_dir", str(new_dir))
    result = test_env_config.get_dir_path("new_dir", create_if_not_exists=True)
    assert result.exists()
    assert result.is_dir()


def test_get_file_path_returns_path(test_env_config, tmp_path) -> None:
    """get_file_path returns a Path object for an existing file."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    test_env_config.set("test_file", str(test_file))
    assert test_env_config.get_file_path("test_file") == test_file


def test_get_file_path_raises_on_missing_file(test_env_config) -> None:
    """get_file_path raises ConfigFileNotFoundError for a non-existent file."""
    test_env_config.set("missing_file", "/nonexistent/path/file.txt")
    with pytest.raises(ConfigFileNotFoundError):
        test_env_config.get_file_path("missing_file")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def test_merge_with_additional_config(test_env_config, tmp_path) -> None:
    """merge_with() adds keys from an additional YAML file."""
    additional = tmp_path / "extra.yaml"
    additional.write_text("additional_env:\n  new_setting: from_additional\n")
    test_env_config.merge_with(str(additional))
    assert test_env_config.get("additional_env.new_setting") == "from_additional"


# ---------------------------------------------------------------------------
# DSN / interpolation
# ---------------------------------------------------------------------------


def test_get_dsn_returns_dsn(test_env_config) -> None:
    """get_dsn returns the DSN string as-is."""
    dsn = "postgresql://user:pass@localhost:5432/testdb"
    test_env_config.set("test_dsn", dsn)
    assert test_env_config.get_dsn("test_dsn") == dsn


def test_get_dsn_with_driver_override(test_env_config) -> None:
    """get_dsn inserts the driver into the scheme when driver= is provided."""
    test_env_config.set("test_dsn", "postgresql://user:pass@localhost:5432/testdb")
    result = test_env_config.get_dsn("test_dsn", driver="asyncpg")
    assert result == "postgresql+asyncpg://user:pass@localhost:5432/testdb"


def test_env_var_interpolation(test_env_config) -> None:
    """OmegaConf ${oc.env:VAR} syntax resolves to the actual env var value."""
    os.environ["TEST_CFG_VAR"] = "interpolated_value"
    test_env_config.set("test_interpolation", "${oc.env:TEST_CFG_VAR}")
    assert test_env_config.get("test_interpolation") == "interpolated_value"


# ---------------------------------------------------------------------------
# Singleton / reload
# ---------------------------------------------------------------------------


def test_global_config_returns_same_instance() -> None:
    """global_config() returns the same object on repeated calls."""
    assert global_config() is global_config()


def test_global_config_reload_creates_new_instance() -> None:
    """global_config_reload() replaces the singleton."""
    original = global_config()
    global_config_reload()
    reloaded = global_config()
    assert original is not reloaded
    # Restore pytest profile
    switch_profile("pytest")


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_missing_config_key_raises(test_env_config) -> None:
    """Reading a completely absent key raises ConfigKeyNotFoundError."""
    with pytest.raises(ConfigKeyNotFoundError):
        test_env_config.get("missing_section.key")


def test_invalid_yaml_file_raises(tmp_path) -> None:
    """Creating config from a malformed YAML file raises ConfigParseError."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("invalid: yaml: [")
    with pytest.raises(ConfigParseError):
        OmegaConfig.create(bad_yaml)


def test_deeply_nested_set_and_get(test_env_config) -> None:
    """set() and get() round-trip through deeply nested keys."""
    test_env_config.set("deep.nested.structure.value", 42)
    assert test_env_config.get("deep.nested.structure.value") == 42


def test_deeply_nested_list(test_env_config) -> None:
    """get_list works on deeply nested list keys."""
    test_env_config.set("deep.nested.list", [1, 2, 3])
    assert test_env_config.get_list("deep.nested.list") == [1, 2, 3]


def test_deeply_nested_dict(test_env_config) -> None:
    """get_dict works on deeply nested dict keys."""
    test_env_config.set("deep.nested.dict", {"inner": "test"})
    assert test_env_config.get_dict("deep.nested.dict") == {"inner": "test"}


# ---------------------------------------------------------------------------
# :env pseudo-key
# ---------------------------------------------------------------------------


def test_env_pseudo_key_sets_env_vars(tmp_path) -> None:
    """The :env: block in a YAML file sets the listed environment variables."""
    for var in ["TEST_ENV_VAR1", "TEST_ENV_VAR2", "TEST_ENV_VAR3"]:
        os.environ.pop(var, None)

    cfg_file = tmp_path / "env_config.yaml"
    cfg_file.write_text(
        "paths:\n"
        "  project: /tmp/test_project\n"
        "  data: /tmp/test_data\n"
        ":env:\n"
        '  TEST_ENV_VAR1: "simple_value"\n'
        '  TEST_ENV_VAR2: "value_with_spaces"\n'
        '  TEST_ENV_VAR3: "${paths.project}/subdir"\n'
        "test_env:\n"
        '  setting: "test"\n'
    )

    OmegaConfig.create(cfg_file)

    assert os.environ.get("TEST_ENV_VAR1") == "simple_value"
    assert os.environ.get("TEST_ENV_VAR2") == "value_with_spaces"
    assert os.environ.get("TEST_ENV_VAR3") == "/tmp/test_project/subdir"


def test_env_pseudo_key_is_removed_from_config(tmp_path) -> None:
    """:env: block is not present in the resulting config tree."""
    cfg_file = tmp_path / "env_config.yaml"
    cfg_file.write_text(":env:\n  TEST_REMOVAL_VAR: value\n")
    config = OmegaConfig.create(cfg_file)
    assert ":env" not in config.root
