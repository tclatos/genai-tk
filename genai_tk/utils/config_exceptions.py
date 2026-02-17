"""Configuration exception hierarchy for structured error handling.

This module defines a hierarchy of exceptions for configuration-related errors,
providing better error reporting and handling capabilities across the application.

Example:
    ```python
    from genai_tk.utils.config_exceptions import ConfigKeyNotFoundError

    try:
        value = global_config().get("nonexistent.key")
    except ConfigKeyNotFoundError as e:
        print(f"Configuration missing: {e.key}")
        print(f"Suggestion: {e.suggestion}")
    ```
"""

from __future__ import annotations


class ConfigError(Exception):
    """Base exception for all configuration-related errors."""

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        """Initialize configuration error.

        Args:
            message: Error description
            suggestion: Optional suggestion for fixing the error
        """
        self.message = message
        self.suggestion = suggestion

        full_message = message
        if suggestion:
            full_message = f"{message}\n💡 Suggestion: {suggestion}"
        super().__init__(full_message)


class ConfigFileError(ConfigError):
    """Exception raised when a configuration file cannot be loaded or parsed."""

    def __init__(
        self, file_path: str, message: str, suggestion: str | None = None, original_error: Exception | None = None
    ) -> None:
        """Initialize configuration file error.

        Args:
            file_path: Path to the configuration file
            message: Error description
            suggestion: Optional suggestion for fixing the error
            original_error: Original exception that caused this error
        """
        self.file_path = file_path
        self.original_error = original_error

        full_message = f"Configuration file error in '{file_path}': {message}"
        if original_error:
            full_message += f"\nCaused by: {type(original_error).__name__}: {original_error}"

        super().__init__(full_message, suggestion)


class ConfigFileNotFoundError(ConfigFileError):
    """Exception raised when a required configuration file is not found."""

    def __init__(self, file_path: str, searched_paths: list[str] | None = None) -> None:
        """Initialize configuration file not found error.

        Args:
            file_path: Path to the missing configuration file
            searched_paths: Optional list of paths that were searched
        """
        self.searched_paths = searched_paths or []

        message = f"Configuration file not found: '{file_path}'"
        if searched_paths:
            searched = "\n  - ".join(searched_paths)
            message += f"\nSearched in:\n  - {searched}"

        suggestion = (
            "Ensure the configuration file exists and the path is correct. "
            "Check that BLUEPRINT_CONFIG environment variable is set correctly."
        )

        super().__init__(file_path, message, suggestion)


class ConfigParseError(ConfigFileError):
    """Exception raised when a configuration file cannot be parsed."""

    def __init__(self, file_path: str, line_number: int | None = None, original_error: Exception | None = None) -> None:
        """Initialize configuration parse error.

        Args:
            file_path: Path to the configuration file
            line_number: Optional line number where the error occurred
            original_error: Original parsing exception
        """
        self.line_number = line_number

        message = "Failed to parse YAML configuration"
        if line_number:
            message += f" at line {line_number}"

        suggestion = (
            "Check YAML syntax: proper indentation, quotes, and structure. "
            "Use a YAML validator if needed: https://www.yamllint.com/"
        )

        super().__init__(file_path, message, suggestion, original_error)


class ConfigKeyError(ConfigError):
    """Base exception for configuration key-related errors."""

    def __init__(
        self, key: str, message: str, suggestion: str | None = None, available_keys: list[str] | None = None
    ) -> None:
        """Initialize configuration key error.

        Args:
            key: Configuration key that caused the error
            message: Error description
            suggestion: Optional suggestion for fixing the error
            available_keys: Optional list of available keys at this level
        """
        self.key = key
        self.available_keys = available_keys or []

        full_message = f"Configuration key '{key}': {message}"
        if available_keys:
            keys_list = ", ".join(available_keys[:10])
            if len(available_keys) > 10:
                keys_list += f" ... ({len(available_keys) - 10} more)"
            full_message += f"\nAvailable keys: {keys_list}"

        super().__init__(full_message, suggestion)


class ConfigKeyNotFoundError(ConfigKeyError):
    """Exception raised when a required configuration key is not found."""

    def __init__(
        self, key: str, available_keys: list[str] | None = None, similar_keys: list[str] | None = None
    ) -> None:
        """Initialize configuration key not found error.

        Args:
            key: Configuration key that was not found
            available_keys: Optional list of available keys at this level
            similar_keys: Optional list of keys with similar names
        """
        self.similar_keys = similar_keys or []

        message = "not found"

        suggestion_parts = []
        if similar_keys:
            similar = ", ".join(f"'{k}'" for k in similar_keys[:3])
            suggestion_parts.append(f"Did you mean: {similar}?")
        suggestion_parts.append(
            f"Add this key to your configuration file or use .get('{key}', default=...) to provide a default value."
        )
        suggestion = " ".join(suggestion_parts)

        super().__init__(key, message, suggestion, available_keys)


class ConfigTypeError(ConfigKeyError):
    """Exception raised when a configuration value has an unexpected type."""

    def __init__(self, key: str, expected_type: type | str, actual_type: type, actual_value: object = None) -> None:
        """Initialize configuration type error.

        Args:
            key: Configuration key
            expected_type: Expected type or type description
            actual_type: Actual type of the value
            actual_value: Optional actual value (for better error messages)
        """
        self.expected_type = expected_type
        self.actual_type = actual_type
        self.actual_value = actual_value

        expected_name = expected_type if isinstance(expected_type, str) else expected_type.__name__
        message = f"expected type '{expected_name}', got '{actual_type.__name__}'"

        if actual_value is not None:
            message += f" (value: {actual_value!r})"

        suggestion = (
            f"Check the configuration value for '{key}' and ensure it matches the expected type. "
            "Verify quotes for strings, remove quotes for numbers, and use proper YAML syntax for lists/dicts."
        )

        super().__init__(key, message, suggestion)


class ConfigValueError(ConfigKeyError):
    """Exception raised when a configuration value is invalid."""

    def __init__(self, key: str, value: object, reason: str, valid_values: list[str] | None = None) -> None:
        """Initialize configuration value error.

        Args:
            key: Configuration key
            value: Invalid value
            reason: Reason why the value is invalid
            valid_values: Optional list of valid values
        """
        self.value = value
        self.reason = reason
        self.valid_values = valid_values or []

        message = f"invalid value '{value}': {reason}"

        suggestion_parts = []
        if valid_values:
            valid = ", ".join(f"'{v}'" for v in valid_values[:5])
            if len(valid_values) > 5:
                valid += f" ... ({len(valid_values) - 5} more)"
            suggestion_parts.append(f"Valid values: {valid}")
        suggestion = " ".join(suggestion_parts) if suggestion_parts else None

        super().__init__(key, message, suggestion)


class ConfigInterpolationError(ConfigError):
    """Exception raised when OmegaConf interpolation fails."""

    def __init__(self, key: str, interpolation: str, original_error: Exception | None = None) -> None:
        """Initialize configuration interpolation error.

        Args:
            key: Configuration key with failed interpolation
            interpolation: The interpolation expression that failed
            original_error: Original OmegaConf exception
        """
        self.key = key
        self.interpolation = interpolation
        self.original_error = original_error

        message = f"Failed to resolve interpolation in '{key}': {interpolation}"
        if original_error:
            message += f"\nCaused by: {original_error}"

        suggestion = (
            "Check that all referenced keys exist and have valid values. "
            "Use ${key.path} for config keys and ${oc.env:VAR_NAME,default} for environment variables."
        )

        super().__init__(message, suggestion)


class ConfigValidationError(ConfigError):
    """Exception raised when configuration validation fails."""

    def __init__(self, errors: list[str], config_name: str | None = None) -> None:
        """Initialize configuration validation error.

        Args:
            errors: List of validation error messages
            config_name: Optional name of the configuration being validated
        """
        self.errors = errors
        self.config_name = config_name

        prefix = "Configuration validation failed"
        if config_name:
            prefix += f" for '{config_name}'"

        error_list = "\n  - ".join(errors)
        message = f"{prefix}:\n  - {error_list}"

        suggestion = "Fix the validation errors listed above and try again."

        super().__init__(message, suggestion)


class ConfigMergeError(ConfigError):
    """Exception raised when merging configuration files fails."""

    def __init__(self, target_file: str, merge_file: str, reason: str, original_error: Exception | None = None) -> None:
        """Initialize configuration merge error.

        Args:
            target_file: Target configuration file
            merge_file: File being merged
            reason: Reason for merge failure
            original_error: Original exception that caused the merge to fail
        """
        self.target_file = target_file
        self.merge_file = merge_file
        self.reason = reason
        self.original_error = original_error

        message = f"Failed to merge '{merge_file}' into '{target_file}': {reason}"
        if original_error:
            message += f"\nCaused by: {type(original_error).__name__}: {original_error}"

        suggestion = (
            "Check that the merge file exists and has compatible structure. Ensure both files use valid YAML syntax."
        )

        super().__init__(message, suggestion)


class ConfigEnvironmentError(ConfigError):
    """Exception raised when environment variables are missing or invalid."""

    def __init__(self, env_var: str, reason: str, required_for: str | None = None) -> None:
        """Initialize configuration environment error.

        Args:
            env_var: Environment variable name
            reason: Reason for the error
            required_for: Optional description of what requires this variable
        """
        self.env_var = env_var
        self.required_for = required_for

        message = f"Environment variable '{env_var}': {reason}"
        if required_for:
            message += f" (required for {required_for})"

        suggestion = (
            f"Set the environment variable '{env_var}' in your .env file or shell environment. "
            "Check .env.example for required variables."
        )

        super().__init__(message, suggestion)
