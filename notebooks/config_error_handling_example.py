"""Configuration Error Handling Example

This module demonstrates best practices for handling configuration errors
in notebooks and scripts using the structured exception hierarchy.
"""

from genai_tk.utils.config_exceptions import (
    ConfigError,
    ConfigKeyNotFoundError,
    ConfigTypeError,
    ConfigValidationError,
)
from genai_tk.utils.config_mngr import global_config


def example_safe_config_access():
    """Demonstrate safe configuration access with proper error handling."""

    print("=" * 60)
    print("Configuration Error Handling Examples")
    print("=" * 60)

    # Example 1: Handling missing keys
    print("\n1. Safe key access with default fallback:")
    try:
        api_key = global_config().get_str("api.openai_key")
        print("✓ API key loaded successfully")
    except ConfigKeyNotFoundError as e:
        print(f"⚠ Key not found: {e.message}")
        print(f"💡 {e.suggestion}")
        api_key = None  # Use default/None

    # Example 2: Handling type errors
    print("\n2. Type-safe configuration access:")
    try:
        max_tokens = global_config().get_int("llm.max_tokens", default=1000)
        print(f"✓ Max tokens: {max_tokens}")
    except ConfigTypeError as e:
        print(f"⚠ Type error: Expected {e.expected_type.__name__}, got {e.actual_type.__name__}")
        print(f"💡 {e.suggestion}")
        max_tokens = 1000  # Use fallback

    # Example 3: Handling file paths
    print("\n3. Safe file path access:")
    try:
        data_dir = global_config().get_dir_path("paths.data")
        print(f"✓ Data directory: {data_dir}")
    except ConfigKeyNotFoundError as e:
        print(f"⚠ Path not configured: {e.message}")
        print(f"💡 {e.suggestion}")
        from pathlib import Path

        data_dir = Path("./data")  # Use default

    # Example 4: Handling list configurations
    print("\n4. Safe list access:")
    try:
        providers = global_config().get_list("llm.providers", value_type=str)
        print(f"✓ Found {len(providers)} providers")
    except ConfigTypeError as e:
        print(f"⚠ Configuration error: {e.message}")
        print(f"💡 {e.suggestion}")
        providers = []  # Use empty list

    # Example 5: Generic error handling
    print("\n5. Generic configuration error handling:")
    try:
        some_value = global_config().get("some.nested.key")
        print(f"✓ Value loaded: {some_value}")
    except ConfigError as e:
        print(f"⚠ Configuration error: {e.message}")
        if hasattr(e, "suggestion") and e.suggestion:
            print(f"💡 {e.suggestion}")
        some_value = None

    print("\n" + "=" * 60)
    print("Error handling complete - application can continue safely")
    print("=" * 60)


def example_critical_config_check():
    """Demonstrate when to fail fast on configuration errors."""

    print("\n" + "=" * 60)
    print("Critical Configuration Check Example")
    print("=" * 60)

    required_keys = ["paths.config", "llm.providers", "embeddings.providers"]

    try:
        print("\nChecking required configuration...")
        for key in required_keys:
            value = global_config().get(key)
            print(f"✓ {key}: OK")

        print("\n✓ All required configuration present")
        return True

    except ConfigKeyNotFoundError as e:
        print("\n❌ CRITICAL: Missing required configuration")
        print(f"Key: {e.key}")
        print(f"Message: {e.message}")
        print(f"💡 {e.suggestion}")
        print("\n⚠ Application cannot start without this configuration")
        return False

    except ConfigValidationError as e:
        print("\n❌ CRITICAL: Invalid configuration")
        print(f"Message: {e.message}")
        print(f"💡 {e.suggestion}")
        print("\n⚠ Fix configuration before proceeding")
        return False

    except ConfigError as e:
        print("\n❌ CRITICAL: Configuration error")
        print(f"Message: {e.message}")
        if hasattr(e, "suggestion") and e.suggestion:
            print(f"💡 {e.suggestion}")
        return False


def main():
    """Run all examples."""

    # Example of graceful error handling
    example_safe_config_access()

    # Example of critical checks
    config_ok = example_critical_config_check()

    if config_ok:
        print("\n✓ Ready to start application")
    else:
        print("\n❌ Application cannot start - fix configuration first")


if __name__ == "__main__":
    main()
