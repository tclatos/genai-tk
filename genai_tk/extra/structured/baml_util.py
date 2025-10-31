"""Shared utilities for BAML-based structured extraction.

This module provides common functionality for working with BAML clients,
including dynamic loading, type inspection, and validation.
"""

import importlib
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Type, get_args, get_origin

from baml_py import ClientRegistry
from loguru import logger

from genai_tk.utils.config_mngr import global_config


def load_baml_client(config_name: str = "default") -> tuple[Any, Any]:
    """Load BAML client modules dynamically from config.

    Args:
        config_name: Name of the structured config to use

    Returns:
        Tuple of (types_module, async_client_instance)

    Raises:
        ValueError: If BAML client package not found in config
        ImportError: If BAML client modules cannot be imported
    """
    config_key = f"structured.{config_name}.baml_client"
    baml_client_package = global_config().get_str(config_key)

    if not baml_client_package:
        raise ValueError(
            f"BAML client package not found in config at '{config_key}'. "
            f"Please configure it in YAML config file (overrides.yaml or else)"
        )

    logger.debug(f"Loading BAML client from package: {baml_client_package}")

    try:
        types_module = importlib.import_module(f"{baml_client_package}.types")
    except ImportError as e:
        raise ImportError(f"Failed to import types module from '{baml_client_package}.types': {e}") from e

    try:
        async_client_module = importlib.import_module(f"{baml_client_package}.async_client")
        baml_async_client = async_client_module.b
    except ImportError as e:
        raise ImportError(f"Failed to import async client from '{baml_client_package}.async_client': {e}") from e
    except AttributeError as e:
        raise AttributeError(f"Async client module does not have expected 'b' attribute: {e}") from e

    return types_module, baml_async_client


def get_return_type_from_baml_function(baml_function_method: Callable[..., Awaitable[Any]]) -> Type[Any] | None:
    """Extract the return type from a BAML function signature.

    Args:
        baml_function_method: BAML async function method

    Returns:
        Return type class (BaseModel, str, int, etc.) if found, None otherwise
    """
    try:
        signature = inspect.signature(baml_function_method)
        return_annotation = signature.return_annotation

        if return_annotation == inspect.Signature.empty:
            return None

        # Handle Awaitable[T] or Coroutine[Any, Any, T]
        origin = get_origin(return_annotation)
        if origin is not None:
            args = get_args(return_annotation)
            if args:
                # For Awaitable[T], get T
                return args[-1] if isinstance(args, tuple) else args

        # Direct type check
        return return_annotation

    except Exception as e:
        logger.debug(f"Could not extract return type: {e}")

    return None


def get_baml_function(
    baml_async_client: Any, function_name: str
) -> tuple[Callable[..., Awaitable[Any]], Type[Any] | None]:
    """Get a BAML function and extract its return type.

    Args:
        baml_async_client: BAML async client instance
        function_name: Name of the BAML function

    Returns:
        Tuple of (wrapped_function, return_type_or_none)

    Raises:
        AttributeError: If function not found in client
    """
    try:
        baml_function_method = getattr(baml_async_client, function_name)
    except AttributeError as e:
        raise AttributeError(f"Unknown BAML function '{function_name}' in async client: {e}") from e

    # Extract return type
    return_type = get_return_type_from_baml_function(baml_function_method)

    # Create wrapper function that supports baml_options and dynamic parameters
    async def baml_function_wrapper(*args: Any, baml_options: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        if baml_options:
            return await baml_function_method(*args, baml_options=baml_options, **kwargs)
        return await baml_function_method(*args, **kwargs)

    return baml_function_wrapper, return_type


def create_baml_client_registry(llm_identifier: str, temperature: float = 0.0) -> ClientRegistry:
    """Create a BAML client registry for a specific LLM configuration.

    This function takes an LLM id or tag, resolves it to get the actual LLM information
    (model name, provider, temperature, etc.), and creates a BAML client registry
    configured for that specific LLM.
    """

    from genai_tk.core.llm_factory import LlmFactory

    try:
        resolved_llm_id = LlmFactory.resolve_llm_identifier(llm_identifier)
    except ValueError as e:
        raise ValueError(f"Unknown LLM identifier '{llm_identifier}': {e}") from e

    try:
        llm_factory = LlmFactory(llm_id=resolved_llm_id, llm_params={"temperature": temperature})
        llm_info = llm_factory.info
        llm = llm_factory.get()
        llm_dict = llm.model_dump()

        if llm.__class__.__name__ == "ChatOpenAI":
            provider = "openai-generic"
            model = llm_dict["model_name"]
        else:
            raise ValueError(f"Provider not (yet?) supported for BAML: {llm_info.provider}")

        # Map langchain field names to BAML field names
        options = {"model": model}
        if "openai_api_key" in llm_dict:
            options["api_key"] = llm_dict["openai_api_key"].get_secret_value()
        if "temperature" in llm_dict:
            options["temperature"] = llm_dict["temperature"]
        if "openai_api_base" in llm_dict:
            options["base_url"] = llm_dict["openai_api_base"]

        cr = ClientRegistry()
        cr.add_llm_client(name=llm_identifier, provider=provider, options=options)
        cr.set_primary(llm_identifier)
        return cr
    except Exception as e:
        raise ValueError(f"Failed to get LLM info for '{resolved_llm_id}': {e}") from e
