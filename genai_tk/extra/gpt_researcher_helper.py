"""GPT Researcher integration utilities.

This module provides simplified functions to run GPT Researcher with configurable parameters.
Main function is: run_gpt_researcher
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from genai_tk.core.llm_factory import LlmFactory
from genai_tk.core.providers import get_provider_api_key
from genai_tk.utils.config_mngr import global_config

try:
    from gpt_researcher import GPTResearcher
except ImportError as ex:
    raise ImportError(f"gpt-researcher package is required: {ex}") from ex

from loguru import logger
from pydantic import BaseModel, Field


class ResearchReport(BaseModel):
    """Container class for GPT Researcher results and metadata."""

    report: str
    context: str
    costs: float = 0.0
    images: list[str] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)


def create_gptr_config(config_name: str) -> str:
    """Get GPT Researcher config from our global configuration.

    Args:
        config_name: Name of the config section to use

    Returns:
        Path to the temporary configuration file with selected config
    """
    config_dict = global_config().get_dict(
        f"gpt_researcher.{config_name}",
        # expected_keys=["fast_llm"],
    )

    # Track which providers need API keys
    providers_needed = set()

    for llm in ["smart_llm", "fast_llm", "strategic_llm"]:
        if llm_id := config_dict.get(llm):
            factory = LlmFactory(llm=llm_id)
            litellm_name = factory.get_litellm_model_name(separator=":")
            config_dict[llm] = litellm_name
            logger.info(f"Using LiteLLM model name for {llm}: {litellm_name}")

            # Track the provider for API key injection
            providers_needed.add(factory.provider)

    # Inject API keys for the providers being used
    # GPT Researcher/LiteLLM expects specific environment variable names
    for provider in providers_needed:
        api_key = get_provider_api_key(provider)
        if api_key:
            api_key_value = api_key.get_secret_value()

            # Map provider names to LiteLLM environment variable names
            # LiteLLM uses uppercase provider name + _API_KEY
            if provider == "openrouter":
                config_dict["OPENROUTER_API_KEY"] = api_key_value
            elif provider == "openai":
                config_dict["OPENAI_API_KEY"] = api_key_value
            elif provider == "anthropic":
                config_dict["ANTHROPIC_API_KEY"] = api_key_value
            elif provider == "groq":
                config_dict["GROQ_API_KEY"] = api_key_value
            elif provider == "together":
                config_dict["TOGETHER_API_KEY"] = api_key_value
            elif provider == "deepseek":
                config_dict["DEEPSEEK_API_KEY"] = api_key_value
            elif provider == "mistralai":
                config_dict["MISTRAL_API_KEY"] = api_key_value
            else:
                # Generic pattern for other providers
                env_var_name = f"{provider.upper()}_API_KEY"
                config_dict[env_var_name] = api_key_value

            logger.info(f"Added API key for provider: {provider}")

    path = Path(tempfile.gettempdir()) / "gptr_conf.json"
    with open(path, "w") as json_file:
        json.dump(config_dict, json_file, indent=2)

    # Log config without API keys for security
    safe_config = {k: v for k, v in config_dict.items() if "API_KEY" not in k.upper()}
    logger.info(f"Using GPT Researcher config '{config_name}': {safe_config}")
    return str(path)


async def run_gpt_researcher(
    query: str, config_name: str = "default", verbose: bool = True, websocket_logger: Any | None = None, **kwargs
) -> ResearchReport:
    """Execute a GPT Researcher task with configurable parameters.

    Args:
        query: Research query
        config_name: Configuration profile name to use
        verbose: Enable verbose output
        websocket_logger: Optional websocket logger
        **kwargs: Additional parameters for GPT Researcher

    Returns:
        ResearchReport: Container with research results and metadata
    """

    try:
        import os

        config_path = create_gptr_config(config_name)

        # Read the config to set environment variables for LiteLLM
        # GPT Researcher uses LiteLLM which reads API keys from environment variables
        with open(config_path) as f:
            config_data = json.load(f)

        # Extract and set API keys as environment variables
        api_key_vars = [k for k in config_data.keys() if "API_KEY" in k.upper()]
        for key_var in api_key_vars:
            if config_data[key_var]:
                os.environ[key_var] = config_data[key_var]
                logger.debug(f"Set environment variable: {key_var}")

        researcher = GPTResearcher(
            query=query, verbose=verbose, websocket=websocket_logger, config_path=config_path, **kwargs
        )

        logger.info(f"Starting GPT Researcher with query: {query}")
        await researcher.conduct_research()
        report = await researcher.write_report()

        # Validate that we got a proper response
        if report is None or report == "":
            error_msg = "LLM returned None or empty response. Check API configuration and credentials."
            logger.error(error_msg)
            return ResearchReport(
                report=f"Research failed: {error_msg}",
                context="No research context available",
                costs=researcher.get_costs() if hasattr(researcher, "get_costs") else 0.0,
                images=[],
                sources=[],
            )

        return ResearchReport(
            report=report,
            context=str(researcher.get_research_context()),
            costs=researcher.get_costs(),
            images=[str(e) for e in researcher.get_research_images()],
            sources=researcher.get_research_sources(),
        )
    except Exception as e:
        logger.error(f"GPT Researcher failed: {e}", exc_info=True)
        # Return a basic error report instead of crashing
        return ResearchReport(
            report=f"Research failed due to error: {str(e)}\n\nThis may be caused by:\n- Invalid API credentials\n- LLM provider API issues\n- Network connectivity problems\n- Incorrect model configuration",
            context="Error occurred during research",
            costs=0.0,
            images=[],
            sources=[],
        )


# FOR QUICK TEST
if __name__ == "__main__":

    async def main():
        query = "what are the ethical risks of LLM powered AI Agents"
        result = await run_gpt_researcher(
            query=query,
        )
        return result

    result = asyncio.run(main())
    print(result.report)
