"""LLM resolution bridge between genai-tk and deepagents-cli.

Translates genai-tk LLM identifiers (tags, IDs) into LangChain ``BaseChatModel``
instances that can be passed directly to ``deepagents_cli.agent.create_cli_agent()``,
bypassing deepagents-cli's own model creation pipeline.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from genai_tk.agents.deepagent_cli.models import DeepagentConfig, DeepagentProfile
from genai_tk.core.llm_factory import LlmFactory, get_llm
from genai_tk.utils.config_mngr import global_config


def resolve_model_from_profile(
    profile: DeepagentProfile,
    llm_override: str | None,
    config: DeepagentConfig,
) -> BaseChatModel:
    """Resolve the LLM for a specific profile run.

    Priority order:
    1. Explicit ``llm_override`` from CLI flag.
    2. Profile ``llm`` field.
    3. Global ``config.default_model``.
    4. ``llm.models.default`` tag from OmegaConf global config.

    Args:
        profile: Active profile (provides ``llm`` field if set).
        llm_override: LLM identifier from CLI ``--llm`` flag, or ``None``.
        config: Global deepagent config (provides ``default_model``).

    Returns:
        Configured ``BaseChatModel`` instance ready for ``create_cli_agent()``.
    """
    identifier = llm_override or profile.llm or config.default_model
    return _resolve_identifier(identifier)


def resolve_model(
    llm_override: str | None,
    config: DeepagentConfig,
) -> BaseChatModel:
    """Resolve the LLM without a profile context.

    Args:
        llm_override: LLM identifier from CLI, or ``None``.
        config: Global deepagent config.

    Returns:
        Configured ``BaseChatModel`` instance.
    """
    identifier = llm_override or config.default_model
    return _resolve_identifier(identifier)


def _resolve_identifier(identifier: str | None) -> BaseChatModel:
    """Resolve a genai-tk LLM identifier to a BaseChatModel.

    Falls back to the global ``llm.models.default`` tag when ``identifier``
    is ``None``.

    Args:
        identifier: LLM tag, ID, or ``None``.

    Returns:
        Configured ``BaseChatModel`` instance.

    Raises:
        ValueError: When the identifier cannot be resolved.
    """
    if identifier is None:
        # Fall back to global default tag
        try:
            identifier = global_config().get("llm.models.default", None)
        except Exception:
            pass

    if not identifier:
        raise ValueError(
            "No LLM specified for deepagent. Set 'deepagent.default_model' in "
            "config/agents/deepagent.yaml, or use the --llm flag."
        )

    resolved_id, error_msg = LlmFactory.resolve_llm_identifier_safe(identifier)
    if error_msg or not resolved_id:
        raise ValueError(error_msg or f"Could not resolve LLM identifier: {identifier!r}")

    return get_llm(llm=resolved_id)
