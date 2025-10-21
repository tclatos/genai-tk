"""LangChain v1 compatibility layer.

This module provides re-exports and helper functions to ease migration from 
LangChain v0.x to v1.x. It bridges import changes and provides utilities for
handling messages and models consistently.

Warning: This compatibility layer is temporary and will be removed after
         the migration is complete. Do not use for new code.
"""

import warnings
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

# Re-export common message classes
__all__ = [
    "AIMessage",
    "BaseMessage", 
    "HumanMessage",
    "SystemMessage",
    "ToolMessage",
    "BaseTool",
    "tool",
    "create_chat_model",
    "message_content_to_text",
]

def _warn_deprecated(name: str) -> None:
    """Issue deprecation warning for compatibility layer usage."""
    warnings.warn(
        f"genai_tk.compat.langchain.{name} is deprecated. "
        f"Use the direct LangChain v1 import instead.",
        DeprecationWarning,
        stacklevel=3
    )

def create_chat_model(model: str, **kwargs: Any):
    """Create a chat model using LangChain v1 init_chat_model.
    
    This is a thin wrapper around langchain.chat_models.init_chat_model
    to provide a consistent interface during migration.
    
    Args:
        model: Model identifier (e.g., "gpt-3.5-turbo", "claude-3-sonnet")
        **kwargs: Additional arguments passed to init_chat_model
        
    Returns:
        BaseChatModel: Initialized chat model
    """
    _warn_deprecated("create_chat_model")
    return init_chat_model(model, **kwargs)

def message_content_to_text(message: BaseMessage) -> str:
    """Extract text content from a message, handling both string and content blocks.
    
    In LangChain v1, message content can be either a string or a list of content
    blocks. This helper ensures we can reliably extract text for logging and
    assertions during migration.
    
    Args:
        message: LangChain message object
        
    Returns:
        str: Text content of the message
    """
    if isinstance(message.content, str):
        return message.content
    elif isinstance(message.content, list):
        # Handle content blocks - extract text from text blocks
        text_parts = []
        for content_block in message.content:
            if isinstance(content_block, dict) and content_block.get("type") == "text":
                text_parts.append(content_block.get("text", ""))
            elif isinstance(content_block, str):
                text_parts.append(content_block)
        return " ".join(text_parts)
    else:
        return str(message.content)