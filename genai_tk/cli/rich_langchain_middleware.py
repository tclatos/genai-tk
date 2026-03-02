"""Backward-compatible re-export. Use ``genai_tk.agents.langchain.rich_middleware`` instead."""

from genai_tk.agents.langchain.rich_middleware import (  # noqa: F401
    RichToolCallMiddleware,
    SingleToolExecutorMiddleware,
    ToolCallLimitMiddleware,
    create_rich_agent_middlewares,
)
