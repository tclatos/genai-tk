"""Backward-compatible re-export. Use ``genai_tk.agents.langchain.agent`` instead."""

from genai_tk.agents.langchain.agent import (  # noqa: F401
    _prepare_rich_agent,
    _render_final_message,
    run_langchain_agent_direct,
    run_langchain_agent_shell,
)
