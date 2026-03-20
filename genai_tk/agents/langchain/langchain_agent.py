"""High-level agent interface for creating and running LangChain-based agents.

Wraps the low-level factory and config machinery into a simple, production-friendly
class that works with or without a YAML profile.

Example:
```python
# From a YAML profile
agent = LangchainAgent("Research")
result = agent.run("Summarize recent AI news")

# Ad-hoc (no profile needed)
from langchain_community.tools.tavily_search import TavilySearchResults

agent = LangchainAgent(llm="fast_model", tools=[TavilySearchResults()])
result = agent.run("What happened today in tech?")

# Async
result = await agent.arun("Explain quantum computing")

# Async streaming
async for chunk in agent.astream("Tell me a story"):
    print(chunk, end="", flush=True)

# Interactive shell
await agent.arun_shell()
```
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, Literal

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from genai_tk.agents.langchain.config import AgentProfileConfig, AgentType

SandboxType = Literal["local", "docker"]


class LangchainAgent(BaseModel):
    """Production-friendly interface for LangChain-based agents (react and deep).

    Can be created from a named YAML profile, ad-hoc from raw parameters, or a
    combination of both (ad-hoc params overlay the profile).

    Args:
        profile_name: Name of a profile in ``langchain.yaml``. When omitted an
            ad-hoc react agent is built entirely from the other parameters.
        llm: LLM identifier (e.g. ``"gpt_41mini@openai"`` or a tag like
            ``"fast_model"``). Overrides the profile's ``llm`` field.
        tools: Pre-built ``BaseTool`` instances appended to (or replacing) the
            profile's tool list.
        agent_type: Override the profile's agent type (``"react"`` or ``"deep"``).
        system_prompt: Override or provide a system prompt.
        mcp_servers: Additional MCP server names to enable.
        checkpointer: When ``True`` a ``MemorySaver`` is attached so the agent
            remembers conversation history across turns.
        details: When ``True`` the ``RichToolCallMiddleware`` shows full panels
            for every LLM call and tool call instead of the compact summary.
        sandbox: Sandbox override: ``"docker"`` starts an ``AioSandboxBackend``
            container and promotes the profile to ``deep`` if needed. ``None``
            or ``"local"`` uses the profile's configured backend unchanged.
    """

    profile_name: str | None = None
    llm: str | None = None
    tools: list[BaseTool] = Field(default_factory=list)
    agent_type: AgentType | None = None
    system_prompt: str | None = None
    mcp_servers: list[str] = Field(default_factory=list)
    checkpointer: bool = False
    details: bool = False
    sandbox: SandboxType | None = None
    vnc: bool = False
    keep_sandbox: bool = False

    # Internal – not part of the public schema
    _profile: AgentProfileConfig | None = None
    _agent: Any = None  # compiled LangGraph agent, populated lazily

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, profile_name: str | None = None, **data: Any) -> None:
        super().__init__(profile_name=profile_name, **data)

    @model_validator(mode="after")
    def _resolve_profile(self) -> LangchainAgent:
        """Build the resolved AgentProfileConfig eagerly (sync-safe)."""
        if self.profile_name:
            self._profile = self._load_profile()
        else:
            self._profile = self._build_adhoc_profile()
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> str:
        """Run the agent synchronously and return the final response.

        Args:
            query: The user query to execute.

        Returns:
            The agent's final response as a plain string.
        """
        return asyncio.run(self.arun(query))

    async def arun(self, query: str) -> str:
        """Run the agent asynchronously and return the final response.

        Args:
            query: The user query to execute.

        Returns:
            The agent's final response as a plain string.
        """
        agent = await self._ensure_initialized()
        result = await agent.ainvoke(
            {"messages": query},
            {"configurable": {"thread_id": "1"}},
        )
        return _extract_content(result)

    async def astream(self, query: str) -> AsyncGenerator[str, None]:
        """Stream the agent's response token by token.

        Args:
            query: The user query to execute.

        Yields:
            Text chunks from the agent's response.
        """
        agent = await self._ensure_initialized()
        async for chunk in agent.astream({"messages": query}):
            content = _extract_content(chunk)
            if content:
                yield content

    async def arun_shell(self) -> None:
        """Launch an interactive prompt shell for multi-turn conversation.

        Delegates to the Rich-based shell in ``agent_cli``. Type ``/quit`` to exit.
        """
        from genai_tk.agents.langchain.agent_cli import run_langchain_agent_shell

        await run_langchain_agent_shell(self)

    async def close(self) -> None:
        """Stop any running backend (e.g. Docker sandbox).

        When ``keep_sandbox`` is True the container and server are left running;
        references are detached so GC won't kill them.
        """
        if self._agent is None:
            return
        backend = getattr(self._agent, "_backend", None)
        if backend is not None:
            if self.keep_sandbox and hasattr(backend, "detach"):
                vnc = getattr(backend, "_base_url", "")
                if vnc:
                    logger.info(f"Sandbox kept alive — VNC: {vnc}/vnc/index.html?autoconnect=true")
                    logger.info("Use 'cli sandbox list' to see running containers, 'cli sandbox stop' to clean up.")
                backend.detach()
            elif hasattr(backend, "stop"):
                await backend.stop()
        self._agent = None

    # Context-manager support
    async def __aenter__(self) -> LangchainAgent:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_initialized(self) -> Any:
        """Lazily create and cache the underlying compiled LangGraph agent."""
        if self._agent is None:
            from genai_tk.agents.langchain.factory import create_langchain_agent

            assert self._profile is not None
            profile = self._profile

            # Apply --sandbox CLI override
            if self.sandbox and self.sandbox != "local":
                from genai_tk.agents.langchain.config import BackendConfig

                if self.sandbox == "docker":
                    sandbox_backend = BackendConfig(type="docker")
                else:
                    raise ValueError(
                        f"Unsupported sandbox type for langchain agent: {self.sandbox!r}. Use: local, docker"
                    )

                # Skill directories will be bind-mounted into the Docker
                # container by the factory, so keep them in the profile.
                update: dict[str, Any] = {"backend": sandbox_backend}
                if profile.type != "deep":
                    update["type"] = "deep"
                    logger.debug(f"Sandbox override '{self.sandbox}': switched profile type to deep")

                profile = profile.model_copy(update=update)

            self._agent = await create_langchain_agent(
                profile,
                extra_tools=self.tools or None,
                extra_mcp_servers=self.mcp_servers or None,
                force_memory_checkpointer=self.checkpointer,
                details=self.details,
            )
            logger.debug(f"LangchainAgent initialized: profile={self._profile.name}")

            # Auto-open VNC in the default browser for visual debugging
            if self.vnc:
                backend = getattr(self._agent, "_backend", None)
                base_url = getattr(backend, "_base_url", "") if backend else ""
                if base_url:
                    import webbrowser  # noqa: PLC0415

                    vnc_url = f"{base_url}/vnc/index.html?autoconnect=true"
                    logger.info(f"Opening VNC in browser: {vnc_url}")
                    webbrowser.open(vnc_url)

        return self._agent

    def _load_profile(self) -> AgentProfileConfig:
        """Load and merge a named profile from ``langchain.yaml``."""
        from genai_tk.agents.langchain.config import load_unified_config, resolve_profile

        cfg = load_unified_config()
        profile = resolve_profile(cfg, self.profile_name, type_override=self.agent_type)  # type: ignore[arg-type]

        # Apply ad-hoc overrides on top of the resolved profile
        overrides: dict[str, Any] = {}
        if self.llm:
            overrides["llm"] = self.llm
        if self.system_prompt:
            overrides["system_prompt"] = self.system_prompt
        if self.mcp_servers:
            overrides["mcp_servers"] = list(profile.mcp_servers) + list(self.mcp_servers)

        if overrides:
            profile = profile.model_copy(update=overrides)

        return profile

    def _build_adhoc_profile(self) -> AgentProfileConfig:
        """Build a minimal AgentProfileConfig from ad-hoc parameters (no YAML needed)."""
        return AgentProfileConfig(
            name="adhoc",
            type=self.agent_type or "react",
            llm=self.llm,
            system_prompt=self.system_prompt,
            mcp_servers=list(self.mcp_servers),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_content(result: Any) -> str:
    """Extract the final assistant message content from a LangGraph result.

    Handles LangGraph state dicts, direct ``BaseMessage`` objects, and plain strings.

    Args:
        result: Output from ``ainvoke`` or ``astream``.

    Returns:
        The message content as a plain string.
    """
    if isinstance(result, dict) and "messages" in result:
        messages: list[BaseMessage] = result["messages"]
        final = messages[-1] if messages else None
    else:
        final = result

    if final is None:
        return ""

    content = getattr(final, "content", str(final))
    if isinstance(content, list):
        return "\n".join(str(block) for block in content)
    return str(content)
