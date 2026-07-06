"""Unit tests for the unified LangChain agent factory.

Exercises ``create_langchain_agent`` end-to-end with the REAL factory and the
``parrot_local@fake`` model (no mocking of the factory itself).  LangChain's
``create_agent`` and deepagents' ``create_deep_agent`` are *spied* (wrapped so
the real call still runs) to capture the wiring kwargs — those are external
library boundaries, not genai_tk internals.

The MCP "tools loaded" path mocks ``MultiServerMCPClient`` (the MCP SDK that
would spawn subprocesses) and the config accessor to avoid launching real
subprocesses — the factory's own branching/combining logic still runs for real.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from deepagents import create_deep_agent as _real_create_deep_agent  # noqa: E402

# Real engine builders — imported up front so the spies can delegate to them.
from langchain.agents import create_agent as _real_create_agent  # noqa: E402
from langchain_core.tools import tool

from genai_tk.agents.langchain.config import (
    AgentProfileConfig,
    BackendConfig,
    CheckpointerConfig,
    MiddlewareConfig,
)
from genai_tk.agents.langchain.factory import (
    _load_skills_as_prompt,
    _resolve_skill_dirs,
    create_langchain_agent,
)
from genai_tk.agents.langchain.middleware.empty_response_retry import EmptyResponseRetryMiddleware
from genai_tk.agents.langchain.middleware.rich_middleware import RichToolCallMiddleware

# --------------------------------------------------------------------------- #
# Test tools (real LangChain @tool functions, used via extra_tools)
# --------------------------------------------------------------------------- #


@tool
def echo(text: str) -> str:
    """Echo back the text."""
    return text


@tool
def adder(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


# --------------------------------------------------------------------------- #
# Spies for the external engine builders (wrap the real call, capture kwargs)
# --------------------------------------------------------------------------- #


@contextmanager
def spy_react_engine():
    """Capture kwargs passed to langchain's ``create_agent`` while still building the real agent."""
    captured: dict[str, Any] = {}

    def _spy(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _real_create_agent(**kwargs)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("langchain.agents.create_agent", _spy)
        yield captured


@contextmanager
def spy_deep_engine():
    """Capture kwargs passed to deepagents' ``create_deep_agent`` while still building the real agent."""
    captured: dict[str, Any] = {}

    def _spy(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _real_create_deep_agent(**kwargs)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("deepagents.create_deep_agent", _spy)
        yield captured


def _react_profile(**overrides: Any) -> AgentProfileConfig:
    """Build a minimal react profile using the fake LLM id."""
    base: dict[str, Any] = {"name": "test-react", "type": "react", "llm": "parrot_local@fake"}
    base.update(overrides)
    return AgentProfileConfig(**base)


def _deep_profile(**overrides: Any) -> AgentProfileConfig:
    """Build a minimal deep profile using the fake LLM id (planning/fs off for speed)."""
    base: dict[str, Any] = {
        "name": "test-deep",
        "type": "deep",
        "llm": "parrot_local@fake",
        "enable_planning": False,
        "enable_file_system": False,
    }
    base.update(overrides)
    return AgentProfileConfig(**base)


# --------------------------------------------------------------------------- #
# React dispatch
# --------------------------------------------------------------------------- #


async def test_react_agent_minimal_no_checkpointer(fake_llm_id: str) -> None:
    profile = _react_profile(llm=fake_llm_id, checkpointer=CheckpointerConfig(type="none"))

    agent = await create_langchain_agent(profile)

    assert type(agent).__name__ == "CompiledStateGraph"
    assert agent.checkpointer is None
    assert hasattr(agent, "ainvoke")
    nodes = set(agent.get_graph().nodes.keys())
    # With no tools, langchain's create_agent omits the 'tools' node; 'model' is always present.
    assert "model" in nodes


async def test_react_agent_force_memory_checkpointer(fake_llm_id: str) -> None:
    profile = _react_profile(llm=fake_llm_id)

    agent = await create_langchain_agent(profile, force_memory_checkpointer=True)

    assert agent.checkpointer is not None
    assert type(agent.checkpointer).__name__ in {"MemorySaver", "InMemorySaver"}


async def test_react_agent_class_checkpointer(fake_llm_id: str) -> None:
    profile = _react_profile(
        llm=fake_llm_id,
        checkpointer=CheckpointerConfig(type="class", class_path="langgraph.checkpoint.memory.MemorySaver"),
    )

    agent = await create_langchain_agent(profile)

    assert agent.checkpointer is not None
    assert type(agent.checkpointer).__name__ in {"MemorySaver", "InMemorySaver"}


async def test_react_agent_invoke_returns_messages(fake_llm_id: str) -> None:
    profile = _react_profile(llm=fake_llm_id)

    agent = await create_langchain_agent(profile, force_memory_checkpointer=True)

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "hello parrot"}]},
        config={"configurable": {"thread_id": "factory-1"}},
    )

    assert "messages" in result
    assert len(result["messages"]) >= 1


# --------------------------------------------------------------------------- #
# Tools wiring
# --------------------------------------------------------------------------- #


async def test_react_agent_extra_tools_combined(fake_llm_id: str) -> None:
    profile = _react_profile(llm=fake_llm_id)

    with spy_react_engine() as captured:
        await create_langchain_agent(profile, extra_tools=[echo, adder])

    tool_names = {t.name for t in captured["tools"]}
    assert tool_names == {"echo", "adder"}


async def test_react_agent_no_tools_by_default(fake_llm_id: str) -> None:
    profile = _react_profile(llm=fake_llm_id)

    with spy_react_engine() as captured:
        await create_langchain_agent(profile)

    assert captured["tools"] == []


# --------------------------------------------------------------------------- #
# Middleware wiring (EmptyResponseRetryMiddleware prepend + details flag)
# --------------------------------------------------------------------------- #


def _middleware_types(middlewares: list[Any]) -> list[str]:
    return [type(m).__name__ for m in middlewares]


async def test_empty_response_retry_prepended_when_absent(fake_llm_id: str) -> None:
    profile = _react_profile(llm=fake_llm_id)

    with spy_react_engine() as captured:
        await create_langchain_agent(profile)

    types = _middleware_types(captured["middleware"])
    assert "EmptyResponseRetryMiddleware" in types
    empties = [m for m in captured["middleware"] if isinstance(m, EmptyResponseRetryMiddleware)]
    assert len(empties) == 1
    assert empties[0]._max_retries == 1
    # Prepended at index 0 (outermost) so it wraps before any other middleware.
    assert isinstance(captured["middleware"][0], EmptyResponseRetryMiddleware)


async def test_empty_response_retry_not_duplicated_when_present(fake_llm_id: str) -> None:
    profile = _react_profile(
        llm=fake_llm_id,
        middlewares=[
            MiddlewareConfig.model_validate(
                {
                    "class": "genai_tk.agents.langchain.middleware.empty_response_retry.EmptyResponseRetryMiddleware",
                    "max_retries": 3,
                }
            ),
        ],
    )

    with spy_react_engine() as captured:
        await create_langchain_agent(profile)

    empties = [m for m in captured["middleware"] if isinstance(m, EmptyResponseRetryMiddleware)]
    assert len(empties) == 1
    assert empties[0]._max_retries == 3


async def test_details_flag_propagates_to_rich_middleware(fake_llm_id: str) -> None:
    profile = _react_profile(
        llm=fake_llm_id,
        middlewares=[
            MiddlewareConfig.model_validate(
                {"class": "genai_tk.agents.langchain.middleware.rich_middleware.RichToolCallMiddleware"}
            ),
        ],
    )

    with spy_react_engine() as captured:
        await create_langchain_agent(profile, details=True)

    richs = [m for m in captured["middleware"] if isinstance(m, RichToolCallMiddleware)]
    assert len(richs) == 1
    assert richs[0]._details is True
    # EmptyResponseRetryMiddleware still prepended ahead of the rich middleware.
    assert isinstance(captured["middleware"][0], EmptyResponseRetryMiddleware)


async def test_rich_middleware_details_off_by_default(fake_llm_id: str) -> None:
    profile = _react_profile(
        llm=fake_llm_id,
        middlewares=[
            MiddlewareConfig.model_validate(
                {"class": "genai_tk.agents.langchain.middleware.rich_middleware.RichToolCallMiddleware"}
            ),
        ],
    )

    with spy_react_engine() as captured:
        await create_langchain_agent(profile)

    richs = [m for m in captured["middleware"] if isinstance(m, RichToolCallMiddleware)]
    assert richs and richs[0]._details is False


# --------------------------------------------------------------------------- #
# System prompt passthrough
# --------------------------------------------------------------------------- #


async def test_react_system_prompt_passed_through(fake_llm_id: str) -> None:
    profile = _react_profile(llm=fake_llm_id, system_prompt="You are a unit-test agent.")

    with spy_react_engine() as captured:
        await create_langchain_agent(profile)

    assert captured["system_prompt"] == "You are a unit-test agent."


async def test_react_pre_prompt_used_when_no_system_prompt(fake_llm_id: str) -> None:
    profile = _react_profile(llm=fake_llm_id, pre_prompt="PREFIX:")

    with spy_react_engine() as captured:
        await create_langchain_agent(profile)

    assert captured["system_prompt"] == "PREFIX:"


async def test_react_no_system_prompt_when_both_unset(fake_llm_id: str) -> None:
    profile = _react_profile(llm=fake_llm_id)

    with spy_react_engine() as captured:
        await create_langchain_agent(profile)

    assert "system_prompt" not in captured


# --------------------------------------------------------------------------- #
# LLM override precedence
# --------------------------------------------------------------------------- #


async def test_llm_override_takes_precedence_over_profile_llm(fake_llm_id: str) -> None:
    # "foo@fake" is an unknown LLM id that raises inside get_llm if used.
    profile = _react_profile(llm="foo@fake")

    # Override with the valid fake id — the agent must build without error,
    # proving the override wins over the broken profile.llm.
    agent = await create_langchain_agent(profile, llm_override=fake_llm_id)

    assert type(agent).__name__ == "CompiledStateGraph"


# --------------------------------------------------------------------------- #
# MCP servers
# --------------------------------------------------------------------------- #


async def test_mcp_servers_present_but_dict_empty_loads_no_tools(fake_llm_id: str) -> None:
    """Servers are requested but resolve to an empty dict → no MCP client, no tools.

    The config accessor is stubbed (returns {}) so the factory's "servers present,
    dict empty" branch runs without launching subprocesses; the factory's own
    branching still executes for real.
    """
    profile = _react_profile(llm=fake_llm_id)

    mp = pytest.MonkeyPatch()
    mp.setattr("genai_tk.agents.langchain.factory.get_mcp_servers_dict", lambda _names: {})
    try:
        with spy_react_engine() as captured:
            await create_langchain_agent(profile, extra_mcp_servers=["some-srv"])
    finally:
        mp.undo()

    # No tools loaded from MCP; agent still built.
    assert captured["tools"] == []


async def test_mcp_tools_loaded_and_combined(fake_llm_id: str) -> None:
    """MCP SDK (subprocess boundary) is mocked; factory wiring still runs for real."""
    profile = _react_profile(llm=fake_llm_id, mcp_servers=["fake-srv"])

    @tool
    def mcp_search(query: str) -> str:
        """Search via MCP."""
        return f"mcp:{query}"

    class _FakeMcpClient:
        async def get_tools(self) -> list[Any]:
            return [mcp_search]

    mp = pytest.MonkeyPatch()
    mp.setattr("genai_tk.agents.langchain.factory.get_mcp_servers_dict", lambda _names: {"fake-srv": {"command": "x"}})
    mp.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", lambda _cfg: _FakeMcpClient())
    try:
        with spy_react_engine() as captured:
            await create_langchain_agent(profile, extra_tools=[echo])
    finally:
        mp.undo()

    tool_names = {t.name for t in captured["tools"]}
    assert {"mcp_search", "echo"} == tool_names


# --------------------------------------------------------------------------- #
# Backend warning for non-deep agents
# --------------------------------------------------------------------------- #


async def test_non_deep_profile_with_backend_is_warned_and_ignored(fake_llm_id: str) -> None:
    from loguru import logger

    profile = _react_profile(
        llm=fake_llm_id,
        backend=BackendConfig(type="filesystem", root_dir="."),
    )

    warnings: list[str] = []
    sink_id = logger.add(warnings.append, level="WARNING", format="{message}")
    try:
        with spy_react_engine() as captured:
            agent = await create_langchain_agent(profile)
    finally:
        logger.remove(sink_id)

    # Backend is ignored for react; agent still built as a react agent.
    assert type(agent).__name__ == "CompiledStateGraph"
    assert type(captured["model"]).__name__ == "ParrotFakeChatModel"
    # A warning was emitted about the backend being ignored.
    assert any("backends are only used by deep agents" in w for w in warnings)


# --------------------------------------------------------------------------- #
# Deep dispatch
# --------------------------------------------------------------------------- #


async def test_deep_agent_no_backend(fake_llm_id: str) -> None:
    profile = _deep_profile(llm=fake_llm_id)

    with spy_deep_engine() as captured:
        agent = await create_langchain_agent(profile, force_memory_checkpointer=True)

    assert type(agent).__name__ == "CompiledStateGraph"
    assert agent._backend is None
    assert captured["backend"] is None
    assert captured["skills"] is None


async def test_deep_agent_with_skills_creates_filesystem_backend(fake_llm_id: str, tmp_path: Path) -> None:
    skill_dir = tmp_path / "myagent"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: myagent\ndescription: demo\n---\n# My skill\nDo things.")

    profile = _deep_profile(llm=fake_llm_id, skill_directories=[str(tmp_path)])

    with spy_deep_engine() as captured:
        agent = await create_langchain_agent(profile, force_memory_checkpointer=True)

    from deepagents.backends.filesystem import FilesystemBackend

    assert isinstance(agent._backend, FilesystemBackend)
    assert isinstance(captured["backend"], FilesystemBackend)
    # Skills are passed through as backend-relative paths.
    assert captured["skills"] is not None
    assert len(captured["skills"]) == 1


async def test_deep_agent_checkpointer_wired(fake_llm_id: str) -> None:
    profile = _deep_profile(llm=fake_llm_id)

    with spy_deep_engine() as captured:
        await create_langchain_agent(profile, force_memory_checkpointer=True)

    assert captured["checkpointer"] is not None
    assert type(captured["checkpointer"]).__name__ in {"MemorySaver", "InMemorySaver"}


# --------------------------------------------------------------------------- #
# Custom dispatch — documents a known production bug (wrong import path)
# --------------------------------------------------------------------------- #


async def test_custom_agent_raises_module_not_found(fake_llm_id: str) -> None:
    """The custom agent type is currently broken: the factory imports
    ``genai_tk.extra.graphs.custom_react_agent`` but the module actually lives at
    ``genai_tk.extra.langgraphs.custom_react_agent``.  This test documents the
    current (broken) behaviour so the line is covered; when the import path is
    fixed it should be updated to assert a successful build.
    """
    profile = AgentProfileConfig(name="test-custom", type="custom", llm=fake_llm_id)

    with pytest.raises(ModuleNotFoundError, match="genai_tk.extra.graphs"):
        await create_langchain_agent(profile, force_memory_checkpointer=True)


# --------------------------------------------------------------------------- #
# Unknown agent type → ValueError
# --------------------------------------------------------------------------- #


async def test_unknown_agent_type_raises_value_error(fake_llm_id: str) -> None:
    # model_copy(update=...) skips validation, letting an invalid type reach dispatch.
    profile = _react_profile(llm=fake_llm_id).model_copy(update={"type": "bogus"})

    with pytest.raises(ValueError, match="Unknown agent type"):
        await create_langchain_agent(profile)


# --------------------------------------------------------------------------- #
# Helpers: _resolve_skill_dirs
# --------------------------------------------------------------------------- #


def test_resolve_skill_dirs_empty() -> None:
    assert _resolve_skill_dirs([]) == []


def test_resolve_skill_dirs_filters_missing_and_expands(tmp_path: Path) -> None:
    # source-level dir: subdirs each contain SKILL.md → returned as-is
    src = tmp_path / "skills"
    src.mkdir()
    (src / "alpha").mkdir()
    (src / "alpha" / "SKILL.md").write_text("alpha")
    (src / "beta").mkdir()
    (src / "beta" / "SKILL.md").write_text("beta")

    result = _resolve_skill_dirs([str(src), "/does/not/exist"])
    assert str(src) in result
    assert "/does/not/exist" not in result


def test_resolve_skill_dirs_expands_grouping_dir(tmp_path: Path) -> None:
    # grouping dir: immediate children lack SKILL.md, grandchildren have them
    group = tmp_path / "group"
    group.mkdir()
    (group / "collection").mkdir()
    (group / "collection" / "skillA").mkdir()
    (group / "collection" / "skillA" / "SKILL.md").write_text("a")

    result = _resolve_skill_dirs([str(group)])
    assert str(group / "collection") in result


# --------------------------------------------------------------------------- #
# Helpers: _load_skills_as_prompt
# --------------------------------------------------------------------------- #


def test_load_skills_as_prompt_strips_front_matter(tmp_path: Path) -> None:
    skill = tmp_path / "agent"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: agent\ndescription: demo skill\n---\n# Agent\nDoes things well.")

    prompt = _load_skills_as_prompt([str(tmp_path)])

    assert prompt is not None
    assert "Does things well." in prompt
    assert "description: demo skill" not in prompt


def test_load_skills_as_prompt_returns_none_when_no_skills(tmp_path: Path) -> None:
    assert _load_skills_as_prompt([str(tmp_path)]) is None
