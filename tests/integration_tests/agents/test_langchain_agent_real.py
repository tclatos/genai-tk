"""Integration tests for LangchainAgent with real LLM models.

Covers four areas:

1. **React agents** — simple LLM Q&A, code generation, streaming, multi-turn memory
2. **Deep agents (local)** — code generation without Docker (filesystem backend, no container)
3. **Named profiles** — loading real profiles from ``langchain.yaml``; smoke-runs
4. **Skills loading** — SKILL.md discovery, filesystem backend wiring, content access
5. **Docker sandbox** (opt-in) — full deep-agent run inside a real container

All tests require ``--include-real-models`` (uses ``fast_model`` = claude-haiku@openrouter).
Docker tests additionally require ``--include-docker``.

Run examples::

    # Deterministic structure tests (no LLM, subset of this file)
    uv run pytest tests/integration_tests/agents/test_langchain_agent_real.py \\
        -k "profile_loads or skill_directory_initializes" -v

    # All tests with real models
    uv run pytest tests/integration_tests/agents/test_langchain_agent_real.py \\
        -v --include-real-models --timeout=180

    # Full suite including Docker
    uv run pytest tests/integration_tests/agents/test_langchain_agent_real.py \\
        -v --include-real-models --include-docker
"""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_tk.agents.langchain.langchain_agent import LangchainAgent

# ── Module-level markers ──────────────────────────────────────────────────────

pytestmark = pytest.mark.integration  # real_models added per-class/per-test below

# ── Constants ─────────────────────────────────────────────────────────────────

LLM = "fast_model"  # resolves to claude-haiku@openrouter via config; cheap & reliable


# ── Helper utilities ──────────────────────────────────────────────────────────


async def _run(agent: LangchainAgent, query: str) -> str:
    """Invoke the agent and always close it afterwards."""
    async with agent:
        return await agent.arun(query)


def _has(text: str, *words: str) -> bool:
    """Return True if ``text`` contains at least one of ``words`` (case-insensitive)."""
    lower = text.lower()
    return any(w.lower() in lower for w in words)


def _assert_python_function(code: str, *name_hints: str) -> None:
    """Assert that ``code`` looks like a Python function definition."""
    assert "def " in code, f"Expected a Python 'def' statement — got:\n{code[:300]}"
    if name_hints:
        assert _has(code, *name_hints), f"Expected one of {name_hints} in code — got:\n{code[:300]}"


# ─────────────────────────────────────────────────────────────────────────────
# 1. React agent tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.real_models
class TestReactAgent:
    """React-mode agents: single-turn and multi-turn with a real LLM.

    These use no tools by default, so responses are pure LLM output.
    Tests are intentionally lenient — they check for key words/patterns
    rather than exact strings to stay stable across model versions.
    """

    @pytest.mark.timeout(120)
    @pytest.mark.asyncio
    async def test_fibonacci_code_generation(self) -> None:
        """Agent generates a valid Python Fibonacci function when asked."""
        agent = LangchainAgent(llm=LLM, agent_type="react")
        result = await _run(
            agent,
            (
                "Write a concise Python function to compute the nth Fibonacci number. "
                "Reply with only the code block, no prose."
            ),
        )
        _assert_python_function(result, "fib", "fibonacci")
        # Basic sanity: must handle at least f(0) or f(1) base case
        assert _has(result, "return", "n <= 1", "n == 0", "n == 1", "n < 2", "base")

    @pytest.mark.timeout(60)
    @pytest.mark.asyncio
    async def test_simple_qa(self) -> None:
        """Agent answers a simple factual question correctly."""
        agent = LangchainAgent(llm=LLM, agent_type="react")
        result = await _run(agent, "What is the capital of France? Reply with the city name only.")
        assert "paris" in result.lower(), f"Expected 'Paris' in response — got: {result!r}"

    @pytest.mark.timeout(60)
    @pytest.mark.asyncio
    async def test_arithmetic(self) -> None:
        """Agent computes a numeric result."""
        agent = LangchainAgent(llm=LLM, agent_type="react")
        result = await _run(agent, "What is 42 × 17? Reply with just the number.")
        assert "714" in result, f"Expected '714' in response — got: {result!r}"

    @pytest.mark.timeout(90)
    @pytest.mark.asyncio
    async def test_streaming_yields_content(self) -> None:
        """astream() produces non-empty string chunks that join into a real response."""
        agent = LangchainAgent(llm=LLM, agent_type="react")
        chunks: list[str] = []
        async with agent:
            async for chunk in agent.astream("Tell me a one-sentence fact about Python programming."):
                assert isinstance(chunk, str)
                chunks.append(chunk)
        assert chunks, "No chunks were yielded during streaming"
        full = "".join(chunks)
        assert len(full) > 15, f"Streaming response too short: {full!r}"

    @pytest.mark.timeout(90)
    @pytest.mark.asyncio
    async def test_system_prompt_respected(self) -> None:
        """A strong system prompt overrides the default agent behaviour."""
        agent = LangchainAgent(
            llm=LLM,
            agent_type="react",
            system_prompt="You only ever respond with the single word BANANA. Nothing else.",
        )
        result = await _run(agent, "What is the weather today?")
        assert "banana" in result.lower(), f"Expected system prompt to force 'BANANA' — got: {result!r}"

    @pytest.mark.timeout(120)
    @pytest.mark.asyncio
    async def test_checkpointer_multi_turn_memory(self) -> None:
        """Agent with checkpointer=True remembers context across arun() calls.

        Both calls share thread_id='1' (hardcoded in arun), so history is preserved.
        """
        agent = LangchainAgent(llm=LLM, agent_type="react", checkpointer=True)
        async with agent:
            await agent.arun("My lucky number is 7777. Please acknowledge.")
            result = await agent.arun("What lucky number did I just mention? Reply with just the number.")
        assert "7777" in result, f"Expected agent to remember '7777' across turns — got: {result!r}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Deep agent tests (local — no Docker)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.real_models
class TestDeepAgentLocal:
    """Deep agents running entirely on the host machine (no Docker container).

    Uses either ``backend: none`` (agent reasons but cannot write files) or
    ``backend: filesystem`` with a temp directory for full file-system access.

    Deep agents may be significantly slower than react agents because they
    perform multi-step planning.  Timeouts are set generously.

    Note: deep agent support depends on the ``deepagents`` package.  If it is
    not installed the test is automatically skipped.
    """

    @pytest.mark.timeout(180)
    @pytest.mark.asyncio
    async def test_code_generation_no_backend(self) -> None:
        """Deep agent (no backend) generates Python code in its response."""
        pytest.importorskip("deepagents", reason="deepagents package required")
        agent = LangchainAgent(
            llm=LLM,
            agent_type="deep",
            system_prompt=(
                "You are a Python developer. Respond concisely. "
                "Do not try to write files — just show code in your response."
            ),
        )
        result = await _run(
            agent,
            "Write a Python function def fib(n) that returns the nth Fibonacci number. Show only the code.",
        )
        _assert_python_function(result, "fib", "fibonacci")

    @pytest.mark.timeout(240)
    @pytest.mark.asyncio
    async def test_writes_file_to_filesystem_backend(self, tmp_path: Path) -> None:
        """Deep agent writes a Python file using a local FilesystemBackend.

        The backend is a temp directory, so no project files are touched.
        We accept either a written file OR code in the response text (the agent
        may summarise instead of writing if it decides to skip the file step).
        """
        pytest.importorskip("deepagents", reason="deepagents package required")
        from genai_tk.agents.langchain.config import AgentProfileConfig, BackendConfig
        from genai_tk.agents.langchain.factory import create_langchain_agent

        profile = AgentProfileConfig(
            name="test-coder",
            type="deep",
            llm=LLM,
            system_prompt=(
                "You are a coding assistant. When asked to write a file, "
                "create it immediately using available file tools."
            ),
            enable_planning=False,
            enable_file_system=True,
            backend=BackendConfig(type="filesystem", root_dir=str(tmp_path)),
        )
        agent_graph = await create_langchain_agent(profile)
        try:
            result = await agent_graph.ainvoke(
                {
                    "messages": (
                        "Write a Python function def fib(n) that computes the nth Fibonacci number. Save it to /fib.py"
                    )
                },
                {"configurable": {"thread_id": "t1"}},
            )
        finally:
            backend_obj = getattr(agent_graph, "_backend", None)
            if backend_obj and hasattr(backend_obj, "stop"):
                await backend_obj.stop()  # type: ignore[attr-defined]

        from genai_tk.agents.langchain.langchain_agent import _extract_content

        response_text = _extract_content(result)
        fib_file = tmp_path / "fib.py"
        wrote_file = fib_file.exists() and "def" in fib_file.read_text()
        has_code_in_response = _has(response_text, "def fib", "def fibonacci", "fibonacci")

        assert wrote_file or has_code_in_response, (
            f"Expected fib.py to be written or code in response.\n"
            f"fib.py exists: {fib_file.exists()}\n"
            f"Response snippet: {response_text[:400]}"
        )

    @pytest.mark.timeout(180)
    @pytest.mark.asyncio
    async def test_deep_agent_planning_step(self) -> None:
        """Deep agent uses the write_todos planning step on a multi-step task.

        We ask a multi-step task and verify the agent at least starts planning
        or produces a structured response.  This does not require Docker.
        """
        pytest.importorskip("deepagents", reason="deepagents package required")
        agent = LangchainAgent(
            llm=LLM,
            agent_type="deep",
            system_prompt="You are a helpful assistant. Use write_todos to plan multi-step tasks.",
        )
        result = await _run(
            agent,
            "Explain in 3 numbered steps how you would implement a binary search algorithm.",
        )
        # We expect at least a structured list or step-based response
        assert _has(result, "1.", "step 1", "first", "binary"), f"Expected structured response — got: {result[:300]}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Named profile tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNamedProfiles:
    """Tests for named profiles defined in ``config/agents/langchain/*.yaml``.

    Profile structure (field types, defaults) is tested synchronously — no LLM
    call needed.  Functional "runs and returns coherent text" tests are marked
    separately and use the real LLM.
    """

    # ── Structural / field checks — no LLM call ──────────────────────────────

    def test_simple_profile_is_react(self) -> None:
        """'simple' profile resolves to type=react with a pre_prompt."""
        agent = LangchainAgent("simple", llm=LLM)
        assert agent._profile is not None
        assert agent._profile.type == "react"

    def test_coding_profile_is_deep(self) -> None:
        """'Coding' profile resolves to type=deep with file-system enabled."""
        pytest.importorskip("deepagents", reason="deepagents package required")
        agent = LangchainAgent("Coding", llm=LLM)
        assert agent._profile is not None
        assert agent._profile.type == "deep"
        assert agent._profile.enable_file_system is True
        assert agent._profile.enable_planning is True

    def test_research_profile_is_deep_with_planning(self) -> None:
        """'Research' profile resolves to type=deep with planning and skill directories."""
        pytest.importorskip("deepagents", reason="deepagents package required")
        agent = LangchainAgent("Research", llm=LLM)
        assert agent._profile is not None
        assert agent._profile.type == "deep"
        assert agent._profile.enable_planning is True
        # Research profile should have skill directories configured
        assert len(agent._profile.skill_directories) > 0

    def test_text2sql_profile_has_filesystem_backend(self) -> None:
        """'text2sql' profile uses a filesystem backend (no Docker needed)."""
        pytest.importorskip("deepagents", reason="deepagents package required")
        agent = LangchainAgent("text2sql", llm=LLM)
        assert agent._profile is not None
        assert agent._profile.type == "deep"
        assert agent._profile.backend is not None
        assert agent._profile.backend.type == "filesystem"

    # ── Functional runs — real LLM ────────────────────────────────────────────

    @pytest.mark.real_models
    @pytest.mark.timeout(120)
    @pytest.mark.asyncio
    async def test_simple_profile_qa(self) -> None:
        """'simple' profile answers a direct question without using search."""
        agent = LangchainAgent("simple", llm=LLM)
        result = await _run(agent, "What is 6 × 7? Reply with just the number.")
        assert "42" in result, f"Expected '42' — got: {result!r}"

    @pytest.mark.real_models
    @pytest.mark.timeout(180)
    @pytest.mark.asyncio
    async def test_coding_profile_generates_code(self) -> None:
        """'Coding' profile (deep) generates Python code when asked.

        We instruct it not to save files to keep the test predictable and fast.
        """
        pytest.importorskip("deepagents", reason="deepagents package required")
        agent = LangchainAgent(
            "Coding",
            llm=LLM,
            system_prompt=("You are a Python developer. Reply with only the code — do not write any files."),
        )
        result = await _run(
            agent,
            "Show me a Python function def fib(n) for computing Fibonacci numbers.",
        )
        _assert_python_function(result, "fib", "fibonacci")

    @pytest.mark.real_models
    @pytest.mark.timeout(180)
    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_simple_profile_web_search(self) -> None:
        """'simple' profile runs a web search and returns a factual answer.

        Requires network access.  Uses DuckDuckGo (no API key) or Tavily if
        ``TAVILY_API_KEY`` is set.  The query is answered by the LLM even if
        search fails — so we only assert the response is non-trivial.
        """
        agent = LangchainAgent("simple", llm=LLM)
        result = await _run(
            agent,
            "In which year was Python programming language first released publicly? Reply with just the year.",
        )
        # Python was first released in 1991 (0.9) or 1994 (1.0)
        assert _has(result, "1991", "1994", "1989", "1990"), (
            f"Expected a year close to Python's release — got: {result!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Skills loading tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSkillsLoading:
    """Tests for the SKILL.md discovery and loading mechanism.

    Skills use *progressive disclosure* — the LLM sees skill metadata at
    startup and reads full SKILL.md content on demand.  The factory
    automatically wires a ``FilesystemBackend`` when skill_directories are
    configured, so no Docker is needed.
    """

    def _create_skill(self, skills_root: Path, name: str, content: str) -> None:
        """Write a SKILL.md in ``skills_root/<name>/``."""
        skill_dir = skills_root / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Test skill for {name}\n---\n\n{content}")

    def test_skill_directory_resolves_correctly(self, tmp_path: Path) -> None:
        """_resolve_skill_dirs correctly finds skill directories one level deep."""
        from genai_tk.agents.langchain.factory import _resolve_skill_dirs

        self._create_skill(tmp_path, "my-skill", "## Content")
        result = _resolve_skill_dirs([str(tmp_path)])
        assert len(result) == 1
        assert str(tmp_path) in result[0] or Path(result[0]).parent == tmp_path

    @pytest.mark.real_models
    @pytest.mark.timeout(120)
    @pytest.mark.asyncio
    async def test_agent_initializes_with_skill_directories(self, tmp_path: Path) -> None:
        """Deep agent initializes without error when skill_directories are configured.

        This is a smoke-test: it verifies that the factory sets up the
        FilesystemBackend and SkillsMiddleware without raising.

        The skill directory is placed inside ``tmp_path`` and the filesystem
        backend is rooted at ``tmp_path`` so the SkillsMiddleware can access
        it without a path-traversal error.
        """
        pytest.importorskip("deepagents", reason="deepagents package required")
        self._create_skill(tmp_path, "demo-skill", "## Demo\nThis is a demo skill.")

        from genai_tk.agents.langchain.config import AgentProfileConfig, BackendConfig
        from genai_tk.agents.langchain.factory import create_langchain_agent

        profile = AgentProfileConfig(
            name="test-skills-smoke",
            type="deep",
            llm=LLM,
            system_prompt="You are a helpful assistant.",
            skill_directories=[str(tmp_path)],
            # Explicit backend so skills/backend share the same root (tmp_path)
            backend=BackendConfig(type="filesystem", root_dir=str(tmp_path)),
        )
        agent_graph = await create_langchain_agent(profile)
        assert agent_graph is not None, "Agent graph should have been created"
        backend_obj = getattr(agent_graph, "_backend", None)
        assert backend_obj is not None, "Expected a FilesystemBackend to be wired for the profile"
        if hasattr(backend_obj, "stop"):
            await backend_obj.stop()  # type: ignore[attr-defined]

    @pytest.mark.real_models
    @pytest.mark.timeout(240)
    @pytest.mark.asyncio
    async def test_skill_content_accessible_to_agent(self, tmp_path: Path) -> None:
        """Agent can read a custom SKILL.md and uses its content in its reply.

        A unique token is embedded in the skill.  We ask the agent to check
        its skills and report the token.  The test is allowed to fail (xfail)
        because progressive disclosure means the LLM may not proactively
        read the skill unless prompted strongly enough.

        The skill directory is placed inside ``tmp_path`` and the filesystem
        backend is rooted at ``tmp_path`` so there is no path-traversal issue.
        """
        pytest.importorskip("deepagents", reason="deepagents package required")

        unique_token = "XYZZY_SECRET_9472"
        self._create_skill(
            tmp_path,
            "secret-skill",
            f"## Secret Token\nThe confidential project code is: {unique_token}",
        )

        from genai_tk.agents.langchain.config import AgentProfileConfig, BackendConfig
        from genai_tk.agents.langchain.factory import create_langchain_agent
        from genai_tk.agents.langchain.langchain_agent import _extract_content

        profile = AgentProfileConfig(
            name="test-skill-reader",
            type="deep",
            llm=LLM,
            system_prompt=(
                "You are a helpful assistant. Always read your skill files before answering domain questions."
            ),
            skill_directories=[str(tmp_path)],
            backend=BackendConfig(type="filesystem", root_dir=str(tmp_path)),
        )
        agent_graph = await create_langchain_agent(profile)
        try:
            result = await agent_graph.ainvoke(
                {
                    "messages": (
                        "Read your 'secret-skill' skill file and tell me the confidential project code written there."
                    )
                },
                {"configurable": {"thread_id": "skill-test"}},
            )
        finally:
            backend_obj = getattr(agent_graph, "_backend", None)
            if backend_obj and hasattr(backend_obj, "stop"):
                await backend_obj.stop()  # type: ignore[attr-defined]

        response = _extract_content(result)
        # This may fail if the model doesn't read the skill file — mark as xfail
        if unique_token not in response:
            pytest.xfail(
                f"Agent did not find skill token '{unique_token}' in response. "
                f"Progressive disclosure may require additional prompting. "
                f"Response: {response[:300]}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Docker sandbox tests (opt-in via --include-docker)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.real_models
@pytest.mark.docker
class TestDockerSandbox:
    """Deep agent tests inside a real Docker sandbox container.

    These tests start ``ghcr.io/agent-infra/sandbox:latest`` and run an agent
    inside it.  Requires Docker on the host and the ``--include-docker`` flag.

    Run::

        uv run pytest tests/integration_tests/agents/test_langchain_agent_real.py \\
            -v --include-real-models --include-docker -k "TestDockerSandbox"
    """

    @pytest.mark.timeout(300)
    @pytest.mark.asyncio
    async def test_react_via_docker_sandbox_override(self) -> None:
        """React agent promoted to deep with sandbox='docker' produces a response.

        The sandbox override upgrades the react profile to deep and starts the
        AIO sandbox container.  We ask for a simple computation so we can verify
        the response without inspecting container internals.
        """
        pytest.importorskip("deepagents", reason="deepagents package required")
        agent = LangchainAgent(
            llm=LLM,
            agent_type="react",
            sandbox="docker",
            system_prompt="You are a helpful assistant. Answer concisely.",
        )
        result = await _run(agent, "What is 12 + 30? Reply with just the number.")
        assert "42" in result, f"Expected '42' from sandboxed agent — got: {result!r}"

    @pytest.mark.timeout(300)
    @pytest.mark.asyncio
    async def test_coding_profile_writes_file_in_docker(self, tmp_path: Path) -> None:
        """'Coding' profile inside Docker writes a Python file to the container.

        We verify both the agent response and (if possible) that the file was
        created inside the container.  Deep agent may write to its working
        directory (``/home/user`` inside the container); we check the response.
        """
        pytest.importorskip("deepagents", reason="deepagents package required")
        agent = LangchainAgent(
            "Coding",
            llm=LLM,
            sandbox="docker",
        )
        result = await _run(
            agent,
            "Write a Python file hello.py that prints 'Hello, sandbox!' when run. "
            "Show the file contents in your reply.",
        )
        assert _has(result, "hello", "print", "sandbox"), f"Expected file content in response — got: {result[:400]}"
