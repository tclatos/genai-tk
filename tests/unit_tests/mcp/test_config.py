"""Unit tests for genai_tk.mcp.config – Pydantic models and YAML loader."""

import tempfile
import textwrap
from pathlib import Path

import pytest

from genai_tk.mcp.config import (
    MCPAgentConfig,
    MCPServerDefinition,
    MCPToolConfig,
    get_mcp_server_definition,
    load_mcp_server_definitions,
)

# ---------------------------------------------------------------------------
# MCPToolConfig
# ---------------------------------------------------------------------------


class TestMCPToolConfig:
    def test_factory_required(self) -> None:
        tc = MCPToolConfig(factory="module:func")
        assert tc.factory == "module:func"

    def test_flat_kwargs_stored(self) -> None:
        tc = MCPToolConfig(factory="module:func", verbose=True, k=5)  # type: ignore[call-arg]
        assert tc.factory_kwargs() == {"verbose": True, "k": 5}

    def test_nested_config_kwarg(self) -> None:
        tc = MCPToolConfig(factory="module:func", config={"database_uri": "sqlite:///db"})  # type: ignore[call-arg]
        kw = tc.factory_kwargs()
        assert kw == {"config": {"database_uri": "sqlite:///db"}}

    def test_factory_not_in_kwargs(self) -> None:
        tc = MCPToolConfig(factory="x:y", verbose=False)  # type: ignore[call-arg]
        assert "factory" not in tc.factory_kwargs()


# ---------------------------------------------------------------------------
# MCPAgentConfig
# ---------------------------------------------------------------------------


class TestMCPAgentConfig:
    def test_defaults(self) -> None:
        cfg = MCPAgentConfig()
        assert cfg.enabled is True
        assert cfg.name == "run_agent"
        assert cfg.llm is None
        assert cfg.profile is None

    def test_custom_fields(self) -> None:
        cfg = MCPAgentConfig(
            enabled=False,
            name="run_research",
            description="Research agent",
            llm="gpt_41mini@openai",
            profile="Research",
        )
        assert cfg.enabled is False
        assert cfg.name == "run_research"
        assert cfg.profile == "Research"


# ---------------------------------------------------------------------------
# MCPServerDefinition
# ---------------------------------------------------------------------------


class TestMCPServerDefinition:
    def test_minimal(self) -> None:
        defn = MCPServerDefinition(name="test")
        assert defn.name == "test"
        assert defn.tools == []
        assert defn.agent is None

    def test_full(self) -> None:
        defn = MCPServerDefinition(
            name="search",
            description="Web search",
            tools=[MCPToolConfig(factory="mod:func")],
            agent=MCPAgentConfig(name="run_search"),
        )
        assert len(defn.tools) == 1
        assert defn.agent is not None


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


_SAMPLE_YAML = textwrap.dedent(
    """
    mcp_expose_servers:
      - name: "alpha"
        description: "Alpha server"
        tools:
          - factory: "mod_a:create_tool"
            verbose: true
        agent:
          enabled: true
          name: run_alpha
          description: "Alpha agent"

      - name: "beta"
        description: "Beta server – tools only"
        tools:
          - factory: "mod_b:make_tool"
    """
)


class TestLoadMCPServerDefinitions:
    def _write_yaml(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False, encoding="utf-8")
        tmp.write(content)
        tmp.flush()
        return Path(tmp.name)

    def test_load_two_servers(self) -> None:
        path = self._write_yaml(_SAMPLE_YAML)
        servers = load_mcp_server_definitions(path)
        assert len(servers) == 2
        assert servers[0].name == "alpha"
        assert servers[1].name == "beta"

    def test_tools_parsed(self) -> None:
        path = self._write_yaml(_SAMPLE_YAML)
        servers = load_mcp_server_definitions(path)
        alpha = servers[0]
        assert len(alpha.tools) == 1
        assert alpha.tools[0].factory == "mod_a:create_tool"
        assert alpha.tools[0].factory_kwargs() == {"verbose": True}

    def test_agent_parsed(self) -> None:
        path = self._write_yaml(_SAMPLE_YAML)
        alpha = load_mcp_server_definitions(path)[0]
        assert alpha.agent is not None
        assert alpha.agent.name == "run_alpha"
        assert alpha.agent.enabled is True

    def test_no_agent_when_absent(self) -> None:
        path = self._write_yaml(_SAMPLE_YAML)
        beta = load_mcp_server_definitions(path)[1]
        assert beta.agent is None

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_mcp_server_definitions(Path("/nonexistent/servers.yaml"))

    def test_get_by_name(self) -> None:
        path = self._write_yaml(_SAMPLE_YAML)
        defn = get_mcp_server_definition("beta", path)
        assert defn.name == "beta"

    def test_get_unknown_name_raises(self) -> None:
        path = self._write_yaml(_SAMPLE_YAML)
        with pytest.raises(ValueError, match="not found"):
            get_mcp_server_definition("unknown", path)
