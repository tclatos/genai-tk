"""
Deep Agents Integration Module

This module provides the core integration layer for deepagents library,
making it easy to create and manage deep agents within the genai-blueprint framework.
It combines the power of deepagents with the existing LangChain/LangGraph infrastructure.

Key Features:
- Factory functions for creating deep agents
- Integration with existing LLM factory
- Tool management and registration
- File system and state management
- Configurable agents with custom subagents
"""

from functools import wraps
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

from deepagents import (
    async_create_configurable_agent,
    async_create_deep_agent,
    create_configurable_agent,
    create_deep_agent,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from genai_tk.core.llm_factory import get_llm


class DeepAgentConfig(BaseModel):
    """Configuration for Deep Agent instances"""

    name: str = Field(default="Deep Agent", description="Name of the agent")
    instructions: str = Field(default="", description="System instructions for the agent")
    model: Optional[str] = Field(default=None, description="Model to use (if None, uses default)")
    builtin_tools: Optional[List[str]] = Field(default=None, description="List of builtin tools to enable")
    on_duplicate_tools: str = Field(
        default="warn", description="How to handle duplicate tools: warn, error, replace, ignore"
    )
    subagents: Optional[List[Dict[str, Any]]] = Field(default=None, description="Custom subagents configuration")
    enable_file_system: bool = Field(default=True, description="Enable virtual file system tools")
    enable_planning: bool = Field(default=True, description="Enable planning tool")


class SubAgentConfig(BaseModel):
    """Configuration for a sub-agent"""

    name: str = Field(description="Name of the subagent")
    description: str = Field(description="Description of the subagent")
    prompt: str = Field(description="System prompt for the subagent")
    tools: Optional[List[str]] = Field(default=None, description="List of tools available to subagent")
    model_settings: Optional[Dict[str, Any]] = Field(default=None, description="Model settings for subagent")


class DeepAgentFactory:
    """Factory class for creating and managing deep agents"""

    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.default_model: Optional[BaseChatModel] = None

    def set_default_model(self, model: Optional[Union[str, BaseChatModel]] = None):
        """Set the default model for all agents"""
        if model is None:
            self.default_model = get_llm()
        elif isinstance(model, str):
            self.default_model = get_llm(llm_id=model)
        else:
            self.default_model = model

    def _convert_langchain_tool(self, tool: BaseTool) -> Callable:
        """Convert a LangChain tool to a regular Python function for deepagents"""

        @wraps(tool.func if hasattr(tool, "func") else tool._run)
        def wrapper(**kwargs):
            return tool.run(kwargs)

        # Preserve the docstring and name
        wrapper.__doc__ = tool.description
        wrapper.__name__ = tool.name

        return wrapper

    def _prepare_tools(self, tools: List[Union[BaseTool, Callable]]) -> List[Callable]:
        """Prepare tools for use with deepagents"""
        prepared_tools = []

        for tool in tools:
            if isinstance(tool, BaseTool):
                prepared_tools.append(self._convert_langchain_tool(tool))
            elif callable(tool):
                prepared_tools.append(tool)
            else:
                raise ValueError(f"Tool must be a BaseTool or callable, got {type(tool)}: {tool}")

        return prepared_tools

    def create_agent(
        self, config: DeepAgentConfig, tools: Optional[List[Union[BaseTool, Callable]]] = None, async_mode: bool = False
    ) -> Any:
        """
        Create a deep agent with the given configuration.

        Args:
            config: DeepAgentConfig object with agent settings
            tools: List of tools (can be LangChain tools or regular functions)
            async_mode: Whether to create an async agent

        Returns:
            A configured deep agent instance
        """

        # Prepare tools
        prepared_tools = self._prepare_tools(tools or [])

        llm = self.default_model or get_llm()

        # Prepare subagents if provided
        subagents = None
        if config.subagents:
            subagents = [
                {
                    "name": sa["name"],
                    "description": sa["description"],
                    "prompt": sa["prompt"],
                    "tools": sa.get("tools"),
                    "model_settings": sa.get("model_settings"),
                }
                for sa in config.subagents
            ]

        # Prepare builtin tools
        builtin_tools = config.builtin_tools
        if builtin_tools is None and config.enable_file_system:
            builtin_tools = ["write_file", "read_file", "ls", "edit_file"]
        if builtin_tools is None and config.enable_planning:
            if builtin_tools is None:
                builtin_tools = []
            builtin_tools.append("write_todos")

        # Create the agent
        create_func = async_create_deep_agent if async_mode else create_deep_agent

        # Note: on_duplicate_tools is not supported in deepagents API yet
        agent = create_func(
            tools=prepared_tools,
            instructions=config.instructions,
            model=llm,
            subagents=subagents,
            builtin_tools=builtin_tools,
        )

        # Store the agent
        self.agents[config.name] = agent
        return agent

    def get_agent(self, name: str) -> Optional[Any]:
        """Get an agent by name"""
        return self.agents.get(name)

    def list_agents(self) -> List[str]:
        """List all agent names"""
        return list(self.agents.keys())

    def create_research_agent(
        self,
        search_tool: Union[BaseTool, Callable],
        name: str = "Research Agent",
        additional_tools: Optional[List[Union[BaseTool, Callable]]] = None,
        async_mode: bool = False,
    ) -> Any:
        """
        Create a research agent with search capabilities.

        Args:
            search_tool: Tool for searching information
            name: Name of the agent
            additional_tools: Additional tools to include
            async_mode: Whether to create an async agent

        Returns:
            A configured research agent
        """
        tools = [search_tool] + (additional_tools or [])

        config = DeepAgentConfig(
            name=name,
            instructions=(
                "You are a research assistant. Your task is to help users find and analyze information. "
                "Use the search tool to find relevant information and provide comprehensive, well-structured answers. "
                "Always cite your sources and verify the accuracy of the information you provide."
            ),
            enable_file_system=True,
            enable_planning=True,
        )

        return self.create_agent(config, tools, async_mode)

    def create_coding_agent(
        self,
        name: str = "Coding Agent",
        language: str = "python",
        project_path: Optional[str] = None,
        async_mode: bool = False,
    ) -> Any:
        """
        Create a coding agent for software development.

        Args:
            name: Name of the agent
            language: Programming language specialization
            project_path: Path to the project directory
            async_mode: Whether to create an async agent

        Returns:
            A configured coding agent
        """
        config = DeepAgentConfig(
            name=name,
            instructions=(
                f"You are an expert {language} developer. Your task is to help users with coding tasks, "
                f"including writing, debugging, and refactoring {language} code. "
                f"Follow best practices, write clean and maintainable code, and provide clear explanations. "
                f"When analyzing code, focus on correctness, efficiency, and readability."
            ),
            enable_file_system=True,
            enable_planning=True,
        )

        tools = []
        if project_path:
            # Add project-specific tools if needed
            pass

        return self.create_agent(config, tools, async_mode)

    def create_data_analysis_agent(self, name: str = "Data Analysis Agent", async_mode: bool = False) -> Any:
        """
        Create a data analysis agent.

        Args:
            name: Name of the agent
            async_mode: Whether to create an async agent

        Returns:
            A configured data analysis agent
        """
        config = DeepAgentConfig(
            name=name,
            instructions=(
                "You are a data analysis expert. Your task is to help users analyze data, create visualizations, "
                "and provide insights from datasets. Use appropriate statistical methods and create clear, "
                "informative visualizations. Always explain your analysis process and the meaning of your results."
            ),
            enable_file_system=True,
            enable_planning=True,
        )

        return self.create_agent(config, [], async_mode)

    def create_configurable_deep_agent(
        self,
        agent_config: Dict[str, Any],
        tools: Optional[List[Union[BaseTool, Callable]]] = None,
        async_mode: bool = False,
        recursion_limit: int = 1000,
    ) -> Any:
        """
        Create a configurable deep agent that can be customized at runtime.

        Args:
            agent_config: Dictionary with agent configuration
            tools: List of tools
            async_mode: Whether to create an async agent
            recursion_limit: Maximum recursion depth

        Returns:
            A configurable deep agent
        """

        prepared_tools = self._prepare_tools(tools or [])

        create_func = async_create_configurable_agent if async_mode else create_configurable_agent

        agent = create_func(
            agent_config.get("instructions", ""),
            agent_config.get("subagents", []),
            prepared_tools,
            agent_config={"recursion_limit": recursion_limit},
        )

        return agent


# Global factory instance
deep_agent_factory = DeepAgentFactory()


# Convenience functions
def create_deep_agent_from_config(
    config: DeepAgentConfig, tools: Optional[List[Union[BaseTool, Callable]]] = None, async_mode: bool = False
) -> Any:
    """
    Convenience function to create a deep agent.

    Args:
        config: DeepAgentConfig object
        tools: List of tools
        async_mode: Whether to create async agent

    Returns:
        Configured deep agent
    """
    return deep_agent_factory.create_agent(config, tools, async_mode)


def create_research_deep_agent(
    search_tool: Union[BaseTool, Callable],
    name: str = "Research Agent",
    additional_tools: Optional[List[Union[BaseTool, Callable]]] = None,
    async_mode: bool = False,
) -> Any:
    """
    Convenience function to create a research agent.

    Args:
        search_tool: Tool for searching information
        name: Name of the agent
        additional_tools: Additional tools to include
        async_mode: Whether to create an async agent

    Returns:
        A configured research agent
    """
    return deep_agent_factory.create_research_agent(search_tool, name, additional_tools, async_mode)


def create_coding_deep_agent(
    name: str = "Coding Agent", language: str = "python", project_path: Optional[str] = None, async_mode: bool = False
) -> Any:
    """
    Convenience function to create a coding agent.

    Args:
        name: Name of the agent
        language: Programming language specialization
        project_path: Path to the project directory
        async_mode: Whether to create an async agent

    Returns:
        A configured coding agent
    """
    return deep_agent_factory.create_coding_agent(name, language, project_path, async_mode)


async def run_deep_agent(
    agent: Any, messages: List[Dict[str, str]], files: Optional[Dict[str, str]] = None, stream: bool = False
) -> Union[Dict[str, Any], AsyncIterator[Any]]:
    """
    Run a deep agent with messages.

    Args:
        agent: The deep agent to run
        messages: List of message dictionaries
        files: Optional file system state
        stream: Whether to stream responses

    Returns:
        Agent response (dict if not streaming, async iterator if streaming)
    """

    input_data = {"messages": messages}
    if files:
        input_data["files"] = files

    if stream:
        # Return the async generator directly
        return agent.astream(input_data)
    else:
        result = await agent.ainvoke(input_data)
        return result
