"""
Custom ReAct Agent based on Functional API

taken from https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch-functional/

"""

from typing import Any

from langchain_core.language_models.base import LanguageModelOutput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.func import entrypoint, task
from langgraph.graph.message import add_messages
from langgraph.pregel import Pregel


def create_custom_react_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver,
) -> Pregel:
    """Create a custom ReAct agent from scratch using Functional API.

    Args:
        model: Language model to use
        tools: List of tools the agent can use
        checkpointer: Checkpoint storage for agent state
    Example:
    ```python
    tools = [get_weather]
    checkpointer = MemorySaver()
    llm = get_llm()
    custom_agent = create_custom_react_agent(llm, tools, checkpointer)
    ```
    """
    tools_by_name = {tool.name: tool for tool in tools}

    @task
    def call_model(messages: list[BaseMessage]) -> LanguageModelOutput:
        """Call model with a sequence of messages."""
        response = model.bind_tools(tools).invoke(messages)
        return response

    @task
    def call_tool(tool_call: dict[str, Any]) -> ToolMessage:
        """Call a tool with the provided arguments."""
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        return ToolMessage(content=observation, tool_call_id=tool_call["id"])

    @entrypoint(checkpointer=checkpointer)
    def agent(messages: str | list[BaseMessage], previous: list[BaseMessage] | None) -> Any:
        # Let type inference handle messages_list - add_messages has flexible input/output types
        if isinstance(messages, str):
            messages_list = [messages]  # type: ignore[list-item]
        else:
            messages_list = messages
        if previous is not None:
            messages_list = add_messages(previous, messages_list)  # type: ignore[arg-type]

        llm_response = call_model(messages_list).result()  # type: ignore[arg-type]
        while True:
            if not llm_response.tool_calls:  # type: ignore[union-attr]
                break

            # Execute tools
            tool_result_futures = [call_tool(tool_call) for tool_call in llm_response.tool_calls]  # type: ignore[union-attr]
            tool_results = [fut.result() for fut in tool_result_futures]

            # Append to message list
            messages_list = add_messages(messages_list, [llm_response, *tool_results])  # type: ignore[arg-type]

            # Call model again
            llm_response = call_model(messages_list).result()  # type: ignore[arg-type]

        # Generate final response
        messages_list = add_messages(messages_list, llm_response)  # type: ignore[arg-type]
        return entrypoint.final(value=llm_response, save=messages_list)

    return agent
