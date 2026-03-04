"""Shared Pydantic models for tool specifications across all agent frameworks.

Provides reusable tool specification models for factory, class, and function-based
tools used in LangChain, Smolagents, and Deerflow agent configurations.
"""

from typing import Any, Dict, Union

from pydantic import BaseModel, ConfigDict, Field

from genai_tk.utils.config_mngr import QualifiedClassName, QualifiedFunctionName


class ClassToolSpec(BaseModel):
    """Tool specification for a class-based tool."""

    tool_class: QualifiedClassName = Field(..., alias="class", description="Qualified class name")
    extra_params: Dict[str, Any] = Field(
        default_factory=dict, description="Additional parameters for class instantiation"
    )

    model_config = ConfigDict(populate_by_name=True)


class FunctionToolSpec(BaseModel):
    """Tool specification for a function-based tool."""

    function: QualifiedFunctionName = Field(..., description="Qualified function name")


class FactoryToolSpec(BaseModel):
    """Tool specification for a factory-based tool."""

    factory: QualifiedFunctionName = Field(..., description="Qualified factory function name")
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters for factory")


# Union type for all tool specifications
ToolSpec = Union[ClassToolSpec, FunctionToolSpec, FactoryToolSpec]


def tool_spec_from_dict(tool_config: Dict[str, Any]) -> Union[ClassToolSpec, FunctionToolSpec, FactoryToolSpec, None]:
    """Convert a dictionary tool configuration to a ToolSpec Pydantic model.

    Args:
        tool_config: Tool configuration dictionary

    Returns:
        ToolSpec instance or None if invalid
    """
    config_copy = tool_config.copy()
    if "class" in config_copy:
        class_ref = config_copy.pop("class")
        extra_params = config_copy
        return ClassToolSpec(tool_class=class_ref, extra_params=extra_params)
    elif "function" in config_copy:
        func_ref = config_copy.pop("function")
        return FunctionToolSpec(function=func_ref)
    elif "factory" in config_copy:
        factory_ref = config_copy.pop("factory")
        extra_params = config_copy
        return FactoryToolSpec(factory=factory_ref, extra_params=extra_params)
    return None
