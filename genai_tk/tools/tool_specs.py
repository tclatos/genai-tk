"""Shared Pydantic models for tool specifications across all agent frameworks.

Provides reusable tool specification models for factory, class, and function-based
tools used in LangChain, Smolagents, and Deerflow agent configurations.

YAML format (flat dict, ``class``/``function``/``factory`` key acts as discriminator):

```yaml
tools:
  - class: my.pkg:MyTool        # ClassToolSpec – extra keys become extra_params
    timeout: 30
  - function: my.pkg:my_func    # FunctionToolSpec
  - factory: my.pkg:make_tools  # FactoryToolSpec – extra keys become extra_params
    param1: value
```
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from genai_tk.utils.config_mngr import QualifiedClassName, QualifiedFunctionName


class ClassToolSpec(BaseModel):
    """Tool specification for a class-based tool."""

    tool_class: QualifiedClassName = Field(..., alias="class")
    extra_params: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _collect_extra_params(cls, v: Any) -> Any:
        if not isinstance(v, dict) or "class" not in v:
            return v
        v = dict(v)
        class_ref = v.pop("class")
        existing = v.pop("extra_params", {})
        return {"class": class_ref, "extra_params": {**v, **existing}}


class FunctionToolSpec(BaseModel):
    """Tool specification for a function-based tool."""

    function: QualifiedFunctionName


class FactoryToolSpec(BaseModel):
    """Tool specification for a factory-based tool."""

    factory: QualifiedFunctionName
    extra_params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_extra_params(cls, v: Any) -> Any:
        if not isinstance(v, dict) or "factory" not in v:
            return v
        v = dict(v)
        factory_ref = v.pop("factory")
        existing = v.pop("extra_params", {})
        return {"factory": factory_ref, "extra_params": {**v, **existing}}


# Union type for all tool specifications — Pydantic parses flat YAML dicts directly.
ToolSpec = ClassToolSpec | FunctionToolSpec | FactoryToolSpec
