"""Shared Pydantic models for tool specifications across all agent frameworks.

Provides reusable tool specification models for factory, class, and function-based
tools used in LangChain and DeerFlow agent configurations.

YAML format:
```yaml
tools:
  - my.pkg.make_tools                            # Bare string qualified name
  - my.pkg:MyToolClass                           # Class qualified name
  - my.pkg.tool_factory:                         # Single-key dict with params / nested tools
      param1: value
      tools:
        - my.pkg.sub_tool
```
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from genai_tk.config_mgmt.config_mngr import QualifiedClassName, QualifiedFunctionName


class UnifiedToolSpec(BaseModel):
    """Unified specification for any tool (class, function, or factory)."""

    target: str
    extra_params: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _validate_tool_spec(cls, v: Any) -> Any:
        if isinstance(v, str):
            return {"target": v, "extra_params": {}}
        if isinstance(v, dict):
            v = dict(v)
            for key in ("tool", "factory", "class", "function", "target"):
                if key in v:
                    target_val = v.pop(key)
                    existing = v.pop("extra_params", {})
                    return {"target": str(target_val), "extra_params": {**v, **existing}}
            if len(v) == 1:
                key, val = next(iter(v.items()))
                params = val if isinstance(val, dict) else ({"tools": val} if isinstance(val, list) else {})
                return {"target": str(key), "extra_params": params}
            if "target" in v:
                return v
        return v


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


# Union type for all tool specifications — UnifiedToolSpec handles strings & dicts directly.
ToolSpec = UnifiedToolSpec | ClassToolSpec | FunctionToolSpec | FactoryToolSpec
