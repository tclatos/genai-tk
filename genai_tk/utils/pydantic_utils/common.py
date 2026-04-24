"""Common Pydantic utilities for validation, type introspection, and annotation formatting."""

from enum import Enum
from typing import Any, Type, TypeVar, get_args, get_origin

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def validate_pydantic_model(model_cls: Any, class_name: str | None = None) -> Type[BaseModel]:
    """Validate that a class is a Pydantic BaseModel.

    Args:
        model_cls: Class to validate
        class_name: Optional name for better error messages

    Returns:
        The validated Pydantic model class

    Raises:
        ValueError: If the class is not a valid Pydantic BaseModel
    """
    name = class_name or getattr(model_cls, "__name__", str(model_cls))

    if not isinstance(model_cls, type):
        raise ValueError(f"'{name}' is not a class")

    if not issubclass(model_cls, BaseModel):
        raise ValueError(f"'{name}' is not a Pydantic BaseModel")

    return model_cls


def get_class_description(cls: type) -> str:
    """Return the first non-empty docstring line of *cls*, or empty string.

    Args:
        cls: Python class to inspect.
    """
    if not cls.__doc__:
        return ""
    for line in cls.__doc__.strip().split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def get_field_description(field_info: Any) -> str:
    """Return the Pydantic v2 ``Field(description=...)`` value, or empty string.

    Args:
        field_info: Pydantic ``FieldInfo`` object from ``model_fields``.
    """
    if hasattr(field_info, "description") and field_info.description:
        return field_info.description
    return ""


def unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Unwrap ``Optional[T]`` / ``T | None`` to ``(T, True)``; pass-through otherwise.

    Args:
        annotation: A Python type annotation.

    Returns:
        ``(inner_type, True)`` if optional, else ``(annotation, False)``.
    """
    import types
    from typing import Union

    origin = get_origin(annotation)
    if origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0], True
        return non_none[0] if non_none else annotation, True
    return annotation, False


def humanize_type(annotation: Any, is_optional: bool = False) -> str:
    """Convert a Python type annotation to a compact, human-readable string.

    Suitable for LLM-facing prompts and documentation.

    Examples:
        ``str`` → ``"string"``
        ``list[str]`` → ``"string[]"``
        ``int | None`` → ``"int?"``

    Args:
        annotation: Python type annotation.
        is_optional: If True, appends ``?`` suffix regardless of annotation.
    """
    if annotation is type(None):
        return "null"

    base_type, is_opt = unwrap_optional(annotation)
    is_optional = is_optional or is_opt

    origin = get_origin(base_type)
    args = get_args(base_type)

    if origin in (list, set, tuple):
        inner = humanize_type(args[0]) if args else "any"
        result = f"{inner.rstrip('?')}[]"
    elif origin is dict:
        result = "object"
    elif base_type is str:
        result = "string"
    elif base_type is int:
        result = "int"
    elif base_type is float:
        result = "float"
    elif base_type is bool:
        result = "boolean"
    elif isinstance(base_type, type) and issubclass(base_type, Enum):
        result = f"enum({base_type.__name__})"
    elif hasattr(base_type, "__name__"):
        result = base_type.__name__
    else:
        result = str(base_type)

    return f"{result}?" if is_optional else result
