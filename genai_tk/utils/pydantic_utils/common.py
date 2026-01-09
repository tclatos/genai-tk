"""Common Pydantic utilities for validation and type checking."""

from typing import Any, Type, TypeVar

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
