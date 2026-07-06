"""Unit tests for ``utils/pydantic_utils/common.py``.

Exercises the validation, docstring/field-description extraction,
optional-unwrapping and type-humanization helpers.  The humanize/unwrap helpers
are driven with real type objects (not string annotations), so PEP 563 is fine.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import pytest
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from genai_tk.utils.pydantic_utils.common import (
    get_class_description,
    get_field_description,
    humanize_type,
    unwrap_optional,
    validate_pydantic_model,
)

# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #


class _Color(Enum):
    RED = 1
    GREEN = 2


class _Person(BaseModel):
    name: str = Field(description="full name")
    age: int = 0


class _Documented:
    """First summary line.

    More detail here.
    """


class _BlankFirstLine:
    """

    First non-blank line.
    """


class _NoDocstring:
    pass


class _WhitespaceDoc:
    """ """


class _ModelWithFields(BaseModel):
    described: str = Field(description="the described field")
    undescribed: str


# --------------------------------------------------------------------------- #
# validate_pydantic_model
# --------------------------------------------------------------------------- #


def test_validate_pydantic_model_returns_class_for_basemodel() -> None:
    assert validate_pydantic_model(_Person) is _Person


def test_validate_pydantic_model_instance_raises_not_a_class() -> None:
    with pytest.raises(ValueError, match="is not a class"):
        validate_pydantic_model(_Person(name="x"))


def test_validate_pydantic_model_non_basemodel_class_raises() -> None:
    with pytest.raises(ValueError, match="is not a Pydantic BaseModel"):
        validate_pydantic_model(int)


def test_validate_pydantic_model_uses_class_name_in_error() -> None:
    with pytest.raises(ValueError, match="'MyInt' is not a Pydantic BaseModel"):
        validate_pydantic_model(int, class_name="MyInt")


# --------------------------------------------------------------------------- #
# get_class_description
# --------------------------------------------------------------------------- #


def test_get_class_description_first_line() -> None:
    assert get_class_description(_Documented) == "First summary line."


def test_get_class_description_skips_blank_first_line() -> None:
    assert get_class_description(_BlankFirstLine) == "First non-blank line."


def test_get_class_description_no_docstring_returns_empty() -> None:
    assert get_class_description(_NoDocstring) == ""


def test_get_class_description_whitespace_only_docstring_returns_empty() -> None:
    assert get_class_description(_WhitespaceDoc) == ""


# --------------------------------------------------------------------------- #
# get_field_description
# --------------------------------------------------------------------------- #


def test_get_field_description_present() -> None:
    assert get_field_description(_ModelWithFields.model_fields["described"]) == "the described field"


def test_get_field_description_absent_returns_empty() -> None:
    assert get_field_description(_ModelWithFields.model_fields["undescribed"]) == ""


def test_get_field_description_bare_field_info_returns_empty() -> None:
    assert get_field_description(FieldInfo()) == ""


# --------------------------------------------------------------------------- #
# unwrap_optional
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("annotation", "expected_inner", "expected_optional"),
    [
        (str | None, str, True),
        (Optional[str], str, True),
        (int, int, False),
        (str | int, str, True),
        (type(None), type(None), False),
        (list[str], list[str], False),
        (int | None | str, int, True),
    ],
)
def test_unwrap_optional(annotation: Any, expected_inner: Any, expected_optional: bool) -> None:
    inner, is_opt = unwrap_optional(annotation)
    assert inner == expected_inner
    assert is_opt == expected_optional


# --------------------------------------------------------------------------- #
# humanize_type
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("annotation", "is_optional", "expected"),
    [
        (str, False, "string"),
        (int, False, "int"),
        (float, False, "float"),
        (bool, False, "boolean"),
        (list[str], False, "string[]"),
        (set[int], False, "int[]"),
        (tuple[str, ...], False, "string[]"),
        (dict[str, int], False, "object"),
        (str | None, False, "string?"),
        (type(None), False, "null"),
        (_Color, False, "enum(_Color)"),
        (_Person, False, "_Person"),
        (int, True, "int?"),
        (list[int | None], False, "int[]"),
        (list[str], True, "string[]?"),
    ],
)
def test_humanize_type(annotation: Any, is_optional: bool, expected: str) -> None:
    assert humanize_type(annotation, is_optional=is_optional) == expected


def test_humanize_type_none_singleton_falls_back_to_str() -> None:
    # Passing the None *singleton* (not the NoneType class) hits the fallback
    # ``str(base_type)`` branch; only ``type(None)`` yields "null" (see parametrize).
    assert humanize_type(None) == "None"
