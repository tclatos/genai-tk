from __future__ import annotations

import re
import sys
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any, ClassVar, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
)


metamodel_version = "1.12.0"
version = "None"


class ConfiguredBaseModel(BaseModel):
    model_config = ConfigDict(
        serialize_by_alias=True,
        validate_by_name=True,
        validate_assignment=True,
        validate_default=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        use_enum_values=True,
        strict=False,
    )


class LinkMLMeta(RootModel):
    root: dict[str, Any] = {}
    model_config = ConfigDict(frozen=True)

    def __getattr__(self, key: str):
        return getattr(self.root, key)

    def __getitem__(self, key: str):
        return self.root[key]

    def __setitem__(self, key: str, value):
        self.root[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.root


linkml_meta = LinkMLMeta(
    {
        "default_prefix": "enums_fix",
        "default_range": "string",
        "description": "Focused schema to verify enum and permissible-value description preservation.",
        "id": "https://example.org/tests/enums_fixture",
        "imports": ["linkml:types"],
        "name": "enums_fixture",
        "prefixes": {
            "enums_fix": {"prefix_prefix": "enums_fix", "prefix_reference": "https://example.org/tests/enums_fixture/"},
            "linkml": {"prefix_prefix": "linkml", "prefix_reference": "https://w3id.org/linkml/"},
        },
        "source_file": "/home/tcl/prj/genai-tk/tests/linkml_src/enums_fixture.yaml",
    }
)


class ProcessingStatus(str, Enum):
    """
    Lifecycle processing status of an item.
    """

    PENDING = "PENDING"
    """
    Awaiting initial review or ingestion.
    """
    IN_PROGRESS = "IN_PROGRESS"
    """
    Active processing is ongoing.
    """
    COMPLETED = "COMPLETED"
    """
    Successfully analyzed and validated.
    """
    FAILED = "FAILED"
    """
    Extraction or validation encountered an unrecoverable error.
    """


class EnumItem(ConfiguredBaseModel):
    """
    An item with an enumerated status.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {"from_schema": "https://example.org/tests/enums_fixture", "tree_root": True}
    )

    item_id: str = Field(
        default=...,
        description="""Unique item identifier""",
        json_schema_extra={"linkml_meta": {"domain_of": ["EnumItem"]}},
    )
    status: ProcessingStatus = Field(
        default=...,
        description="""Current processing status""",
        json_schema_extra={"linkml_meta": {"domain_of": ["EnumItem"]}},
    )


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
EnumItem.model_rebuild()
