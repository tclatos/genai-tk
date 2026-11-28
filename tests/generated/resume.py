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
        "default_prefix": "resume",
        "default_range": "string",
        "description": "Schema for candidate resume extraction.",
        "id": "https://example.org/tests/resume",
        "imports": ["linkml:types"],
        "name": "resume",
        "prefixes": {
            "linkml": {"prefix_prefix": "linkml", "prefix_reference": "https://w3id.org/linkml/"},
            "resume": {"prefix_prefix": "resume", "prefix_reference": "https://example.org/tests/resume/"},
        },
        "source_file": "/home/tcl/prj/genai-tk/tests/linkml_src/resume.yaml",
    }
)


class Resume(ConfiguredBaseModel):
    """
    Candidate resume containing contact and background details.
    """

    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta(
        {"from_schema": "https://example.org/tests/resume", "tree_root": True}
    )

    name: str = Field(
        default=...,
        description="""name of the candidate""",
        json_schema_extra={
            "linkml_meta": {
                "annotations": {"prompt": {"tag": "prompt", "value": "Full name of the candidate"}},
                "domain_of": ["Resume"],
            }
        },
    )
    email: str = Field(
        default=...,
        description="""email address of the candidate""",
        json_schema_extra={"linkml_meta": {"domain_of": ["Resume"]}},
    )
    experience: Optional[list[str]] = Field(
        default=None,
        description="""List of professional work experiences""",
        json_schema_extra={"linkml_meta": {"domain_of": ["Resume"]}},
    )
    skills: Optional[list[str]] = Field(
        default=None,
        description="""List of skills and competencies""",
        json_schema_extra={"linkml_meta": {"domain_of": ["Resume"]}},
    )


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
Resume.model_rebuild()
