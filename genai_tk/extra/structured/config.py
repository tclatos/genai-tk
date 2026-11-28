"""Configuration model for LinkML and structured extraction."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LinkMLConfig(BaseModel):
    """Typed configuration for a single ``linkml`` (or legacy ``structured``) YAML entry.

    Supports LinkML configuration:
        schema: path/to/schema.yaml
        root_class: ModelName
        generated_module: dotted.module.path
        prompt: path/to/prompt.md
        llm: default

    And legacy BAML configuration:
        baml_client: package.name
        default_function: FunctionName
        llm: default
    """

    baml_client: str | None = None
    schema_path: str | None = Field(default=None, alias="schema")
    root_class: str | None = None
    generated_module: str | None = None
    prompt: str | None = None
    llm: str = "default"
    default_function: str | None = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")


# Backward-compatibility alias
StructuredConfig = LinkMLConfig
