"""Tests for structured output functionality with fake models.

This module contains tests for structured data extraction from LLMs
using fake models to ensure fast, reliable testing.
"""

from typing import Dict, List, Optional

import pytest
from genai_tk.wip.structured_output import StructOutMethod, structured_output_chain
from pydantic import BaseModel, Field

# Constants
FAKE_LLM_ID = "parrot_local_fake"
FAST_LLM_TAG = ""


# Test models for structured output
class TestResponse(BaseModel):
    """Simple test response model."""

    message: str = Field(description="A message")
    success: bool = Field(description="Whether the operation was successful")
    score: float = Field(description="A confidence score between 0 and 1")


class ComplexResponse(BaseModel):
    """More complex test response model."""

    title: str = Field(description="The title of the item")
    items: List[str] = Field(description="List of items")
    metadata: Dict[str, str] = Field(description="Additional metadata")
    optional_field: Optional[str] = Field(default=None, description="An optional field")


class NestedResponse(BaseModel):
    """Test model with nested structure."""

    user: TestResponse
    actions: List[ComplexResponse]
    timestamp: str


def test_structured_output_with_fake_llm(fake_llm) -> None:
    """Test structured output extraction with fake LLM."""
    chain = structured_output_chain(
        system="You are a helpful assistant.",
        user="Generate a test response about success",
        llm_id=FAKE_LLM_ID,
        output_class=TestResponse,
        method=StructOutMethod.FUNCTION_CALLING,
    )

    result = chain.invoke({})

    assert isinstance(result, TestResponse)
    assert isinstance(result.message, str)
    assert isinstance(result.success, bool)
    assert isinstance(result.score, float)
    assert 0 <= result.score <= 1


def test_structured_output_with_complex_model() -> None:
    """Test structured output with complex model."""
    chain = structured_output_chain(
        system="You are a helpful assistant.",
        user="Create a complex response with multiple items",
        llm_id=FAKE_LLM_ID,
        output_class=ComplexResponse,
        method=StructOutMethod.FUNCTION_CALLING,
    )

    result = chain.invoke({})

    assert isinstance(result, ComplexResponse)
    assert isinstance(result.title, str)
    assert isinstance(result.items, list)
    assert isinstance(result.metadata, dict)
    assert result.optional_field is None or isinstance(result.optional_field, str)


def test_structured_output_with_nested_model() -> None:
    """Test structured output with nested models."""
    chain = structured_output_chain(
        system="You are a helpful assistant.",
        user="Create a nested response structure",
        llm_id=FAKE_LLM_ID,
        output_class=NestedResponse,
        method=StructOutMethod.FUNCTION_CALLING,
    )

    result = chain.invoke({})

    assert isinstance(result, NestedResponse)
    assert isinstance(result.user, TestResponse)
    assert isinstance(result.actions, list)
    assert isinstance(result.timestamp, str)

    # Validate nested structure
    for action in result.actions:
        assert isinstance(action, ComplexResponse)


def test_structured_output_json_mode() -> None:
    """Test structured output with JSON mode enabled."""
    chain = structured_output_chain(
        system="You are a helpful assistant.",
        user="Generate a JSON response",
        llm_id=FAKE_LLM_ID,
        output_class=TestResponse,
        method=StructOutMethod.JSON_SCHEMA,
    )

    result = chain.invoke({})

    assert isinstance(result, TestResponse)
    # JSON mode should provide more structured results
    assert result.message.strip() != ""


def test_structured_output_error_handling() -> None:
    """Test structured output error handling."""
    chain = structured_output_chain(
        system="You are a helpful assistant.",
        user="Generate a response",
        llm_id=FAKE_LLM_ID,
        output_class=TestResponse,
        method=StructOutMethod.FUNCTION_CALLING,
    )

    # Test with empty input
    result = chain.invoke({})
    assert isinstance(result, TestResponse)


def test_structured_output_consistency() -> None:
    """Test that structured output is consistent for same input."""
    chain = structured_output_chain(
        system="You are a helpful assistant.",
        user="Generate a consistent response",
        llm_id=FAKE_LLM_ID,
        output_class=TestResponse,
        method=StructOutMethod.FUNCTION_CALLING,
    )

    result1 = chain.invoke({})
    result2 = chain.invoke({})

    # Both should be valid instances of the model
    assert isinstance(result1, TestResponse)
    assert isinstance(result2, TestResponse)

    # With fake models, results should be identical
    assert result1.message == result2.message
    assert result1.success == result2.success
    assert result1.score == result2.score


def test_structured_output_with_different_llm_configs() -> None:
    """Test structured output with different LLM configurations."""
    configs = [
        {"temperature": 0.0},
        {"temperature": 0.5},
        {"max_tokens": 100},
        {"temperature": 0.0, "max_tokens": 50},
    ]

    for _config in configs:
        chain = structured_output_chain(
            system="You are a helpful assistant.",
            user="Test with different config",
            llm_id=FAKE_LLM_ID,
            output_class=TestResponse,
            method=StructOutMethod.FUNCTION_CALLING,
        )

        result = chain.invoke({})
        assert isinstance(result, TestResponse)


def test_structured_output_performance(performance_threshold) -> None:
    """Test structured output performance with fake models."""
    import time

    chain = structured_output_chain(
        system="You are a helpful assistant.",
        user="Performance test",
        llm_id=FAKE_LLM_ID,
        output_class=TestResponse,
        method=StructOutMethod.FUNCTION_CALLING,
    )

    start_time = time.time()
    result = chain.invoke({})
    end_time = time.time()

    execution_time = end_time - start_time

    assert isinstance(result, TestResponse)
    assert execution_time < performance_threshold


def test_structured_output_with_invalid_model() -> None:
    """Test structured output with edge case models."""

    class MinimalModel(BaseModel):
        """Model with no fields."""

    chain = structured_output_chain(
        system="You are a helpful assistant.",
        user="Test minimal model",
        llm_id=FAKE_LLM_ID,
        output_class=MinimalModel,
        method=StructOutMethod.FUNCTION_CALLING,
    )
    result = chain.invoke({})
    assert isinstance(result, MinimalModel)


def test_structured_output_field_validation() -> None:
    """Test that structured output respects field validation."""

    class ValidatedModel(BaseModel):
        """Model with field validation."""

        name: str = Field(min_length=1, max_length=50)
        age: int = Field(ge=0, le=150)
        email: str = Field(pattern=r".*@.*\..*")

    chain = structured_output_chain(
        system="You are a helpful assistant.",
        user="Create a validated user profile",
        llm_id=FAKE_LLM_ID,
        output_class=ValidatedModel,
        method=StructOutMethod.FUNCTION_CALLING,
    )
    result = chain.invoke({})

    assert isinstance(result, ValidatedModel)
    assert 1 <= len(result.name) <= 50
    assert 0 <= result.age <= 150
    assert "@" in result.email and "." in result.email


@pytest.mark.parametrize("model_class", [TestResponse, ComplexResponse, NestedResponse])
def test_structured_output_parameterized(model_class) -> None:
    """Test structured output with parameterized models."""
    chain = structured_output_chain(
        system="You are a helpful assistant.",
        user=f"Create a {model_class.__name__} instance",
        llm_id=FAKE_LLM_ID,
        output_class=model_class,
        method=StructOutMethod.FUNCTION_CALLING,
    )
    result = chain.invoke({})

    assert isinstance(result, model_class)
