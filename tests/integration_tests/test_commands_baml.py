"""Integration tests for BAML CLI commands.

This module tests the CLI commands for BAML-based structured extraction,
including running BAML functions with various LLMs and configurations.
"""

import asyncio

import pytest
from pydantic import BaseModel

from genai_tk.extra.structured.baml_util import baml_invoke


class Resume(BaseModel):
    """Resume data model matching BAML schema."""

    name: str
    email: str
    experience: list[str]
    skills: list[str]


class TestBamlInvokeFunction:
    """Test suite for baml_invoke function."""

    @pytest.mark.network
    def test_baml_invoke_with_fake_resume_default_llm(self):
        """Test baml_invoke with FakeResume function using default LLM."""
        input_text = "John Smith; SW engineer"

        result = asyncio.run(baml_invoke("FakeResume", {"blabla": input_text}, config_name="default"))

        # Verify result is a Resume instance
        assert isinstance(result, BaseModel)
        assert hasattr(result, "name")
        assert hasattr(result, "email")
        assert hasattr(result, "experience")
        assert hasattr(result, "skills")

        # Verify basic content
        assert result.name
        assert "@" in result.email
        assert len(result.experience) > 0
        assert len(result.skills) > 0

    @pytest.mark.network
    def test_baml_invoke_with_fake_resume_mistral_8b(self):
        """Test baml_invoke with FakeResume function using mistral_8b_openrouter LLM."""
        input_text = "Jane Doe; Data Scientist"

        result = asyncio.run(
            baml_invoke("FakeResume", {"blabla": input_text}, config_name="default", llm="mistral_8b_openrouter")
        )

        # Verify result is a Resume instance
        assert isinstance(result, BaseModel)
        assert hasattr(result, "name")
        assert hasattr(result, "email")
        assert hasattr(result, "experience")
        assert hasattr(result, "skills")

        # Verify basic content
        assert result.name
        assert "@" in result.email
        assert len(result.experience) > 0
        assert len(result.skills) > 0

        # Verify content relates to input
        name_lower = result.name.lower()
        assert any(
            skill.lower() in ["python", "machine learning", "data", "statistics", "sql"] for skill in result.skills
        )

    @pytest.mark.network
    def test_baml_invoke_with_extract_resume_default_llm(self):
        """Test baml_invoke with ExtractResume function using default LLM."""
        input_text = """
        Alice Johnson
        alice.johnson@example.com

        Experience:
        - Senior Developer at TechCorp (2020-2023)
        - Junior Developer at StartupXYZ (2018-2020)

        Skills:
        - Python
        - JavaScript
        - Docker
        """

        result = asyncio.run(baml_invoke("ExtractResume", {"resume": input_text}, config_name="default"))

        # Verify result is a Resume instance
        assert isinstance(result, BaseModel)
        assert hasattr(result, "name")
        assert hasattr(result, "email")
        assert hasattr(result, "experience")
        assert hasattr(result, "skills")

        # Verify extracted content
        assert "alice" in result.name.lower() or "johnson" in result.name.lower()
        assert "alice.johnson@example.com" in result.email.lower()
        assert len(result.experience) > 0
        assert len(result.skills) > 0

    def test_baml_invoke_with_invalid_function_name(self):
        """Test baml_invoke with invalid function name."""
        with pytest.raises(Exception) as exc_info:
            asyncio.run(baml_invoke("NonExistentFunction", {"param": "value"}, config_name="default"))

        assert "NonExistentFunction" in str(exc_info.value) or "Failed to load" in str(exc_info.value)

    @pytest.mark.network
    def test_baml_invoke_with_different_llms(self):
        """Test baml_invoke with multiple LLMs to ensure consistency."""
        input_text = "Bob Brown; Backend Developer"

        # Test with default LLM
        result1 = asyncio.run(baml_invoke("FakeResume", {"blabla": input_text}, config_name="default"))

        # Test with mistral_8b_openrouter
        result2 = asyncio.run(
            baml_invoke("FakeResume", {"blabla": input_text}, config_name="default", llm="mistral_8b_openrouter")
        )

        # Both should return valid Resume objects
        assert isinstance(result1, BaseModel)
        assert isinstance(result2, BaseModel)

        # Both should have populated fields
        assert result1.name and result2.name
        assert result1.email and result2.email
        assert len(result1.experience) > 0 and len(result2.experience) > 0
        assert len(result1.skills) > 0 and len(result2.skills) > 0
