"""Unit tests for PydanticStructuredProcessor and structured_extract."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

try:
    from genai_tk.extra.structured.exceptions import (
        LinkMLExtractionError,
        ModelCapabilityError,
        StructuredExtractionError,
    )
    from genai_tk.extra.structured.linkml_contract import LinkMLContract
    from genai_tk.extra.structured.linkml_processor import (
        LinkMLProcessor,
        PydanticStructuredProcessor,
        extract_json_from_raw,
        linkml_extract,
        structured_extract,
        validate_model_capabilities,
    )
except ImportError:
    pytest.skip("linkml feature not installed", allow_module_level=True)

from tests.generated.resume import Resume

RESUME_SCHEMA = Path(__file__).parent.parent.parent / "linkml_src" / "resume.yaml"
PROMPT_FILE = Path(__file__).parent.parent.parent / "prompts" / "resume.md"


# ---------------------------------------------------------------------------
# Backward-compatibility alias tests
# ---------------------------------------------------------------------------


def test_backward_compat_aliases() -> None:
    assert PydanticStructuredProcessor is LinkMLProcessor
    assert structured_extract is linkml_extract
    assert StructuredExtractionError is LinkMLExtractionError


# ---------------------------------------------------------------------------
# extract_json_from_raw tests
# ---------------------------------------------------------------------------


def test_extract_json_from_raw_string() -> None:
    assert extract_json_from_raw('{"a": 1}') == '{"a": 1}'
    assert extract_json_from_raw("") == ""


def test_extract_json_from_raw_ai_message() -> None:
    msg = AIMessage(content='{"name": "Alice"}')
    assert extract_json_from_raw(msg) == '{"name": "Alice"}'


def test_extract_json_from_raw_tool_calls() -> None:
    msg = AIMessage(content="", tool_calls=[{"name": "Resume", "args": {"name": "Bob"}, "id": "call_1"}])
    assert extract_json_from_raw(msg) == '{"name": "Bob"}'


def test_extract_json_from_raw_multipart_content() -> None:
    msg = AIMessage(content=[{"type": "text", "text": '{"skills":'}, {"type": "text", "text": ' ["Rust"]}'}])
    assert extract_json_from_raw(msg) == '{"skills": ["Rust"]}'


# ---------------------------------------------------------------------------
# structured_extract tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structured_extract_fast_path() -> None:
    llm = MagicMock()
    runnable = MagicMock()
    runnable.ainvoke = AsyncMock(
        return_value={
            "parsed": {
                "name": "Jane Doe",
                "email": "jane@example.com",
                "experience": ["5 years backend dev"],
                "skills": ["Python", "FastAPI"],
            },
            "raw": AIMessage(content="ignored"),
        }
    )
    llm.with_structured_output.return_value = runnable

    res = await linkml_extract(llm, "Extract resume", Resume)
    assert isinstance(res, Resume)
    assert res.name == "Jane Doe"
    assert res.email == "jane@example.com"
    assert "Python" in res.skills


@pytest.mark.asyncio
async def test_structured_extract_recovery_path_via_json_repair() -> None:
    llm = MagicMock()
    runnable = MagicMock()
    # Parsed failed (None), but raw has truncated JSON that json_repair can salvage
    truncated_json = (
        '{"name": "John Doe", "email": "john@example.com", "experience": ["Founder"], "skills": ["Rust", "C++"'
    )
    runnable.ainvoke = AsyncMock(
        return_value={
            "parsed": None,
            "raw": AIMessage(content=truncated_json),
        }
    )
    llm.with_structured_output.return_value = runnable

    res = await linkml_extract(llm, "Extract resume", Resume)
    assert isinstance(res, Resume)
    assert res.name == "John Doe"
    assert res.email == "john@example.com"
    assert "Rust" in res.skills


@pytest.mark.asyncio
async def test_structured_extract_fallback_when_with_structured_output_not_implemented() -> None:
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError("Structured output not supported")
    json_response = (
        '{"name": "Sam Smith", "email": "sam@example.com", "experience": ["DevOps"], "skills": ["Kubernetes"]}'
    )
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=json_response))

    res = await linkml_extract(llm, "Extract resume", Resume)
    assert isinstance(res, Resume)
    assert res.name == "Sam Smith"
    assert "Kubernetes" in res.skills


@pytest.mark.asyncio
async def test_structured_extract_exhausts_retries_and_raises() -> None:
    llm = MagicMock()
    runnable = MagicMock()
    runnable.ainvoke = AsyncMock(
        return_value={
            "parsed": None,
            "raw": AIMessage(content="Total garbage that cannot produce a resume"),
        }
    )
    llm.with_structured_output.return_value = runnable

    with pytest.raises(LinkMLExtractionError) as exc_info:
        await linkml_extract(llm, "prompt", Resume, max_retries=1)
    assert "Failed to extract valid Resume" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Capability validation tests
# ---------------------------------------------------------------------------


def test_validate_model_capabilities_text_only_rejects_pdf() -> None:
    with pytest.raises(ModelCapabilityError) as exc_info:
        validate_model_capabilities("gpt-3.5-turbo@openai", requires_pdf=True)
    assert "lacks required capability 'pdf'" in str(exc_info.value)


def test_validate_model_capabilities_unknown_model_graceful() -> None:
    # If model is not in models.dev, it logs debug and allows execution
    validate_model_capabilities("custom-internal-model@custom", requires_pdf=True)


# ---------------------------------------------------------------------------
# LinkMLProcessor tests
# ---------------------------------------------------------------------------


def test_processor_initialization_requires_base_model() -> None:
    class NotAModel:
        pass

    with pytest.raises(TypeError):
        LinkMLProcessor(model_cls=NotAModel)  # type: ignore[arg-type]


def test_processor_prompt_rendering_with_contract() -> None:
    contract = LinkMLContract(
        RESUME_SCHEMA,
        root_class="Resume",
        generated_module="tests.generated.resume",
    )
    processor = LinkMLProcessor(
        model_cls=Resume,
        prompt="Candidate text:\n{{ document }}",
        contract=contract,
    )
    rendered = processor._render_prompt("John Smith text")
    assert "Candidate text:\nJohn Smith text" in rendered
    assert "### Field Extraction Guidance" in rendered
    assert "Full name of the candidate" in rendered


@pytest.mark.asyncio
async def test_processor_abatch_analyze_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = LinkMLProcessor(model_cls=Resume, prompt=PROMPT_FILE)

    fake_resume = Resume(
        name="Alice",
        email="alice@example.com",
        experience=["Eng"],
        skills=["Python"],
    )

    async def fake_aextract(doc_id: str, content: str | Path | bytes, is_pdf: bool = False) -> Resume:
        return fake_resume

    monkeypatch.setattr(processor, "aextract", fake_aextract)

    results = await processor.abatch_analyze_documents(
        document_ids=["doc1", "doc2"],
        contents=["content 1", "content 2"],
    )
    assert len(results) == 2
    assert results[0].name == "Alice"
    assert results[1].name == "Alice"
