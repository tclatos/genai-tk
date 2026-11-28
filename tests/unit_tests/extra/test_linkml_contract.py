"""Unit tests for LinkMLContract, SchemaView introspection, and neutral documentation."""

from pathlib import Path

import pytest
from pydantic import BaseModel

try:
    from genai_tk.extra.structured.exceptions import LinkMLDriftError, LinkMLSchemaNotFoundError
    from genai_tk.extra.structured.linkml_contract import (
        EnumDocumentation,
        LinkMLContract,
        SchemaDocumentation,
    )
except ImportError:
    pytest.skip("linkml feature not installed", allow_module_level=True)

RESUME_SCHEMA = Path(__file__).parent.parent.parent / "linkml_src" / "resume.yaml"
DELIVERY_NOTE_SCHEMA = Path(__file__).parent.parent.parent / "linkml_src" / "delivery_note.yaml"
ENUMS_SCHEMA = Path(__file__).parent.parent.parent / "linkml_src" / "enums_fixture.yaml"
GENERATED_RESUME = Path(__file__).parent.parent.parent / "generated" / "resume.py"


def test_schema_not_found_raises() -> None:
    with pytest.raises(LinkMLSchemaNotFoundError):
        LinkMLContract("non_existent_file_123.yaml")


def test_contract_schema_and_closure_hash() -> None:
    contract = LinkMLContract(RESUME_SCHEMA)
    h1 = contract.schema_hash
    h2 = contract.closure_hash
    assert isinstance(h1, str) and len(h1) > 0
    assert isinstance(h2, str) and len(h2) > 0
    # Deterministic
    assert contract.schema_hash == h1
    assert contract.closure_hash == h2


def test_contract_resolve_root_class() -> None:
    contract = LinkMLContract(
        RESUME_SCHEMA,
        root_class="Resume",
        generated_module="tests.generated.resume",
    )
    cls = contract.resolve_root_class()
    assert issubclass(cls, BaseModel)
    assert cls.__name__ == "Resume"


def test_contract_resolve_root_class_tree_root_detection() -> None:
    # If root_class is None, it should detect tree_root: true from schema
    contract = LinkMLContract(
        RESUME_SCHEMA,
        root_class=None,
        generated_module="tests.generated.resume",
    )
    cls = contract.resolve_root_class()
    assert cls.__name__ == "Resume"


def test_contract_schema_documentation() -> None:
    contract = LinkMLContract(ENUMS_SCHEMA)
    docs = contract.get_schema_documentation()
    assert isinstance(docs, SchemaDocumentation)
    assert "EnumItem" in docs.classes
    assert "item_id" in docs.fields["EnumItem"]
    assert "ProcessingStatus" in docs.enums

    enum_doc = docs.enums["ProcessingStatus"]
    assert isinstance(enum_doc, EnumDocumentation)
    assert enum_doc.description == "Lifecycle processing status of an item."
    assert "PENDING" in enum_doc.values
    assert enum_doc.values["PENDING"] == "Awaiting initial review or ingestion."
    assert enum_doc.values["COMPLETED"] == "Successfully analyzed and validated."


def test_contract_enum_glossary_generation() -> None:
    contract = LinkMLContract(ENUMS_SCHEMA)
    glossary = contract.get_enum_glossary()
    assert "### Allowed Values and Descriptions" in glossary
    assert "#### ProcessingStatus" in glossary
    assert "PENDING" in glossary
    assert "Awaiting initial review or ingestion." in glossary
    assert "COMPLETED" in glossary


def test_contract_slot_prompt_guidance() -> None:
    contract = LinkMLContract(RESUME_SCHEMA)
    guidance = contract.get_slot_prompt_guidance("Resume")
    assert "### Field Extraction Guidance" in guidance
    assert "name" in guidance
    assert "Full name of the candidate" in guidance


def test_contract_drift_check_matches_generated_file() -> None:
    contract = LinkMLContract(RESUME_SCHEMA)
    has_drift, diff = contract.check_drift(GENERATED_RESUME)
    assert not has_drift, f"Expected no drift, got: {diff}"
    assert diff == ""


def test_contract_drift_check_detects_modification(tmp_path: Path) -> None:
    contract = LinkMLContract(RESUME_SCHEMA)
    modified_target = tmp_path / "resume_modified.py"
    modified_target.write_text("# modified code", encoding="utf-8")

    has_drift, diff = contract.check_drift(modified_target)
    assert has_drift
    assert len(diff) > 0

    with pytest.raises(LinkMLDriftError):
        contract.validate_drift(modified_target)
