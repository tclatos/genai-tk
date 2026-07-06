"""Unit tests for ``baml_util.py``.

Covers the pure helpers (version parsing, ``@description`` extraction, BAML
content parsing, return-type introspection, function-parameter inspection) and
the error paths of the client-loading functions.  The success paths of
``load_baml_client`` / ``baml_invoke`` require a generated ``baml_client``
Python package, which is not present in this repo (only ``.baml`` source files
exist under ``tests/baml_src``); those paths are exercised by integration tests
with a real generated client.

The BAML client used as a collaborator in ``get_baml_function`` /
``get_function_parameters`` is a lightweight fake — a stand-in for the external
BAML client boundary.
"""

from collections.abc import Awaitable, Coroutine
from typing import Any

import pytest
from pydantic import BaseModel

try:  # Defensive: skip the whole module if the baml runtime can't initialise.
    from genai_tk.extra.structured import baml_util as baml
    from genai_tk.extra.structured.exceptions import (
        BamlClientLoadError,
        BamlVersionMismatchError,
    )
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"BAML runtime not available: {exc}", allow_module_level=True)

# --------------------------------------------------------------------------- #
# Fixtures / fakes
# --------------------------------------------------------------------------- #


class _Resume(BaseModel):
    """A Pydantic model used as a BAML return type in introspection tests."""

    name: str


class _FakeBamlClient:
    """Fake async BAML client (external BAML boundary stand-in)."""

    async def ExtractResume(self, *, text: str, baml_options: dict[str, Any] | None = None) -> Awaitable[_Resume]:
        return _Resume(name=text)

    async def NoArgs(self) -> Awaitable[_Resume]:
        return _Resume(name="none")


# --------------------------------------------------------------------------- #
# _parse_baml_versions
# --------------------------------------------------------------------------- #


def test_parse_baml_versions_generator_and_library() -> None:
    msg = "baml_client is out of date.\nGenerator version 0.220.0\nCurrent version baml-py 0.222.0"
    assert baml._parse_baml_versions(msg) == ("0.220.0", "0.222.0")


def test_parse_baml_versions_generator_only() -> None:
    assert baml._parse_baml_versions("baml_client generated with 0.220.0") == ("0.220.0", None)


def test_parse_baml_versions_library_only() -> None:
    assert baml._parse_baml_versions("please update baml-py to 0.222.0") == (None, "0.222.0")


def test_parse_baml_versions_no_versions() -> None:
    assert baml._parse_baml_versions("something went wrong 1.2.3 but no keywords") == (None, None)


def test_parse_baml_versions_empty() -> None:
    assert baml._parse_baml_versions("") == (None, None)


# --------------------------------------------------------------------------- #
# extract_baml_description
# --------------------------------------------------------------------------- #


def test_extract_baml_description_single_line_double_quote() -> None:
    lines = ['@description("hello world")']
    assert baml.extract_baml_description(lines[0], lines, 0) == ("hello world", 1)


def test_extract_baml_description_single_line_single_quote() -> None:
    lines = ["@description('hi')"]
    assert baml.extract_baml_description(lines[0], lines, 0) == ("hi", 1)


def test_extract_baml_description_multi_line_hash_string() -> None:
    lines = ['@description(#"', "long", 'text"#)', "}"]
    result = baml.extract_baml_description(lines[0], lines, 0)
    assert result is not None
    assert result[0] == "long text"


def test_extract_baml_description_no_description_returns_none() -> None:
    lines = ["class Foo {"]
    assert baml.extract_baml_description(lines[0], lines, 0) is None


def test_extract_baml_description_no_closing_hash_string_returns_none() -> None:
    lines = ['@description(#"', "text without closing"]
    assert baml.extract_baml_description(lines[0], lines, 0) is None


def test_extract_baml_description_malformed_single_line_returns_none() -> None:
    lines = ["@description(noquote)"]
    assert baml.extract_baml_description(lines[0], lines, 0) is None


def test_extract_baml_description_double_at_sign() -> None:
    lines = ['@@description("double")']
    assert baml.extract_baml_description(lines[0], lines, 0) == ("double", 1)


# --------------------------------------------------------------------------- #
# parse_baml_content
# --------------------------------------------------------------------------- #


def test_parse_baml_content_classes_fields_and_enums() -> None:
    content = (
        '@description("A person")\n'
        "class Person {\n"
        '  name string @description("full name")\n'
        '  age int @description("age in years")\n'
        "}\n"
        '@description("Status enum")\n'
        "enum Status {\n"
        '  ACTIVE @description("active value")\n'
        "  INACTIVE\n"
        "}\n"
    )
    classes: dict[str, str] = {}
    fields: dict[str, dict[str, str]] = {}
    enums: dict[str, dict[str, str]] = {}

    baml.parse_baml_content(content, classes, fields, enums)

    assert classes == {"Person": "A person"}
    assert fields == {"Person": {"name": "full name", "age": "age in years"}}
    assert enums == {"Status": {"ACTIVE": "active value", "INACTIVE": ""}}


def test_parse_baml_content_enum_without_description_still_registered() -> None:
    content = "enum Color {\n  RED\n  GREEN\n}\n"
    classes: dict[str, str] = {}
    fields: dict[str, dict[str, str]] = {}
    enums: dict[str, dict[str, str]] = {}

    baml.parse_baml_content(content, classes, fields, enums)

    assert "Color" in enums
    assert enums["Color"] == {"RED": "", "GREEN": ""}


def test_parse_baml_content_class_without_description_not_in_classes() -> None:
    content = "class Bare {\n  x int\n}\n"
    classes: dict[str, str] = {}
    fields: dict[str, dict[str, str]] = {}
    enums: dict[str, dict[str, str]] = {}

    baml.parse_baml_content(content, classes, fields, enums)

    assert "Bare" not in classes
    assert fields == {}


def test_parse_baml_content_inline_class_description() -> None:
    # @description on the class line itself (inline block description).
    content = 'class Person @description("inline") {\n  name string\n}\n'
    classes: dict[str, str] = {}
    fields: dict[str, dict[str, str]] = {}
    enums: dict[str, dict[str, str]] = {}

    baml.parse_baml_content(content, classes, fields, enums)

    assert classes == {"Person": "inline"}


def test_parse_baml_content_inline_enum_description_registers_enum() -> None:
    content = 'enum Status @description("en") {\n  ACTIVE\n}\n'
    classes: dict[str, str] = {}
    fields: dict[str, dict[str, str]] = {}
    enums: dict[str, dict[str, str]] = {}

    baml.parse_baml_content(content, classes, fields, enums)

    assert "Status" in enums


def test_parse_baml_content_description_line_inside_class_body_is_ignored() -> None:
    # A @description line inside a class body (preceding a field) is not on the
    # field line, so it is not captured — but it must not crash parsing.
    content = 'class Person {\n  @description("ignored")\n  name string\n}\n'
    classes: dict[str, str] = {}
    fields: dict[str, dict[str, str]] = {}
    enums: dict[str, dict[str, str]] = {}

    baml.parse_baml_content(content, classes, fields, enums)

    assert fields == {}


# --------------------------------------------------------------------------- #
# StructuredConfig
# --------------------------------------------------------------------------- #


def test_structured_config_holds_baml_client() -> None:
    cfg = baml.StructuredConfig(baml_client="some.pkg")
    assert cfg.baml_client == "some.pkg"


def test_structured_config_from_dict() -> None:
    cfg = baml.StructuredConfig.model_validate({"baml_client": "another.pkg"})
    assert cfg.baml_client == "another.pkg"


# --------------------------------------------------------------------------- #
# get_return_type_from_baml_function
# --------------------------------------------------------------------------- #


async def _awaitable_resume() -> Awaitable[_Resume]:
    return _Resume(name="x")


async def _coroutine_resume() -> Coroutine[Any, Any, _Resume]:
    return _Resume(name="x")


def _no_annotation() -> None:
    return None


def test_get_return_type_from_awaitable() -> None:
    assert baml.get_return_type_from_baml_function(_awaitable_resume) is _Resume


def test_get_return_type_from_coroutine() -> None:
    assert baml.get_return_type_from_baml_function(_coroutine_resume) is _Resume


def test_get_return_type_empty_annotation_returns_none() -> None:
    assert baml.get_return_type_from_baml_function(_no_annotation) is None


# --------------------------------------------------------------------------- #
# get_baml_function
# --------------------------------------------------------------------------- #


async def test_get_baml_function_returns_wrapper_and_type() -> None:
    client = _FakeBamlClient()
    fn, return_type = baml.get_baml_function(client, "ExtractResume")

    assert return_type is _Resume
    result = await fn(text="hi")
    assert isinstance(result, _Resume)
    assert result.name == "hi"


async def test_get_baml_function_wrapper_passes_baml_options() -> None:
    client = _FakeBamlClient()
    fn, _ = baml.get_baml_function(client, "ExtractResume")

    # Should not raise; the wrapper forwards baml_options to the underlying method.
    result = await fn(text="hi", baml_options={"client_registry": object()})
    assert isinstance(result, _Resume)


def test_get_baml_function_missing_raises_attribute_error() -> None:
    client = _FakeBamlClient()
    with pytest.raises(AttributeError, match="Unknown BAML function 'nope'"):
        baml.get_baml_function(client, "nope")


# --------------------------------------------------------------------------- #
# get_function_parameters
# --------------------------------------------------------------------------- #


def test_get_function_parameters_excludes_baml_options() -> None:
    client = _FakeBamlClient()
    assert baml.get_function_parameters(client, "ExtractResume") == ["text"]


def test_get_function_parameters_no_args() -> None:
    client = _FakeBamlClient()
    assert baml.get_function_parameters(client, "NoArgs") == []


# --------------------------------------------------------------------------- #
# create_baml_options / create_baml_client_registry (error paths)
# --------------------------------------------------------------------------- #


def test_create_baml_options_default_returns_none() -> None:
    assert baml.create_baml_options("default") is None


def test_create_baml_options_none_returns_none() -> None:
    assert baml.create_baml_options(None) is None


def test_create_baml_client_registry_fake_model_raises() -> None:
    # The parrot fake model is not an OpenAI-compatible client → unsupported for BAML.
    with pytest.raises(ValueError, match="Failed to get LLM info"):
        baml.create_baml_client_registry("parrot_local@fake")


def test_create_baml_options_with_fake_llm_raises() -> None:
    with pytest.raises(ValueError, match="Failed to get LLM info"):
        baml.create_baml_options("parrot_local@fake")


# --------------------------------------------------------------------------- #
# load_baml_client / load_and_validate_baml_function / prompt_fingerprint / baml_invoke
# (error paths — no generated baml_client package is present in this repo)
# --------------------------------------------------------------------------- #


def test_load_baml_client_unknown_config_raises_value_error() -> None:
    with pytest.raises(ValueError, match="BAML client package not found in config"):
        baml.load_baml_client("nonexistent_config")


def test_load_baml_client_default_raises_load_error_without_package() -> None:
    # The configured package (tests.baml_client) has no Python package here.
    with pytest.raises(BamlClientLoadError):
        baml.load_baml_client("default")


def test_load_and_validate_baml_function_propagates_load_error() -> None:
    with pytest.raises(BamlClientLoadError):
        baml.load_and_validate_baml_function("default", "ExtractResume")


def test_prompt_fingerprint_propagates_load_error() -> None:
    with pytest.raises(BamlClientLoadError):
        baml.prompt_fingerprint("ExtractResume")


async def test_baml_invoke_propagates_load_error() -> None:
    with pytest.raises(BamlClientLoadError):
        await baml.baml_invoke("ExtractResume", {"text": "hi"})


# --------------------------------------------------------------------------- #
# BamlVersionMismatchError is raised when the types import reports a version mismatch
# --------------------------------------------------------------------------- #


def test_load_baml_client_version_mismatch_raises_dedicated_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If importing the types module raises an 'out of date' error, a
    ``BamlVersionMismatchError`` is raised instead of ``BamlClientLoadError``.
    """
    import importlib

    real_import = importlib.import_module

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "tests.baml_client.types":
            raise ImportError("baml_client is out of date. Generator 0.220.0 vs baml-py 0.222.0")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    with pytest.raises(BamlVersionMismatchError):
        baml.load_baml_client("default")
