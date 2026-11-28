"""LinkML contract management, SchemaView introspection, and neutral documentation.

This module provides the LinkMLContract class for genai-tk, which owns:
- LinkML schema loading and SchemaView caching
- Schema and import closure hashing
- Generated Pydantic root class resolution
- Neutral schema documentation (classes, fields, enums, permissible values)
- Compact enum glossary generation for prompts
- OntoGPT-compatible prompt annotation extraction (prompt, prompt.examples, prompt.skip)
- Pydantic code generation and CI drift detection

Important architecture invariant:
This module must NOT import genai-graph.
"""

from __future__ import annotations

import difflib
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from genai_tk.config_mgmt.features import require_feature
from genai_tk.extra.structured.exceptions import (
    LinkMLDriftError,
    LinkMLSchemaNotFoundError,
)
from genai_tk.utils.hashing import buffer_digest

require_feature("linkml", context="genai_tk.extra.structured.linkml_contract")

from linkml.generators.pydanticgen import PydanticGenerator  # noqa: E402
from linkml_runtime.utils.schemaview import SchemaView  # noqa: E402

# ---------------------------------------------------------------------------
# Neutral Schema Documentation Models
# ---------------------------------------------------------------------------


class EnumDocumentation(BaseModel):
    """Documentation for an enumeration and its permissible values."""

    description: str | None = None
    values: dict[str, str] = Field(default_factory=dict)
    """Mapping of permissible_value_name -> description."""

    model_config = ConfigDict(extra="ignore")


class SchemaDocumentation(BaseModel):
    """Framework-neutral schema documentation extracted from LinkML/SchemaView.

    Carried alongside GraphSchema through the registry to feed Text-to-Cypher,
    Markdown, JSON, and HTML exports without parsing BAML.
    """

    classes: dict[str, str] = Field(default_factory=dict)
    """Mapping of class_name -> class description."""

    fields: dict[str, dict[str, str]] = Field(default_factory=dict)
    """Mapping of class_name -> {field_name -> field description}."""

    enums: dict[str, EnumDocumentation] = Field(default_factory=dict)
    """Mapping of enum_name -> EnumDocumentation."""

    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Annotation and Metadata Helpers
# ---------------------------------------------------------------------------


def _normalize_source_file_metadata(code: str) -> str:
    """Normalize 'source_file' metadata line to avoid false drift across relative/absolute paths."""
    import re

    return re.sub(r"['\"]source_file['\"]\s*:\s*['\"][^'\"]+['\"]", "'source_file': '<normalized>'", code)


def get_annotation_value(annotations: Any, key: str) -> Any:
    """Extract an annotation value from a LinkML element's annotations container."""
    if not annotations:
        return None
    try:
        val = annotations[key]
        return getattr(val, "value", val)
    except (KeyError, TypeError, AttributeError, IndexError):
        pass
    if hasattr(annotations, "_as_dict"):
        d = annotations._as_dict
        if key in d:
            val = d[key]
            return val.get("value", val) if isinstance(val, dict) else getattr(val, "value", val)
    return None


# ---------------------------------------------------------------------------
# LinkML Contract
# ---------------------------------------------------------------------------


class LinkMLContract:
    """Manages LinkML schema metadata, generated Pydantic models, and documentation."""

    def __init__(
        self,
        schema_path: Path | str,
        root_class: str | None = None,
        generated_module: str | ModuleType | None = None,
    ) -> None:
        self.schema_path = Path(schema_path).resolve()
        if not self.schema_path.exists():
            raise LinkMLSchemaNotFoundError(str(self.schema_path))

        self.root_class_name = root_class
        self.generated_module = generated_module

        self._schemaview: SchemaView | None = None
        self._schema_hash: str | None = None
        self._closure_hash: str | None = None
        self._resolved_root_class: type[BaseModel] | None = None
        self._schema_docs: SchemaDocumentation | None = None

    @property
    def schemaview(self) -> SchemaView:
        """Cached SchemaView instance for the schema."""
        if self._schemaview is None:
            self._schemaview = SchemaView(str(self.schema_path))
        return self._schemaview

    @property
    def schema_hash(self) -> str:
        """Fingerprint hash of the primary schema file content."""
        if self._schema_hash is None:
            self._schema_hash = buffer_digest(self.schema_path.read_bytes())
        return self._schema_hash

    @property
    def closure_hash(self) -> str:
        """Stable combined hash of the schema file and all locally resolvable imports."""
        if self._closure_hash is None:
            combined = bytearray(self.schema_path.read_bytes())
            schema_dir = self.schema_path.parent

            # Inspect imports from the schema
            schema_obj = self.schemaview.schema
            if hasattr(schema_obj, "imports") and schema_obj.imports:
                for imp in sorted(schema_obj.imports):
                    # Check if import corresponds to a local file
                    imp_str = str(imp)
                    if imp_str.startswith("linkml:"):
                        continue
                    candidate = schema_dir / imp_str
                    if not candidate.suffix and not candidate.exists():
                        candidate_yaml = candidate.with_suffix(".yaml")
                        if candidate_yaml.exists():
                            candidate = candidate_yaml
                    if candidate.is_file():
                        combined.extend(candidate.read_bytes())

            self._closure_hash = buffer_digest(bytes(combined))
        return self._closure_hash

    def resolve_root_class(self) -> type[BaseModel]:
        """Resolve and return the Pydantic root class from the generated module."""
        if self._resolved_root_class is not None:
            return self._resolved_root_class

        target_name = self.root_class_name
        if not target_name:
            # Find class with tree_root: true
            roots = [name for name, c in self.schemaview.all_classes().items() if c.tree_root]
            if not roots:
                raise ValueError(
                    f"No root class specified and no class marked with 'tree_root: true' in {self.schema_path}"
                )
            target_name = roots[0]
            self.root_class_name = target_name

        if self.generated_module is None:
            raise ValueError(
                f"Cannot resolve root class '{target_name}': 'generated_module' was not provided to LinkMLContract"
            )

        if isinstance(self.generated_module, str):
            mod = importlib.import_module(self.generated_module)
        else:
            mod = self.generated_module

        if not hasattr(mod, target_name):
            raise AttributeError(
                f"Generated module '{getattr(mod, '__name__', str(mod))}' has no class named '{target_name}'"
            )

        cls = getattr(mod, target_name)
        if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
            raise TypeError(f"Resolved root '{target_name}' must be a Pydantic BaseModel subclass, got {cls}")

        self._resolved_root_class = cls
        return cls

    def get_schema_documentation(self) -> SchemaDocumentation:
        """Extract neutral schema documentation from the schema."""
        if self._schema_docs is not None:
            return self._schema_docs

        sv = self.schemaview
        classes_doc: dict[str, str] = {}
        fields_doc: dict[str, dict[str, str]] = {}
        enums_doc: dict[str, EnumDocumentation] = {}

        for cls_name, cls_def in sv.all_classes().items():
            classes_doc[cls_name] = cls_def.description or cls_def.title or ""
            fields_for_cls: dict[str, str] = {}
            for slot in sv.class_induced_slots(cls_name):
                desc = slot.description or slot.title or ""
                if desc:
                    fields_for_cls[slot.name] = desc
            if fields_for_cls:
                fields_doc[cls_name] = fields_for_cls

        for enum_name, enum_def in sv.all_enums().items():
            enum_desc = enum_def.description or enum_def.title or ""
            values_map: dict[str, str] = {}
            if hasattr(enum_def, "permissible_values") and enum_def.permissible_values:
                for pv_name, pv in enum_def.permissible_values.items():
                    pv_desc = pv.description or pv.title or ""
                    if pv_desc:
                        values_map[pv_name] = pv_desc
            enums_doc[enum_name] = EnumDocumentation(description=enum_desc, values=values_map)

        self._schema_docs = SchemaDocumentation(
            classes=classes_doc,
            fields=fields_doc,
            enums=enums_doc,
        )
        return self._schema_docs

    def get_enum_glossary(self, enum_names: list[str] | None = None) -> str:
        """Generate a compact Markdown glossary of enums and permissible value descriptions.

        Useful for appending to extraction prompts so the LLM receives complete permissible-value
        definitions even when JSON Schema omits them.
        """
        docs = self.get_schema_documentation()
        if not docs.enums:
            return ""

        target_enums = enum_names if enum_names is not None else sorted(docs.enums.keys())
        lines: list[str] = ["### Allowed Values and Descriptions\n"]

        for enum_name in target_enums:
            enum_info = docs.enums.get(enum_name)
            if not enum_info:
                continue
            lines.append(f"#### {enum_name}")
            if enum_info.description:
                lines.append(f"{enum_info.description}\n")
            if enum_info.values:
                for val, val_desc in enum_info.values.items():
                    lines.append(f"- `{val}`: {val_desc}")
            lines.append("")

        return "\n".join(lines).strip()

    def get_slot_prompt_guidance(self, class_name: str | None = None) -> str:
        """Extract OntoGPT-style prompt annotations from slots for prompt formatting.

        Supports:
        - prompt: specific prompt instruction for the slot
        - prompt.examples: examples for the slot
        - prompt.skip: skip instruction generation for this slot
        """
        sv = self.schemaview
        cls_name = class_name or self.root_class_name
        if not cls_name:
            roots = [name for name, c in sv.all_classes().items() if c.tree_root]
            if roots:
                cls_name = roots[0]
        if not cls_name:
            return ""

        guidance_lines: list[str] = []
        for slot in sv.class_induced_slots(cls_name):
            annotations = getattr(slot, "annotations", None)
            if not annotations:
                continue

            skip = get_annotation_value(annotations, "prompt.skip")
            if skip in (True, "true", "True", "1"):
                continue

            prompt_text = get_annotation_value(annotations, "prompt")
            examples = get_annotation_value(annotations, "prompt.examples")

            if prompt_text or examples:
                guidance_lines.append(f"- **{slot.name}**:")
                if prompt_text:
                    guidance_lines.append(f"  Guidance: {prompt_text}")
                if examples:
                    guidance_lines.append(f"  Examples: {examples}")

        if not guidance_lines:
            return ""

        return "### Field Extraction Guidance\n" + "\n".join(guidance_lines)

    def generate_pydantic_code(self, extra_fields: str = "forbid", format_with_ruff: bool = True) -> str:
        """Generate Pydantic v2 code from this LinkML schema."""
        import subprocess

        gen = PydanticGenerator(str(self.schema_path), extra_fields=extra_fields)
        code = gen.serialize()
        if format_with_ruff:
            try:
                res = subprocess.run(
                    ["ruff", "format", "-"],
                    input=code,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if res.returncode == 0 and res.stdout:
                    code = res.stdout
            except Exception:
                pass
        return code

    def check_drift(self, target_file: Path | str, extra_fields: str = "forbid") -> tuple[bool, str]:
        """Compare generated Pydantic code against target file.

        Returns:
            (has_drift, diff_text)
        """
        target = Path(target_file)
        if not target.exists():
            return True, f"Target file '{target}' does not exist."

        current_code = target.read_text(encoding="utf-8")
        generated_code = self.generate_pydantic_code(extra_fields=extra_fields)

        current_norm = _normalize_source_file_metadata(current_code).strip()
        generated_norm = _normalize_source_file_metadata(generated_code).strip()

        if current_norm == generated_norm:
            return False, ""

        diff = difflib.unified_diff(
            current_code.splitlines(keepends=True),
            generated_code.splitlines(keepends=True),
            fromfile=str(target),
            tofile="<generated-from-linkml>",
        )
        return True, "".join(diff)

    def validate_drift(self, target_file: Path | str, extra_fields: str = "forbid") -> None:
        """Raise LinkMLDriftError if generated code differs from target_file."""
        has_drift, diff = self.check_drift(target_file, extra_fields=extra_fields)
        if has_drift:
            raise LinkMLDriftError(str(self.schema_path), str(target_file), diff)
