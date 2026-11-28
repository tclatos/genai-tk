"""LinkML-based extraction processor using Pydantic models.

Uses LangChain chat models and with_structured_output(include_raw=True)
with fast-path parsed extraction, json_repair schema-mode recovery,
strict JSON fallback for models without native structured output,
model capability checks via models.dev, and bounded retries.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Generic, TypeVar

from langchain_core.messages import HumanMessage
from loguru import logger
from pydantic import BaseModel, ValidationError

from genai_tk.config_mgmt.features import require_feature
from genai_tk.core.factories.llm_factory import LlmFactory, get_llm
from genai_tk.extra.structured.exceptions import (
    LinkMLExtractionError,
    ModelCapabilityError,
)
from genai_tk.extra.structured.linkml_contract import LinkMLContract

require_feature("linkml", context="genai_tk.extra.structured.linkml_processor")

import json_repair  # noqa: E402

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# JSON Extraction Helper
# ---------------------------------------------------------------------------


def extract_json_from_raw(raw: Any) -> str:
    """Extract raw JSON text from an AIMessage, tool call, or raw response object."""
    if raw is None:
        return ""

    if isinstance(raw, str):
        return raw.strip()

    if isinstance(raw, dict):
        if "tool_calls" in raw and raw["tool_calls"]:
            args = raw["tool_calls"][0].get("args")
            if isinstance(args, dict):
                return json.dumps(args)
            if isinstance(args, str):
                return args.strip()
        if "content" in raw:
            return extract_json_from_raw(raw["content"])
        return json.dumps(raw)

    if hasattr(raw, "tool_calls") and raw.tool_calls:
        args = raw.tool_calls[0].get("args")
        if isinstance(args, dict):
            return json.dumps(args)
        if isinstance(args, str):
            return args.strip()

    if hasattr(raw, "content"):
        content = raw.content
        if isinstance(content, list):
            text_parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in content]
            return "".join(text_parts).strip()
        return str(content).strip()

    return str(raw).strip()


# ---------------------------------------------------------------------------
# Model Capability Validation
# ---------------------------------------------------------------------------


def validate_model_capabilities(
    llm_identifier: str,
    requires_pdf: bool = False,
    requires_vision: bool = False,
) -> None:
    """Verify that the chosen model supports required modalities via models.dev."""
    try:
        factory = LlmFactory(llm=llm_identifier)
        profile = factory.info.profile
    except Exception as exc:
        logger.debug("Could not inspect models.dev profile for '{}': {}", llm_identifier, exc)
        return

    if profile is None:
        return

    if requires_pdf and not profile.has_pdf:
        raise ModelCapabilityError(
            model_id=llm_identifier,
            required_capability="pdf",
            details=(
                f"Model '{llm_identifier}' does not declare PDF input capability. "
                f"Declared input modalities: {profile.modalities_input}. "
                "Consider converting documents to Markdown first or selecting a vision/pdf model."
            ),
        )

    if requires_vision and not profile.has_vision:
        raise ModelCapabilityError(
            model_id=llm_identifier,
            required_capability="vision",
            details=(
                f"Model '{llm_identifier}' does not declare image/vision input capability. "
                f"Declared input modalities: {profile.modalities_input}."
            ),
        )


# ---------------------------------------------------------------------------
# Low-Level LinkML Extract Function
# ---------------------------------------------------------------------------


async def linkml_extract(
    llm: Any,
    prompt_messages: Any,
    model: type[T],
    max_retries: int = 2,
    fallback_prompt: str | None = None,
) -> T:
    """Extract structured data adhering to a LinkML Pydantic model with fast path and json_repair recovery.

    Fast path: native with_structured_output(model, include_raw=True) returns result['parsed'].
    Recovery path: json_repair in schema mode repairs raw JSON and validates against model.
    Fallback path: for models without native structured output, prompts with JSON schema and repairs.
    """
    runnable = None
    use_native = False

    try:
        if hasattr(llm, "with_structured_output"):
            runnable = llm.with_structured_output(model, include_raw=True)
            use_native = True
    except (NotImplementedError, AttributeError, Exception) as exc:
        logger.debug("Native with_structured_output not available ({}): using schema prompt fallback", exc)
        use_native = False

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            if use_native and runnable is not None:
                result = await runnable.ainvoke(prompt_messages)

                # Fast path: check if parsed successfully
                if isinstance(result, dict) and result.get("parsed") is not None:
                    try:
                        return model.model_validate(result["parsed"])
                    except ValidationError as val_err:
                        last_error = val_err
                        logger.warning("Native parsed output failed local revalidation: {}", val_err)

                # Recovery path: parse/repair raw output using json_repair with schema mode
                raw = result.get("raw") if isinstance(result, dict) else result
                raw_json = extract_json_from_raw(raw)

                if raw_json:
                    repaired = json_repair.repair_json(
                        raw_json,
                        schema=model.model_json_schema(),
                        return_objects=True,
                    )
                    if isinstance(repaired, dict):
                        return model.model_validate(repaired)
            else:
                # Fallback path for models without with_structured_output
                schema_json = json.dumps(model.model_json_schema(), indent=2)
                schema_instruction = f"\n\nRespond ONLY with a valid JSON object matching this schema:\n{schema_json}"

                if isinstance(prompt_messages, list):
                    augmented_messages = list(prompt_messages)
                    if augmented_messages and hasattr(augmented_messages[-1], "content"):
                        last_content = augmented_messages[-1].content
                        if isinstance(last_content, str):
                            augmented_messages[-1] = HumanMessage(content=last_content + schema_instruction)
                else:
                    augmented_messages = str(prompt_messages) + schema_instruction

                resp = await llm.ainvoke(augmented_messages)
                raw_json = extract_json_from_raw(resp)

                if raw_json:
                    repaired = json_repair.repair_json(
                        raw_json,
                        schema=model.model_json_schema(),
                        return_objects=True,
                    )
                    if isinstance(repaired, dict):
                        return model.model_validate(repaired)

        except Exception as exc:
            last_error = exc
            logger.warning(
                "LinkML extraction attempt {}/{} failed for {}: {}",
                attempt + 1,
                max_retries + 1,
                model.__name__,
                exc,
            )

        if attempt < max_retries:
            await asyncio.sleep(0.5 * (attempt + 1))

    raise LinkMLExtractionError(
        model_name=model.__name__,
        reason=f"Failed to extract valid {model.__name__} after {max_retries + 1} attempts. Last error: {last_error}",
    )


# Backward-compatibility alias
structured_extract = linkml_extract


# ---------------------------------------------------------------------------
# High-Level Processor
# ---------------------------------------------------------------------------


class LinkMLProcessor(Generic[T]):
    """LinkML structured extraction processor using Pydantic models.

    Args:
        model_cls: Required Pydantic BaseModel class for extracted output.
        prompt: Optional prompt template string or Path to prompt markdown file.
        llm: LLM identifier / tag (default: 'default').
        kvstore_id: Optional KV store identifier for caching.
        force: Whether to bypass cache.
        max_retries: Bounded retries on malformed or unparseable output.
        contract: Optional LinkMLContract to enrich prompts with enum glossaries and slot guidance.
    """

    def __init__(
        self,
        model_cls: type[T],
        prompt: str | Path | None = None,
        llm: str = "default",
        kvstore_id: str | None = None,
        force: bool = False,
        max_retries: int = 2,
        contract: LinkMLContract | None = None,
    ) -> None:
        if not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
            raise TypeError(f"model_cls must be a Pydantic BaseModel subclass, got {model_cls}")

        self.model_cls = model_cls
        self.llm = llm
        self.kvstore_id = kvstore_id
        self.force = force
        self.max_retries = max_retries
        self.contract = contract

        self._prompt_template = self._load_prompt_template(prompt)

    @staticmethod
    def _load_prompt_template(prompt: str | Path | None) -> str:
        if prompt is None:
            return ""
        if isinstance(prompt, Path) or (isinstance(prompt, str) and "\n" not in prompt and len(prompt) < 1024):
            try:
                p = Path(prompt).expanduser()
                if p.is_file():
                    return p.read_text(encoding="utf-8")
            except OSError:
                pass
        return str(prompt)

    def _render_prompt(self, document_content: str) -> str:
        """Render the extraction prompt including document text and LinkML glossary/guidance."""
        base = self._prompt_template or "Extract structured data from the document:\n\n{{ document }}"

        if "{{ document }}" in base:
            rendered = base.replace("{{ document }}", document_content)
        elif "{{ document_text }}" in base:
            rendered = base.replace("{{ document_text }}", document_content)
        elif "{{ content }}" in base:
            rendered = base.replace("{{ content }}", document_content)
        elif "{document}" in base:
            rendered = base.replace("{document}", document_content)
        elif "{content}" in base:
            rendered = base.replace("{content}", document_content)
        else:
            rendered = f"{base}\n\nDocument:\n{document_content}"

        if self.contract is not None:
            guidance = self.contract.get_slot_prompt_guidance(self.model_cls.__name__)
            if guidance:
                rendered += f"\n\n{guidance}"

            glossary = self.contract.get_enum_glossary()
            if glossary:
                rendered += f"\n\n{glossary}"

        return rendered

    async def aextract(
        self,
        document_id: str,
        content: str | Path | bytes,
        is_pdf: bool = False,
    ) -> T:
        """Asynchronously extract structured data from a single document."""
        from genai_tk.utils.pydantic_utils.kv_store import PydanticStore

        if self.kvstore_id and not self.force:
            cached = await PydanticStore(kvstore_id=self.kvstore_id, model=self.model_cls).load_object(document_id)
            if cached is not None:
                logger.debug("Loaded cached extraction for '{}'", document_id)
                return cached

        validate_model_capabilities(self.llm, requires_pdf=is_pdf)

        pdf_bytes: bytes | None = None
        if is_pdf:
            if isinstance(content, Path):
                pdf_bytes = content.read_bytes()
            elif isinstance(content, bytes):
                pdf_bytes = content
            doc_text = f"[Attached PDF: {document_id}]"
        else:
            if isinstance(content, Path):
                doc_text = content.read_text(encoding="utf-8")
            elif isinstance(content, bytes):
                doc_text = content.decode("utf-8", errors="replace")
            else:
                doc_text = str(content)

        rendered_prompt = self._render_prompt(doc_text)

        if is_pdf and pdf_bytes is not None:
            pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
            messages = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": rendered_prompt},
                        {
                            "type": "media",
                            "mime_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    ]
                )
            ]
        else:
            messages = [HumanMessage(content=rendered_prompt)]

        llm_instance = get_llm(llm=self.llm)

        try:
            import nemo_relay

            has_relay = True
        except ImportError:
            has_relay = False

        if has_relay:
            with nemo_relay.scope.scope(
                name=f"linkml.{self.model_cls.__name__}",
                scope_type=nemo_relay.ScopeType.Function,
                input={"document_id": document_id, "model": self.model_cls.__name__},
                metadata={"llm": self.llm},
            ):
                result = await linkml_extract(
                    llm=llm_instance,
                    prompt_messages=messages,
                    model=self.model_cls,
                    max_retries=self.max_retries,
                )
        else:
            result = await linkml_extract(
                llm=llm_instance,
                prompt_messages=messages,
                model=self.model_cls,
                max_retries=self.max_retries,
            )

        if self.kvstore_id:
            await PydanticStore(kvstore_id=self.kvstore_id, model=self.model_cls).save_obj(document_id, result)

        return result

    def extract(
        self,
        document_id: str,
        content: str | Path | bytes,
        is_pdf: bool = False,
    ) -> T:
        """Synchronous extraction for a single document."""
        try:
            return asyncio.run(self.aextract(document_id, content, is_pdf=is_pdf))
        except RuntimeError:
            import nest_asyncio

            nest_asyncio.apply()
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self.aextract(document_id, content, is_pdf=is_pdf))

    def analyze_document(self, document_id: str, markdown: str) -> T:
        """Compatibility method for document analysis."""
        return self.extract(document_id, markdown, is_pdf=False)

    async def abatch_analyze_documents(
        self,
        document_ids: list[str],
        contents: list[str | Path],
    ) -> list[T]:
        """Process multiple documents concurrently."""
        tasks = [
            self.aextract(
                doc_id,
                content,
                is_pdf=str(content).lower().endswith(".pdf") if isinstance(content, (str, Path)) else False,
            )
            for doc_id, content in zip(document_ids, contents, strict=True)
        ]

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        successful: list[T] = []

        for doc_id, res in zip(document_ids, raw_results, strict=True):
            if isinstance(res, Exception):
                logger.error("Failed to extract document '{}': {}", doc_id, res)
            elif isinstance(res, self.model_cls):
                successful.append(res)

        return successful


# Backward-compatibility alias
PydanticStructuredProcessor = LinkMLProcessor
