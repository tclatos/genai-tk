"""Prefect-powered LinkML extraction flow using LinkML and Pydantic.

Extracts structured data from Markdown, text, and PDF files into validated Pydantic
models defined by LinkML schemas, writing results to output directories with manifest caching.

Manifest tracking covers:
- source file content hash
- LinkML schema and import closure hash
- prompt template content hash
- resolved model identifier
- normalization / pipeline version
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner  # type: ignore[attr-defined]
from pydantic import BaseModel, Field

from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.config_mgmt.features import require_feature
from genai_tk.config_mgmt.file_patterns import resolve_config_path, resolve_files
from genai_tk.core.factories.llm_factory import LlmFactory
from genai_tk.extra.structured.config import LinkMLConfig
from genai_tk.extra.structured.linkml_contract import LinkMLContract
from genai_tk.extra.structured.linkml_processor import LinkMLProcessor
from genai_tk.utils.hashing import buffer_digest
from genai_tk.workflow.flow_cache.manifest import ManifestCache

require_feature("linkml", context="genai_tk.workflow.prefect.flows.linkml_flow")

PDF_EXTENSIONS = {".pdf"}
TEXT_EXTENSIONS = {".md", ".markdown", ".txt"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS
PIPELINE_VERSION = "v1"


def _safe_resolve_file(text: str) -> Path | None:
    if "\n" in text or len(text) > 1024:
        return None
    try:
        p = Path(os.path.expandvars(text)).expanduser()
        if p.is_file():
            return p
    except OSError:
        pass
    return None


class LinkMLExtractionManifestEntry(BaseModel):
    """A single extraction entry in the manifest."""

    source_hash: str
    output_path: str
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LinkMLExtractionManifest(BaseModel):
    """Manifest for LinkML extraction to avoid reprocessing unchanged files."""

    config_name: str
    schema_path: str | None = None
    root_class: str | None = None
    llm: str | None = None
    resolved_llm: str | None = None
    model_name: str | None = None
    entries: dict[str, LinkMLExtractionManifestEntry] = Field(default_factory=dict)


# Backward-compatibility aliases
StructuredExtractionManifestEntry = LinkMLExtractionManifestEntry
StructuredExtractionManifest = LinkMLExtractionManifest


class _FileToProcess(BaseModel):
    path: Path
    content_hash: str
    content_text: str
    content_bytes: bytes | None = None
    is_pdf: bool = False


def _compute_hash(content: bytes) -> str:
    return buffer_digest(content)


def _iter_supported_files(files: list[Path]) -> Iterable[Path]:
    for path in files:
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def _prepare_files(
    files: Iterable[Path],
    cache: ManifestCache,
    code_version: str | None,
    force: bool,
) -> tuple[list[_FileToProcess], int]:
    to_process: list[_FileToProcess] = []
    skipped = 0

    for path in files:
        try:
            content_bytes = path.read_bytes()
            content_hash = _compute_hash(content_bytes)
        except Exception as exc:
            logger.error("Error reading {}: {}", path, exc)
            continue

        if cache.is_fresh(str(path), fingerprint=content_hash, code_version=code_version, force=force):
            skipped += 1
            logger.info("Skipping unchanged file: {}", path)
            continue

        is_pdf = path.suffix.lower() in PDF_EXTENSIONS
        to_process.append(
            _FileToProcess(
                path=path,
                content_hash=content_hash,
                content_text="" if is_pdf else content_bytes.decode("utf-8", errors="replace"),
                content_bytes=content_bytes if is_pdf else None,
                is_pdf=is_pdf,
            )
        )

    return to_process, skipped


def _chunked[T](items: list[T], size: int) -> Iterable[list[T]]:
    if size <= 0:
        yield items
        return
    for i in range(0, len(items), size):
        yield items[i : i + size]


def resolve_linkml_contract(
    config_name: str,
    schema_path: str | None = None,
    root_class: str | None = None,
    generated_module: str | None = None,
    prompt: str | None = None,
    llm: str | None = None,
) -> tuple[LinkMLContract, type[BaseModel], str | None, str]:
    """Resolve LinkML contract, Pydantic root model, prompt, and effective LLM.

    Checks 'linkml' config section first, falling back to legacy 'structured'.
    """
    configs = global_config().section_dict("linkml", LinkMLConfig, inject_name=False)
    if not configs:
        configs = global_config().section_dict("structured", LinkMLConfig, inject_name=False)
    cfg = configs.get(config_name)

    eff_schema = schema_path or (cfg.schema_path if cfg else None)
    eff_root = root_class or (cfg.root_class if cfg else None)
    eff_module = generated_module or (cfg.generated_module if cfg else None)
    eff_prompt = prompt or (cfg.prompt if cfg else None)
    eff_llm = llm if (llm and llm != "default") else ((cfg.llm if cfg else None) or "default")

    if not eff_schema:
        raise ValueError(
            f"No schema specified for LinkML extraction config '{config_name}'. "
            "Provide 'schema' in config or pass --schema CLI argument."
        )

    contract = LinkMLContract(eff_schema, root_class=eff_root, generated_module=eff_module)
    model_cls = contract.resolve_root_class()

    return contract, model_cls, eff_prompt, eff_llm


# Backward-compatibility alias
resolve_structured_contract = resolve_linkml_contract


def compute_code_version(contract: LinkMLContract, prompt: str | None, llm: str) -> str:
    """Compute combined code_version covering schema closure, prompt, and model."""
    schema_hash = contract.closure_hash
    prompt_bytes = prompt.encode("utf-8") if prompt else b""
    prompt_hash = buffer_digest(prompt_bytes)
    return f"{schema_hash}:{prompt_hash}:{llm}:{PIPELINE_VERSION}"


@task
async def _process_single_file_task(
    file_info: _FileToProcess,
    model_cls: type[BaseModel],
    prompt: str | None,
    llm: str,
    contract: LinkMLContract,
    linkml_root: str,
    root_dir: str,
) -> tuple[str, str, str]:
    upath = file_info.path
    logger.info("Processing file: {}", upath)

    processor = LinkMLProcessor(
        model_cls=model_cls,
        prompt=prompt,
        llm=llm,
        contract=contract,
    )

    content: str | bytes = file_info.content_bytes if file_info.is_pdf else file_info.content_text
    result = await processor.aextract(
        document_id=upath.stem,
        content=content,
        is_pdf=file_info.is_pdf,
    )

    model_name = type(result).__name__
    output_root = Path(linkml_root) / model_name
    output_root.mkdir(parents=True, exist_ok=True)

    root_dir_path = Path(root_dir)
    try:
        relative_source_path = upath.relative_to(root_dir_path)
    except ValueError:
        relative_source_path = Path(upath.name)

    relative_output_path = relative_source_path.with_suffix(".json")
    output_path = output_root / relative_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_text = result.model_dump_json(indent=2)
    output_path.write_text(json_text, encoding="utf-8")
    logger.success("Wrote LinkML output to {}", output_path)

    return str(upath), str(relative_output_path), model_name


@flow(name="linkml_extraction", task_runner=ConcurrentTaskRunner())  # type: ignore[call-arg]
def linkml_extraction_flow(
    base_dir: str,
    output_dir: str,
    *,
    config_name: str = "default",
    schema_path: str | None = None,
    root_class: str | None = None,
    generated_module: str | None = None,
    prompt: str | None = None,
    llm: str = "default",
    pathspecs: list[str] | None = None,
    batch_size: int = 5,
    force: bool = False,
) -> LinkMLExtractionManifest:
    """Run LinkML extraction as a Prefect flow."""
    contract, model_cls, eff_prompt, eff_llm = resolve_linkml_contract(
        config_name=config_name,
        schema_path=schema_path,
        root_class=root_class,
        generated_module=generated_module,
        prompt=prompt,
        llm=llm,
    )

    if pathspecs is None:
        pathspecs = ["**/*.md", "**/*.pdf", "**/*.txt"]

    file_paths = resolve_files(base_dir, pathspecs=pathspecs)
    resolved_output = resolve_config_path(output_dir)
    linkml_root_path = Path(resolved_output)
    linkml_root_path.mkdir(parents=True, exist_ok=True)

    resolved_llm: str | None = None
    if eff_llm and eff_llm != "default":
        resolved_llm = LlmFactory(llm=eff_llm).get_id()

    code_version = compute_code_version(contract, eff_prompt, eff_llm)
    manifest_path = linkml_root_path / "manifest.json"
    cache = ManifestCache.load(manifest_path)

    files = list(_iter_supported_files([Path(p) for p in file_paths]))
    to_process, skipped = _prepare_files(files, cache, code_version=code_version, force=force)

    if skipped:
        logger.info("Skipped {} unchanged files based on manifest", skipped)

    if not to_process:
        logger.info("No files left to process after manifest filtering")
        return LinkMLExtractionManifest(
            config_name=config_name,
            schema_path=str(contract.schema_path),
            root_class=model_cls.__name__,
            llm=eff_llm,
            resolved_llm=resolved_llm,
            model_name=model_cls.__name__,
            entries={
                k: LinkMLExtractionManifestEntry(
                    source_hash=rec.fingerprint,
                    output_path=rec.outputs.get("output_path", ""),
                    processed_at=rec.processed_at,
                )
                for k, rec in cache.records.items()
            },
        )

    logger.info("Processing {} files with LinkML extraction", len(to_process))

    for batch in _chunked(to_process, batch_size):
        futures = [
            _process_single_file_task.submit(
                file_info,
                model_cls=model_cls,
                prompt=eff_prompt,
                llm=eff_llm,
                contract=contract,
                linkml_root=str(linkml_root_path),
                root_dir=base_dir,
            )
            for file_info in batch
        ]

        for future in futures:
            result = future.result()  # type: ignore[misc]
            if asyncio.iscoroutine(result):
                source_path, relative_output_path, model_name = asyncio.run(result)  # type: ignore[misc]
            else:
                source_path, relative_output_path, model_name = result  # type: ignore[misc]

            file_hash = next((f.content_hash for f in to_process if str(f.path) == source_path), "")
            cache.record_success(
                key=source_path,
                fingerprint=file_hash,
                outputs={"output_path": relative_output_path},
                code_version=code_version,
            )

    cache.save(manifest_path)
    logger.success("LinkML extraction completed. {} files processed, {} skipped.", len(to_process), skipped)

    return LinkMLExtractionManifest(
        config_name=config_name,
        schema_path=str(contract.schema_path),
        root_class=model_cls.__name__,
        llm=eff_llm,
        resolved_llm=resolved_llm,
        model_name=model_cls.__name__,
        entries={
            k: LinkMLExtractionManifestEntry(
                source_hash=rec.fingerprint,
                output_path=rec.outputs.get("output_path", ""),
                processed_at=rec.processed_at,
            )
            for k, rec in cache.records.items()
        },
    )


# Backward-compatibility alias
structured_extraction_flow = linkml_extraction_flow


@task
async def _process_single_input_task(
    input_text: str,
    model_cls: type[BaseModel],
    prompt: str | None,
    llm: str,
    contract: LinkMLContract,
    output_dir: str | None,
    output_file: str | None,
    input_hash: str,
    force: bool,
    code_version: str,
    cache: ManifestCache | None,
) -> tuple[BaseModel, str, str | None]:
    """Process a single text or file input and optionally save result."""
    upath = _safe_resolve_file(input_text)
    is_pdf = upath is not None and upath.suffix.lower() in PDF_EXTENSIONS

    if not force and cache and output_dir and output_file:
        input_key = f"input:{input_hash}"
        if cache.is_fresh(input_key, fingerprint=input_hash, code_version=code_version, force=force):
            record = cache.records.get(input_key)
            out_rel = record.outputs.get("output_path") if record else output_file
            out_full = Path(output_dir) / out_rel
            if out_full.exists():
                logger.info("Skipping - result already fresh: {}", out_full)
                data = json.loads(out_full.read_text(encoding="utf-8"))
                return model_cls.model_validate(data), model_cls.__name__, str(out_rel)

    processor = LinkMLProcessor(
        model_cls=model_cls,
        prompt=prompt,
        llm=llm,
        contract=contract,
    )

    doc_id = upath.stem if upath else "input"
    content: str | bytes = (
        upath.read_bytes() if (upath and is_pdf) else (upath.read_text(encoding="utf-8") if upath else input_text)
    )

    result = await processor.aextract(
        document_id=doc_id,
        content=content,
        is_pdf=is_pdf,
    )

    model_name = type(result).__name__

    if not output_dir or not output_file:
        return result, model_name, None

    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / output_file
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.success("Wrote single result to {}", out_path)

    relative_output_path = f"{model_name}/{output_file}"
    return result, model_name, relative_output_path


@flow(name="linkml_single_input")
def linkml_single_input_flow(
    input_text: str,
    *,
    config_name: str = "default",
    schema_path: str | None = None,
    root_class: str | None = None,
    generated_module: str | None = None,
    prompt: str | None = None,
    llm: str = "default",
    output_dir: str | None = None,
    output_file: str | None = None,
    force: bool = False,
) -> tuple[BaseModel, str | None, str | None]:
    """Execute LinkML extraction on a single input text or file path."""
    contract, model_cls, eff_prompt, eff_llm = resolve_linkml_contract(
        config_name=config_name,
        schema_path=schema_path,
        root_class=root_class,
        generated_module=generated_module,
        prompt=prompt,
        llm=llm,
    )

    resolved_output_dir = resolve_config_path(output_dir) if output_dir else None
    input_hash = buffer_digest(input_text.encode("utf-8"))
    code_version = compute_code_version(contract, eff_prompt, eff_llm)

    cache: ManifestCache | None = None
    manifest_path: Path | None = None
    if resolved_output_dir:
        manifest_path = Path(resolved_output_dir) / "manifest.json"
        cache = ManifestCache.load(manifest_path)

    future = _process_single_input_task.submit(
        input_text=input_text,
        model_cls=model_cls,
        prompt=eff_prompt,
        llm=eff_llm,
        contract=contract,
        output_dir=resolved_output_dir,
        output_file=output_file,
        input_hash=input_hash,
        force=force,
        code_version=code_version,
        cache=cache,
    )

    task_res = future.result()  # type: ignore[misc]
    if asyncio.iscoroutine(task_res):
        result, model_name, relative_output_path = asyncio.run(task_res)  # type: ignore[misc]
    else:
        result, model_name, relative_output_path = task_res  # type: ignore[misc]

    if cache and manifest_path and relative_output_path:
        cache.record_success(
            key=f"input:{input_hash}",
            fingerprint=input_hash,
            outputs={"output_path": relative_output_path},
            code_version=code_version,
        )
        cache.save(manifest_path)

    return result, model_name, relative_output_path


# Backward-compatibility alias
structured_single_input_flow = linkml_single_input_flow
