"""Image description extraction, VLM analysis, and caching for uncaptioned document images."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from genai_tk.utils.hashing import buffer_digest
from genai_tk.utils.pydantic_utils.kv_store import PydanticStore


class ImageDescriptionCache(BaseModel):
    """Cached description and keywords for an extracted document image."""

    image_hash: str = Field(..., description="Unique hash of the image content")
    description: str = Field(..., description="VLM or caption description of the image")
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords for search")
    model: str | None = Field(default=None, description="VLM model identifier used")


def _get_local_cache_file(images_dir: Path | str | None) -> Path | None:
    """Get the path to the sidecar JSON cache file in images_dir."""
    if not images_dir:
        return None
    dir_path = Path(images_dir)
    return dir_path / ".image_descriptions.json"


def load_cached_image_description(
    image_hash: str,
    images_dir: Path | str | None = None,
    kvstore_id: str | None = None,
) -> ImageDescriptionCache | None:
    """Load cached image description from KV store or fallback file cache."""
    # 1. Try KV store if available
    if kvstore_id:
        try:
            store = PydanticStore(kvstore_id=kvstore_id, model=ImageDescriptionCache)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                # In running async loop, can't easily block, will check file fallback
                pass
            else:
                result = asyncio.run(store.load_object(image_hash))
                if result:
                    return result
        except Exception as exc:
            logger.debug(f"KV store cache lookup skipped for {image_hash}: {exc}")

    # 2. File-based cache fallback
    cache_file = _get_local_cache_file(images_dir)
    if cache_file and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if image_hash in data:
                return ImageDescriptionCache.model_validate(data[image_hash])
        except Exception as exc:
            logger.warning(f"Error reading local image cache {cache_file}: {exc}")

    return None


def save_cached_image_description(
    desc: ImageDescriptionCache,
    images_dir: Path | str | None = None,
    kvstore_id: str | None = None,
) -> None:
    """Save image description to KV store and local sidecar cache."""
    # 1. Try KV store
    if kvstore_id:
        try:
            store = PydanticStore(kvstore_id=kvstore_id, model=ImageDescriptionCache)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if not (loop and loop.is_running()):
                asyncio.run(store.save_obj(desc.image_hash, desc))
        except Exception as exc:
            logger.debug(f"KV store save skipped for {desc.image_hash}: {exc}")

    # 2. Local file cache
    cache_file = _get_local_cache_file(images_dir)
    if cache_file:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            data: dict[str, Any] = {}
            if cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            data[desc.image_hash] = desc.model_dump()
            cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Error writing local image cache {cache_file}: {exc}")


def describe_image_with_vlm(
    image_path: Path,
    doc_title: str = "",
    section_title: str = "",
    vlm_model: str = "glm_5.3_flash@openrouter",
    min_size_bytes: int = 10 * 1024,
    images_dir: Path | str | None = None,
    kvstore_id: str | None = None,
) -> ImageDescriptionCache | None:
    """Describe an uncaptioned image using a lightweight VLM and cache the result.

    Args:
        image_path: Path to the image file.
        doc_title: Contextual document title or filename.
        section_title: Contextual section heading.
        vlm_model: VLM model ID for analysis.
        min_size_bytes: Minimum image size in bytes to trigger description (default 10KB).
        images_dir: Optional directory for sidecar cache file.
        kvstore_id: Optional KV store ID.

    Returns:
        ImageDescriptionCache instance or None if skipped/failed.
    """
    if not image_path.exists():
        return None

    raw_bytes = image_path.read_bytes()
    if len(raw_bytes) < min_size_bytes:
        logger.debug(f"Skipping VLM description for small image {image_path.name} ({len(raw_bytes)} bytes)")
        return None

    img_hash = buffer_digest(raw_bytes, algorithm="xxh32")

    # Check cache first
    cached = load_cached_image_description(img_hash, images_dir=images_dir, kvstore_id=kvstore_id)
    if cached:
        logger.debug(f"Reusing cached description for image {img_hash}")
        return cached

    # Prepare VLM message
    b64_str = base64.b64encode(raw_bytes).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type or not mime_type.startswith("image/"):
        ext = image_path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            mime_type = "image/jpeg"
        elif ext == ".webp":
            mime_type = "image/webp"
        elif ext == ".gif":
            mime_type = "image/gif"
        else:
            mime_type = "image/png"

    data_url = f"data:{mime_type};base64,{b64_str}"

    prompt_lines = [
        "You are an expert visual document and image analyst.",
        "Analyze the provided image and produce a concise 1-3 sentence description.",
        "If the image is a chart, diagram, line plot, bar graph, or technical schematic, extract and summarize the axes, legends, trends, and key data points.",
        "At the end, provide a comma-separated list of keywords on a line starting with 'Keywords:'.",
    ]
    if doc_title:
        prompt_lines.append(f"Document context: {doc_title}")
    if section_title:
        prompt_lines.append(f"Section context: {section_title}")

    system_prompt = "\n".join(prompt_lines)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from genai_tk.core.factories.llm_factory import get_llm

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Describe this image and provide relevant keywords."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            ),
        ]

        llm = get_llm(vlm_model)
        resp = llm.invoke(messages)
        content = str(resp.content).strip()

        # Parse description and keywords
        desc_text = content
        keywords: list[str] = []
        kw_match = re.search(r"Keywords:\s*(.*)$", content, re.IGNORECASE | re.MULTILINE)
        if kw_match:
            kw_str = kw_match.group(1).strip()
            keywords = [k.strip() for k in re.split(r"[,;]", kw_str) if k.strip()]
            desc_text = content[: kw_match.start()].strip()

        result = ImageDescriptionCache(
            image_hash=img_hash,
            description=desc_text,
            keywords=keywords,
            model=vlm_model,
        )

        save_cached_image_description(result, images_dir=images_dir, kvstore_id=kvstore_id)
        return result

    except Exception as exc:
        logger.warning(f"Failed to generate VLM description for {image_path.name}: {exc}")
        return None
