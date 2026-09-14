"""Unit tests for image description caching and loading."""

import pytest

from genai_tk.extra.markdownize.image_describer import (
    ImageDescriptionCache,
    load_cached_image_description,
    save_cached_image_description,
)


@pytest.mark.unit
def test_image_description_cache_serialization():
    desc = ImageDescriptionCache(
        image_hash="abc12345",
        description="A bar chart comparing revenue in 2023 vs 2024.",
        keywords=["revenue", "bar chart", "financials"],
        model="test-vlm",
    )
    assert desc.image_hash == "abc12345"
    assert "revenue" in desc.keywords
    assert desc.model_dump()["description"] == "A bar chart comparing revenue in 2023 vs 2024."


@pytest.mark.unit
def test_local_file_cache_roundtrip(tmp_path):
    desc = ImageDescriptionCache(
        image_hash="deadbeef",
        description="Diagram of neural network architecture.",
        keywords=["transformer", "attention", "layers"],
        model="glm-5.3-flash",
    )
    save_cached_image_description(desc, images_dir=tmp_path)
    loaded = load_cached_image_description("deadbeef", images_dir=tmp_path)
    assert loaded is not None
    assert loaded.image_hash == "deadbeef"
    assert loaded.description == "Diagram of neural network architecture."
    assert loaded.keywords == ["transformer", "attention", "layers"]

    # Missing hash returns None
    assert load_cached_image_description("nonexistent", images_dir=tmp_path) is None
