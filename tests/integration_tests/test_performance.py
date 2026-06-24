"""Performance benchmarks for fake model workflows.

Run with: ``uv run pytest -m performance_tests``
"""

import time

import pytest

from genai_tk.core.factories.embeddings_factory import get_embeddings
from genai_tk.core.factories.llm_factory import get_llm


@pytest.mark.integration
@pytest.mark.fake_models
@pytest.mark.performance_tests
def test_fake_embeddings_batch_throughput(fake_embeddings_id, performance_threshold) -> None:
    """Fake embeddings model embeds 10 texts in under 2 seconds."""
    embeddings = get_embeddings(embeddings=fake_embeddings_id)

    start = time.monotonic()
    vectors = embeddings.embed_documents(["test text"] * 10)
    elapsed = time.monotonic() - start

    assert len(vectors) == 10
    assert elapsed < 2.0


@pytest.mark.integration
@pytest.mark.fake_models
@pytest.mark.performance_tests
def test_fake_llm_sequential_throughput(fake_llm_id, performance_threshold) -> None:
    """Fake LLM generates 5 sequential responses in under 3 seconds."""
    llm = get_llm(llm=fake_llm_id)

    start = time.monotonic()
    responses = [llm.invoke(f"Test prompt {i}") for i in range(5)]
    elapsed = time.monotonic() - start

    assert len(responses) == 5
    assert elapsed < 3.0
