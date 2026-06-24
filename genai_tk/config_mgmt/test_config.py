"""Typed test configuration model.

Provides a single-source-of-truth Pydantic model for test settings derived
from the active config profile (typically ``pytest``). Downstream projects
and test suites should use ``get_pytest_config()`` instead of hardcoding
fake model IDs or other test defaults.
"""

from pydantic import BaseModel, Field

from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.core.factories.embeddings_factory import embeddings_config
from genai_tk.core.factories.llm_factory import _llm_section


class PytestConfig(BaseModel):
    """Typed view of test-relevant configuration from the active profile.

    All fields are derived from the already-loaded config singleton
    (``llm``, ``embeddings``, ``llm_cache``, ``vector_store`` sections).
    """

    default_llm: str = Field(description="Default LLM model ID (llm.models.default)")
    fake_llm: str = Field(description="Fake LLM model ID (llm.models.fake)")
    fast_model: str = Field(description="Cheap/fast real LLM for real_models tests (llm.models.fast_model)")
    default_embeddings: str = Field(description="Default embeddings model ID (embeddings.models.default)")
    fake_embeddings: str = Field(description="Fake embeddings model ID (embeddings.models.fake)")
    cache_method: str = Field(default="memory", description="LLM cache method (llm_cache.method)")
    vector_store_backend: str = Field(default="InMemory", description="Default vector store backend")


def get_pytest_config() -> PytestConfig:
    """Build a ``PytestConfig`` from the currently active profile.

    Requires that the config singleton has already been initialized
    (e.g. via ``switch_profile("pytest")`` in a session fixture).
    """
    llm_sec = _llm_section()
    emb_sec = embeddings_config()

    fake_llm = llm_sec.models.get_tag("fake") or llm_sec.models.default
    fast_model = llm_sec.models.get_tag("fast_model") or llm_sec.models.get_tag("cheap_model") or llm_sec.models.default
    fake_emb = emb_sec.models.get_tag("fake") or emb_sec.models.default

    cache_method = global_config().get("llm_cache.method", default="memory")
    vector_store = global_config().get("vector_store.default", default="InMemory")

    return PytestConfig(
        default_llm=llm_sec.models.default,
        fake_llm=fake_llm,
        fast_model=fast_model,
        default_embeddings=emb_sec.models.default,
        fake_embeddings=fake_emb,
        cache_method=str(cache_method),
        vector_store_backend=str(vector_store),
    )
