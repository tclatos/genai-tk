"""CacheBackendFactory: builds the right Prefect cache policy for a step.

Supported backends
------------------
- ``none``           → ``NO_CACHE`` (Prefect default for tasks without cache)
- ``prefect_result`` → Prefect's native result-based cache keyed on inputs
- ``manifest``       → :class:`~genai_tk.workflow.cache.manifest.ManifestCache`
                       (file-based incremental processing, shared across flows)
- ``hybrid``         → ``prefect_result`` for the step return value **plus**
                       :class:`ManifestCache` for per-item artifact tracking
"""

from __future__ import annotations

from pydantic import BaseModel

from genai_tk.workflow.compiled_models import CacheSpec


class CacheBackendFactory(BaseModel):
    """Resolve a :class:`CacheSpec` into the appropriate Prefect cache policy object.

    Example:
        ```python
        from genai_tk.workflow.prefect.cache_factory import CacheBackendFactory
        from genai_tk.workflow.compiled_models import CacheSpec

        spec = CacheSpec(backend="prefect_result", level="step")
        cache_policy = CacheBackendFactory().get(spec)
        ```
    """

    def get(self, spec: CacheSpec):
        """Return a Prefect ``cache_policy`` value for the given spec.

        Args:
            spec: Compiled cache specification for the step.

        Returns:
            A Prefect ``CachePolicy`` object, or ``None`` for the ``none`` backend.
        """
        if spec.backend == "none":
            return _no_cache()
        if spec.backend == "prefect_result":
            return _prefect_result_cache(spec)
        if spec.backend == "manifest":
            # Manifest caching is handled inside the step function itself via
            # ManifestCache; no Prefect-level cache policy is needed.
            return _no_cache()
        if spec.backend == "hybrid":
            # Use Prefect result cache for the step return value.
            return _prefect_result_cache(spec)
        return _no_cache()


def _no_cache():
    from prefect.cache_policies import NO_CACHE

    return NO_CACHE


def _prefect_result_cache(spec: CacheSpec):
    from prefect.cache_policies import INPUTS

    # INPUTS policy: Prefect caches task results when the input parameters
    # hash matches a previous run.
    return INPUTS
