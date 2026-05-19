"""Cache policy resolution for workflow steps.

The active cache logic lives in
:func:`genai_tk.workflow.prefect.step_factory._cache_policy_for_step` and is
applied automatically when building step tasks.  This module is kept for
backwards compatibility but no longer needs to be used directly.

Supported backends
------------------
- ``none``           → ``NO_CACHE`` (Prefect default for tasks without cache)
- ``prefect_result`` → Prefect's native result-based cache keyed on inputs
- ``manifest``       → Engine-level :class:`~genai_tk.workflow.flow_cache.manifest.ManifestCache`
                       (fingerprint checked in ``flow_factory._build_prefect_flow``)
- ``hybrid``         → ``prefect_result`` Prefect cache + engine-level manifest check
"""

from __future__ import annotations

from genai_tk.workflow.compiled_models import CacheSpec
from genai_tk.workflow.prefect.step_factory import _cache_policy_for_step  # noqa: F401


def get_cache_policy(spec: CacheSpec):
    """Return a Prefect ``cache_policy`` value for the given spec.

    Args:
        spec: Compiled cache specification for a step.

    Returns:
        A Prefect ``CachePolicy`` object.
    """
    from genai_tk.workflow.compiled_models import CompiledStep, InvokeSpec

    stub = CompiledStep(id="_stub", invoke=InvokeSpec(), cache=spec)
    return _cache_policy_for_step(stub)
