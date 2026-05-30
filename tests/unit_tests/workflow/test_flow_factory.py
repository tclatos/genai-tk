"""Unit tests for PrefectFlowFactory: DAG construction, manifest caching, parallel submission."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from genai_tk.workflow.compiled_models import (
    CacheSpec,
    CompiledStep,
    CompiledWorkflow,
    ExecutionSpec,
    InvokeSpec,
    StepKind,
)
from genai_tk.workflow.prefect.flow_factory import (
    WorkflowExecutionError,
    _prepare_inputs,
    _resolve_step_ref,
    compute_step_fingerprint,
    workflow_manifest_path,
)

# ---------------------------------------------------------------------------
# Helpers to build compiled objects without going through config_mngr
# ---------------------------------------------------------------------------


def _step(
    step_id: str,
    target: str = "json.dumps",
    wait_for: list[str] | None = None,
    cache_backend: str = "none",
    on_failure: str = "abort",
    with_: dict[str, Any] | None = None,
) -> CompiledStep:
    return CompiledStep(
        id=step_id,
        invoke=InvokeSpec(kind=StepKind.callable, target=target),
        wait_for=wait_for or [],
        cache=CacheSpec(backend=cache_backend),
        execution=ExecutionSpec(on_failure=on_failure),
        **{"with": with_ or {}},
    )


def _workflow(name: str, steps: list[CompiledStep], values: dict | None = None) -> CompiledWorkflow:
    return CompiledWorkflow(name=name, steps=steps, values=values or {})


# ---------------------------------------------------------------------------
# _prepare_inputs
# ---------------------------------------------------------------------------


class TestPrepareInputs:
    def test_passthrough_plain_values(self) -> None:
        result = _prepare_inputs({"x": 1, "y": "hello"}, {})
        assert result == {"x": 1, "y": "hello"}

    def test_strips_none_values(self) -> None:
        result = _prepare_inputs({"x": None, "y": 2}, {})
        assert result == {"y": 2}

    def test_resolves_step_ref(self) -> None:
        results = {"prev": {"count": 7}}
        result = _prepare_inputs({"n": "${steps.prev.result.count}"}, results)
        assert result == {"n": 7}

    def test_unresolvable_step_ref_strips_key(self) -> None:
        result = _prepare_inputs({"n": "${steps.missing.result.count}"}, {})
        assert "n" not in result


# ---------------------------------------------------------------------------
# _resolve_step_ref
# ---------------------------------------------------------------------------


class TestResolveStepRef:
    def test_passthrough_non_string(self) -> None:
        assert _resolve_step_ref(42, {}) == 42
        assert _resolve_step_ref([1, 2], {}) == [1, 2]

    def test_passthrough_non_ref_string(self) -> None:
        assert _resolve_step_ref("hello", {}) == "hello"

    def test_resolves_dict_result(self) -> None:
        results = {"s1": {"x": 99}}
        assert _resolve_step_ref("${steps.s1.result.x}", results) == 99

    def test_resolves_object_attribute(self) -> None:
        obj = MagicMock()
        obj.score = 0.9
        results = {"s1": obj}
        assert _resolve_step_ref("${steps.s1.result.score}", results) == 0.9

    def test_missing_step_returns_none(self) -> None:
        assert _resolve_step_ref("${steps.gone.result.x}", {}) is None


# ---------------------------------------------------------------------------
# workflow_manifest_path
# ---------------------------------------------------------------------------


class TestWorkflowManifestPath:
    def test_returns_path_under_data_root(self, tmp_path) -> None:
        with patch("genai_tk.utils.config_mngr.global_config") as mock_cfg:
            mock_cfg.return_value.paths.data_root = str(tmp_path)
            path = workflow_manifest_path("my_workflow")

        assert path.name == "manifest.json"
        assert "my_workflow" in str(path)
        assert str(tmp_path) in str(path)

    def test_fallback_to_home_cache_on_error(self) -> None:
        with patch(
            "genai_tk.utils.config_mngr.global_config",
            side_effect=RuntimeError("no config"),
        ):
            path = workflow_manifest_path("wf")
        assert ".cache" in str(path) or str(path).startswith("/")
        assert "wf" in str(path)


# ---------------------------------------------------------------------------
# PrefectFlowFactory — flow body (unit-tested via flow.fn)
# ---------------------------------------------------------------------------


class _MockFuture:
    """Fake Prefect future for unit tests (no task runner needed)."""

    def __init__(self, return_value: Any = None, exception: BaseException | None = None) -> None:
        self._value = return_value
        self._exc = exception

    def result(self) -> Any:
        if self._exc is not None:
            raise self._exc
        return self._value


class _MockTask:
    """Fake Prefect task that synchronously executes its callable."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn

    def submit(self, *args: Any, wait_for: Any = None, **kwargs: Any) -> _MockFuture:
        try:
            return _MockFuture(return_value=self._fn(*args, **kwargs))
        except Exception as exc:
            return _MockFuture(exception=exc)


def _patch_step_factory(callable_fn: Any) -> Any:
    """Return a context manager that patches PrefectStepFactory.create → _MockTask."""
    return patch(
        "genai_tk.workflow.prefect.step_factory.PrefectStepFactory.create",
        return_value=_MockTask(callable_fn),
    )


class TestFlowBodyCaching:
    """Test the manifest caching logic in _build_prefect_flow via .fn()."""

    def _build_flow_fn(self, workflow: CompiledWorkflow) -> Any:
        from genai_tk.workflow.prefect.flow_factory import _build_prefect_flow

        return _build_prefect_flow(workflow)

    @patch("genai_tk.workflow.prefect.flow_factory.workflow_manifest_path")
    def test_cached_step_is_skipped(self, mock_manifest_path, tmp_path) -> None:
        """A step with a fresh manifest entry must not be submitted."""
        from pathlib import Path

        from genai_tk.workflow.flow_cache.manifest import ManifestCache

        manifest_path = Path(tmp_path / "manifest.json")
        mock_manifest_path.return_value = manifest_path

        # Pre-populate the manifest with a fresh entry
        cache = ManifestCache()
        step_inputs = {}  # matches what the step will receive (empty with_)
        fp = compute_step_fingerprint("build", step_inputs)
        cache.record_success("build", fp, outputs={"result": {"cached": True}})
        cache.save(manifest_path)

        # Build a single-step workflow with manifest caching
        step = _step("build", cache_backend="manifest")
        workflow = _workflow("test_wf", [step])

        spy_callable = MagicMock(return_value={"fresh": False})
        with _patch_step_factory(spy_callable):
            flow_fn = self._build_flow_fn(workflow)
            results = flow_fn.fn()

        # Step callable must NOT have been called (cached)
        spy_callable.assert_not_called()
        assert results["build"] == {"cached": True}

    @patch("genai_tk.workflow.prefect.flow_factory.workflow_manifest_path")
    def test_stale_step_is_executed_and_recorded(self, mock_manifest_path, tmp_path) -> None:
        """A step with a stale manifest entry must be executed and re-recorded."""
        from pathlib import Path

        from genai_tk.workflow.flow_cache.manifest import ManifestCache

        manifest_path = Path(tmp_path / "manifest2.json")
        mock_manifest_path.return_value = manifest_path

        # Pre-populate with STALE fingerprint
        cache = ManifestCache()
        cache.record_success("build", "old_fingerprint", outputs={"result": {"old": True}})
        cache.save(manifest_path)

        step = _step("build", cache_backend="manifest")
        workflow = _workflow("test_wf", [step])

        fresh_result = {"new": True}
        spy_callable = MagicMock(return_value=fresh_result)
        with _patch_step_factory(spy_callable):
            flow_fn = self._build_flow_fn(workflow)
            results = flow_fn.fn()

        spy_callable.assert_called_once()
        assert results["build"] == fresh_result

        # Manifest must have been updated
        reloaded = ManifestCache.load(manifest_path)
        step_inputs = {}
        new_fp = compute_step_fingerprint("build", step_inputs)
        assert reloaded.is_fresh("build", fingerprint=new_fp) is True

    @patch("genai_tk.workflow.prefect.flow_factory.workflow_manifest_path")
    def test_force_bypasses_cache(self, mock_manifest_path, tmp_path) -> None:
        """When force=True, even a fresh step must be re-executed."""
        from pathlib import Path

        from genai_tk.workflow.flow_cache.manifest import ManifestCache

        manifest_path = Path(tmp_path / "manifest3.json")
        mock_manifest_path.return_value = manifest_path

        cache = ManifestCache()
        fp = compute_step_fingerprint("build", {})
        cache.record_success("build", fp, outputs={"result": {"cached": True}})
        cache.save(manifest_path)

        step = _step("build", cache_backend="manifest")
        # force_rebuild=True in values → engine bypasses cache
        workflow = _workflow("test_wf", [step], values={"force_rebuild": True})

        spy_callable = MagicMock(return_value={"rebuilt": True})
        with _patch_step_factory(spy_callable):
            flow_fn = self._build_flow_fn(workflow)
            results = flow_fn.fn()

        spy_callable.assert_called_once()
        assert results["build"] == {"rebuilt": True}

    @patch("genai_tk.workflow.prefect.flow_factory.workflow_manifest_path")
    def test_no_cache_backend_always_executes(self, mock_manifest_path, tmp_path) -> None:
        """Steps with cache backend 'none' are always executed."""
        from pathlib import Path

        manifest_path = Path(tmp_path / "manifest4.json")
        mock_manifest_path.return_value = manifest_path

        step = _step("run", cache_backend="none")
        workflow = _workflow("wf", [step])

        spy_callable = MagicMock(return_value=42)
        with _patch_step_factory(spy_callable):
            flow_fn = self._build_flow_fn(workflow)
            results = flow_fn.fn()

        spy_callable.assert_called_once()
        assert results["run"] == 42
        # No manifest written for 'none' backend
        assert not manifest_path.exists()

    @patch("genai_tk.workflow.prefect.flow_factory.workflow_manifest_path")
    def test_step_failure_with_abort_raises(self, mock_manifest_path, tmp_path) -> None:
        from pathlib import Path

        mock_manifest_path.return_value = Path(tmp_path / "mf.json")

        step = _step("bad", on_failure="abort")
        workflow = _workflow("wf", [step])

        with _patch_step_factory(MagicMock(side_effect=RuntimeError("boom"))):
            flow_fn = self._build_flow_fn(workflow)
            with pytest.raises(WorkflowExecutionError, match="boom"):
                flow_fn.fn()

    @patch("genai_tk.workflow.prefect.flow_factory.workflow_manifest_path")
    def test_step_failure_with_skip_continues(self, mock_manifest_path, tmp_path) -> None:
        from pathlib import Path

        mock_manifest_path.return_value = Path(tmp_path / "mf.json")

        step = _step("soft", on_failure="skip")
        workflow = _workflow("wf", [step])

        with _patch_step_factory(MagicMock(side_effect=RuntimeError("oops"))):
            flow_fn = self._build_flow_fn(workflow)
            results = flow_fn.fn()

        assert results["soft"] is None
