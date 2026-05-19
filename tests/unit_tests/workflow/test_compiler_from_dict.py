"""Unit tests for WorkflowCompiler using raw Python dicts — no config manager required.

All tests create WorkflowSpec / CompiledWorkflow objects directly from dicts,
which makes them fast, isolated, and independent of OmegaConf config files.
"""

from __future__ import annotations

import pytest

from genai_tk.workflow.compiled_models import CacheSpec, CompiledStep, CompiledWorkflow, InvokeSpec
from genai_tk.workflow.compiler import WorkflowCompilationError, WorkflowCompiler
from genai_tk.workflow.models import WorkflowSpec

# ---------------------------------------------------------------------------
# Helpers: build WorkflowSpec from dicts without OmegaConf
# ---------------------------------------------------------------------------


def _spec_from_dict(d: dict) -> WorkflowSpec:
    """Build a WorkflowSpec from a plain dict (mirrors YAML structure)."""
    return WorkflowSpec.model_validate(d)


def _compile(spec_dict: dict, values: dict | None = None) -> CompiledWorkflow:
    spec = _spec_from_dict(spec_dict)
    return WorkflowCompiler().compile(spec, values or {})


# ---------------------------------------------------------------------------
# Basic compilation
# ---------------------------------------------------------------------------


class TestBasicCompilation:
    def test_single_step_no_deps(self) -> None:
        compiled = _compile(
            {
                "name": "simple",
                "steps": [{"id": "load", "invoke": {"kind": "callable", "target": "json.dumps"}}],
            }
        )
        assert compiled.name == "simple"
        assert len(compiled.steps) == 1
        assert compiled.steps[0].id == "load"

    def test_values_interpolated_in_with(self) -> None:
        compiled = _compile(
            {
                "name": "wf",
                "steps": [
                    {
                        "id": "run",
                        "invoke": {"kind": "callable", "target": "json.dumps"},
                        "with": {"path": "${values.root_dir}", "batch": 5},
                    }
                ],
            },
            values={"root_dir": "/tmp/docs"},
        )
        assert compiled.steps[0].with_["path"] == "/tmp/docs"
        assert compiled.steps[0].with_["batch"] == 5

    def test_unresolvable_value_placeholder_is_stripped(self) -> None:
        compiled = _compile(
            {
                "name": "wf",
                "steps": [
                    {
                        "id": "run",
                        "invoke": {"kind": "callable", "target": "json.dumps"},
                        "with": {"path": "${values.missing_key}", "count": 3},
                    }
                ],
            },
            values={},
        )
        # Unresolvable ${values.*} resolves to None in the compiled model.
        assert compiled.steps[0].with_.get("path") is None
        # Values with concrete values are kept.
        assert compiled.steps[0].with_["count"] == 3

    def test_wait_for_wired_correctly(self) -> None:
        compiled = _compile(
            {
                "name": "pipeline",
                "steps": [
                    {"id": "a", "invoke": {"target": "json.dumps"}},
                    {"id": "b", "invoke": {"target": "json.dumps"}, "wait_for": ["a"]},
                ],
            }
        )
        assert compiled.steps[1].wait_for == ["a"]

    def test_cache_spec_parsed(self) -> None:
        compiled = _compile(
            {
                "name": "wf",
                "steps": [
                    {
                        "id": "build",
                        "invoke": {"target": "json.dumps"},
                        "cache": {"backend": "manifest"},
                    }
                ],
            }
        )
        assert compiled.steps[0].cache.backend == "manifest"

    def test_cache_spec_level_field_removed(self) -> None:
        """CacheSpec.level was removed; the field must not exist."""
        spec = CacheSpec(backend="manifest")
        assert not hasattr(spec, "level")


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_duplicate_step_ids_raise(self) -> None:
        with pytest.raises(WorkflowCompilationError, match="[Dd]uplicate"):
            _compile(
                {
                    "name": "bad",
                    "steps": [
                        {"id": "x", "invoke": {"target": "json.dumps"}},
                        {"id": "x", "invoke": {"target": "json.dumps"}},
                    ],
                }
            )

    def test_unknown_wait_for_ref_raises(self) -> None:
        with pytest.raises(WorkflowCompilationError, match="unknown"):
            _compile(
                {
                    "name": "bad",
                    "steps": [
                        {"id": "b", "invoke": {"target": "json.dumps"}, "wait_for": ["nonexistent"]},
                    ],
                }
            )

    def test_cyclic_dependency_raises(self) -> None:
        with pytest.raises(WorkflowCompilationError):
            _compile(
                {
                    "name": "bad",
                    "steps": [
                        {"id": "a", "invoke": {"target": "json.dumps"}, "wait_for": ["b"]},
                        {"id": "b", "invoke": {"target": "json.dumps"}, "wait_for": ["a"]},
                    ],
                }
            )


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    def test_linear_chain_order(self) -> None:
        from genai_tk.workflow.compiler import topological_sort

        steps = [
            CompiledStep(id="c", invoke=InvokeSpec(), wait_for=["b"]),
            CompiledStep(id="a", invoke=InvokeSpec()),
            CompiledStep(id="b", invoke=InvokeSpec(), wait_for=["a"]),
        ]
        ordered = topological_sort(steps)
        ids = [s.id for s in ordered]
        assert ids.index("a") < ids.index("b") < ids.index("c")

    def test_parallel_steps_both_present(self) -> None:
        from genai_tk.workflow.compiler import topological_sort

        steps = [
            CompiledStep(id="a", invoke=InvokeSpec()),
            CompiledStep(id="b", invoke=InvokeSpec()),
            CompiledStep(id="c", invoke=InvokeSpec(), wait_for=["a", "b"]),
        ]
        ordered = topological_sort(steps)
        ids = [s.id for s in ordered]
        assert ids.index("a") < ids.index("c")
        assert ids.index("b") < ids.index("c")


# ---------------------------------------------------------------------------
# Defaults merging
# ---------------------------------------------------------------------------


class TestDefaultsMerging:
    def test_explicit_values_passed_to_compiler_are_interpolated(self) -> None:
        """The compiler interpolates values passed to it (defaults are merged by the resolver)."""
        compiled = _compile(
            {
                "name": "wf",
                "steps": [
                    {
                        "id": "run",
                        "invoke": {"target": "json.dumps"},
                        "with": {"batch": "${values.batch_size}"},
                    }
                ],
            },
            values={"batch_size": 10},  # caller passes merged values
        )
        assert compiled.steps[0].with_["batch"] == 10

    def test_explicit_values_override_defaults_when_pre_merged(self) -> None:
        compiled = _compile(
            {
                "name": "wf",
                "steps": [
                    {
                        "id": "run",
                        "invoke": {"target": "json.dumps"},
                        "with": {"batch": "${values.batch_size}"},
                    }
                ],
            },
            values={"batch_size": 99},
        )
        assert compiled.steps[0].with_["batch"] == 99
