"""Unit tests for workflow configuration resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_tk.utils.config_mngr import OmegaConfig
from genai_tk.workflow.resolver import (
    WorkflowResolutionError,
    _expand_step_templates,
    _expand_sub_workflows,
    list_step_template_names,
    list_workflow_names,
    list_workflow_profile_names,
    parse_cli_overrides,
    resolve_workflow_invocation,
)


@pytest.fixture
def workflow_config(tmp_path: Path) -> OmegaConfig:
    data_root = tmp_path / "data"
    config_text = f"""
default_config: baseline
paths:
  home: {tmp_path}
  project: {tmp_path}
  config: {tmp_path}
  data_root: {data_root}
baseline:
  workflows:
    ingest_docs:
      description: Ingest a directory of documents
      steps:
        - id: load
          invoke:
            kind: callable
            target: genai_tk.workflow.steps.LoadDocuments
          with:
            root_dir: ${{values.root_dir}}
        - id: ingest
          invoke:
            kind: callable
            target: genai_tk.workflow.steps.IngestDocuments
          wait_for: [load]
          with:
            retriever_name: ${{values.retriever_name}}
    shared:
      steps:
        - id: only
          invoke:
            kind: callable
            target: genai_tk.workflow.steps.Shared
  workflow_profiles:
    docs.default:
      workflow: ingest_docs
      values:
        root_dir: {tmp_path}/docs
        retriever_name: default
    shared:
      workflow: ingest_docs
      values:
        root_dir: {tmp_path}/shared
"""
    config_path = tmp_path / "workflow_config.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    return OmegaConfig.create(config_path)


def test_lists_workflows_and_profiles(workflow_config: OmegaConfig) -> None:
    assert list_workflow_names(workflow_config) == ["ingest_docs", "shared"]
    assert list_workflow_profile_names(workflow_config) == ["docs.default", "shared"]


def test_parse_cli_overrides_supports_nested_values() -> None:
    parsed = parse_cli_overrides(["root_dir=/tmp/docs", "options.batch_size=12", "flags.force=true"])
    assert parsed["root_dir"] == "/tmp/docs"
    assert parsed["options"]["batch_size"] == 12
    assert parsed["flags"]["force"] is True


def test_parse_cli_overrides_rejects_invalid_item() -> None:
    with pytest.raises(WorkflowResolutionError):
        parse_cli_overrides(["missing-separator"])


def test_resolves_profile_name_to_workflow(workflow_config: OmegaConfig) -> None:
    resolved = resolve_workflow_invocation("docs.default", config=workflow_config)
    assert resolved.workflow_name == "ingest_docs"
    assert resolved.profile_name == "docs.default"
    assert resolved.values["retriever_name"] == "default"
    assert len(resolved.workflow.steps) == 2


def test_resolves_workflow_with_explicit_profile_and_cli_override(workflow_config: OmegaConfig) -> None:
    resolved = resolve_workflow_invocation(
        "ingest_docs",
        profile_name="docs.default",
        cli_overrides={"retriever_name": "custom"},
        config=workflow_config,
        force=True,
    )
    assert resolved.profile_name == "docs.default"
    assert resolved.values["root_dir"].endswith("/docs")
    assert resolved.values["retriever_name"] == "custom"
    assert resolved.force is True


def test_ambiguous_name_requires_disambiguation(workflow_config: OmegaConfig) -> None:
    with pytest.raises(WorkflowResolutionError, match="matches both a workflow and a profile"):
        resolve_workflow_invocation("shared", config=workflow_config)


def test_profile_must_match_explicit_workflow(workflow_config: OmegaConfig) -> None:
    with pytest.raises(WorkflowResolutionError, match="targets workflow"):
        resolve_workflow_invocation("shared", profile_name="docs.default", config=workflow_config)


# ---------------------------------------------------------------------------
# Step template expansion tests
# ---------------------------------------------------------------------------


@pytest.fixture
def template_config(tmp_path: Path) -> OmegaConfig:
    data_root = tmp_path / "data"
    config_text = f"""
default_config: baseline
paths:
  home: {tmp_path}
  project: {tmp_path}
  config: {tmp_path}
  data_root: {data_root}
baseline:
  step_templates:
    base_step:
      invoke:
        kind: flow
        target: some.module.base_func
      with:
        base_dir: ${{values.base_dir}}
        output_dir: ${{values.output_dir}}
        batch_size: ${{values.batch_size}}
        verbose: true

  workflows:
    templated_workflow:
      description: Test workflow using step templates
      defaults:
        batch_size: 5
        output_dir: /default/output
      steps:
        - id: step_a
          ref: base_step

        - id: step_b
          ref: base_step
          wait_for: [step_a]
          with:
            output_dir: ${{values.custom_dir}}
            verbose: false

  workflow_profiles:
    my_profile:
      workflow: templated_workflow
      values:
        base_dir: {tmp_path}/input
        custom_dir: {tmp_path}/custom
"""
    config_path = tmp_path / "template_config.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    return OmegaConfig.create(config_path)


def test_list_step_template_names(template_config: OmegaConfig) -> None:
    assert list_step_template_names(template_config) == ["base_step"]


def test_expand_step_templates_resolves_ref() -> None:
    templates = {
        "my_template": {
            "invoke": {"kind": "flow", "target": "module.func"},
            "with": {"root": "${values.root}", "batch": 10},
        }
    }
    steps = [{"id": "step1", "ref": "my_template"}]
    expanded = _expand_step_templates(steps, templates)
    assert len(expanded) == 1
    assert expanded[0]["invoke"]["target"] == "module.func"
    assert expanded[0]["with"] == {"root": "${values.root}", "batch": 10}
    assert "ref" not in expanded[0]


def test_expand_step_templates_step_overrides_template() -> None:
    templates = {
        "base": {
            "invoke": {"kind": "flow", "target": "module.base"},
            "with": {"dir": "/default", "extra": "keep", "verbose": True},
        }
    }
    steps = [
        {
            "id": "step1",
            "ref": "base",
            "invoke": {"kind": "flow", "target": "module.override"},
            "with": {"dir": "/custom", "verbose": False},
        }
    ]
    expanded = _expand_step_templates(steps, templates)
    assert expanded[0]["invoke"]["target"] == "module.override"
    # with dict is merged: step wins at key level, template keys are kept
    assert expanded[0]["with"] == {"dir": "/custom", "extra": "keep", "verbose": False}


def test_expand_step_templates_unknown_ref_raises() -> None:
    with pytest.raises(WorkflowResolutionError, match="unknown step template 'missing'"):
        _expand_step_templates([{"id": "s", "ref": "missing"}], {})


def test_expand_step_templates_passthrough_without_ref() -> None:
    steps = [{"id": "s", "invoke": {"kind": "flow", "target": "module.func"}}]
    expanded = _expand_step_templates(steps, {})
    assert expanded == steps


def test_load_workflow_spec_expands_templates(template_config: OmegaConfig) -> None:
    from genai_tk.workflow.resolver import load_workflow_spec

    spec = load_workflow_spec("templated_workflow", template_config)
    assert len(spec.steps) == 2

    step_a = spec.steps[0]
    assert step_a.id == "step_a"
    assert step_a.invoke.target == "some.module.base_func"
    assert step_a.wait_for == []

    step_b = spec.steps[1]
    assert step_b.id == "step_b"
    assert step_b.invoke.target == "some.module.base_func"
    assert step_b.wait_for == ["step_a"]
    # step_b overrides output_dir but keeps base_dir from template
    assert step_b.with_["output_dir"] == "${values.custom_dir}"
    assert step_b.with_["base_dir"] == "${values.base_dir}"
    # step_b overrides verbose=false
    assert step_b.with_["verbose"] is False


def test_workflow_defaults_applied_when_profile_omits_values(template_config: OmegaConfig) -> None:
    resolved = resolve_workflow_invocation("my_profile", config=template_config)
    # batch_size and output_dir come from workflow defaults since profile doesn't set them
    assert resolved.values["batch_size"] == 5
    assert resolved.values["output_dir"] == "/default/output"


def test_workflow_defaults_overridden_by_profile_values(template_config: OmegaConfig) -> None:
    resolved = resolve_workflow_invocation(
        "my_profile",
        cli_overrides={"batch_size": 99},
        config=template_config,
    )
    assert resolved.values["batch_size"] == 99


def test_workflow_defaults_overridden_by_cli(template_config: OmegaConfig) -> None:
    resolved = resolve_workflow_invocation(
        "my_profile",
        cli_overrides={"output_dir": "/cli/output"},
        config=template_config,
    )
    assert resolved.values["output_dir"] == "/cli/output"


# ---------------------------------------------------------------------------
# Sub-workflow expansion tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sub_workflow_config(tmp_path: Path) -> OmegaConfig:
    data_root = tmp_path / "data"
    config_text = f"""
default_config: baseline
paths:
  home: {tmp_path}
  project: {tmp_path}
  config: {tmp_path}
  data_root: {data_root}
baseline:
  step_templates:
    build_step:
      invoke:
        kind: callable
        target: graph.build
      with:
        graphs: ${{values.graphs}}

  workflows:
    base_graph:
      steps:
        - id: build
          ref: build_step

    extended_graph:
      steps:
        - id: build
          ref: build_step
        - id: extra
          invoke:
            kind: callable
            target: graph.extra
          wait_for: [build]

    composite:
      steps:
        - id: base
          invoke:
            kind: workflow
            target: base_graph
        - id: ext
          invoke:
            kind: workflow
            target: extended_graph
          wait_for: [base]

    deep_composite:
      steps:
        - id: prep
          invoke:
            kind: callable
            target: prep.step
        - id: graphs
          invoke:
            kind: workflow
            target: composite
          wait_for: [prep]

    cyclic_a:
      steps:
        - id: s
          invoke:
            kind: workflow
            target: cyclic_b

    cyclic_b:
      steps:
        - id: s
          invoke:
            kind: workflow
            target: cyclic_a

  workflow_profiles:
    run_composite:
      workflow: composite
      values:
        graphs: []
"""
    config_path = tmp_path / "sub_wf_config.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    return OmegaConfig.create(config_path)


def test_sub_workflow_expansion_basic(sub_workflow_config: OmegaConfig) -> None:
    from genai_tk.workflow.resolver import load_workflow_spec

    spec = load_workflow_spec("composite", sub_workflow_config)
    step_ids = [s.id for s in spec.steps]
    # base_graph.build expanded as "base.build"
    # extended_graph.build expanded as "ext.build", extended_graph.extra as "ext.extra"
    assert "base.build" in step_ids
    assert "ext.build" in step_ids
    assert "ext.extra" in step_ids
    assert len(spec.steps) == 3


def test_sub_workflow_needs_wiring(sub_workflow_config: OmegaConfig) -> None:
    from genai_tk.workflow.resolver import load_workflow_spec

    spec = load_workflow_spec("composite", sub_workflow_config)
    steps_by_id = {s.id: s for s in spec.steps}

    # base.build is a root step of base_graph, composite says base has no wait_for
    assert steps_by_id["base.build"].wait_for == []

    # ext.build is a root step of extended_graph; composite says ext wait_for [base]
    # After terminal resolution, "base" is replaced with its terminal step(s)
    assert "base.build" in steps_by_id["ext.build"].wait_for

    # ext.extra has internal wait_for=[build] → prefixed to [ext.build]
    assert steps_by_id["ext.extra"].wait_for == ["ext.build"]


def test_sub_workflow_deep_expansion(sub_workflow_config: OmegaConfig) -> None:
    from genai_tk.workflow.resolver import load_workflow_spec

    spec = load_workflow_spec("deep_composite", sub_workflow_config)
    step_ids = [s.id for s in spec.steps]
    assert "prep" in step_ids
    assert "graphs.base.build" in step_ids
    assert "graphs.ext.build" in step_ids
    assert "graphs.ext.extra" in step_ids
    assert len(spec.steps) == 4


def test_sub_workflow_deep_needs(sub_workflow_config: OmegaConfig) -> None:
    from genai_tk.workflow.resolver import load_workflow_spec

    spec = load_workflow_spec("deep_composite", sub_workflow_config)
    steps_by_id = {s.id: s for s in spec.steps}
    assert "prep" in steps_by_id["graphs.base.build"].wait_for
    assert "graphs.base.build" in steps_by_id["graphs.ext.build"].wait_for


def test_sub_workflow_cycle_detection(sub_workflow_config: OmegaConfig) -> None:
    from genai_tk.workflow.resolver import load_workflow_spec

    with pytest.raises(WorkflowResolutionError, match="Cycle detected"):
        load_workflow_spec("cyclic_a", sub_workflow_config)


def test_sub_workflow_unknown_workflow_raises(sub_workflow_config: OmegaConfig) -> None:
    steps = [{"id": "s", "invoke": {"kind": "workflow", "target": "nonexistent"}}]
    with pytest.raises(WorkflowResolutionError, match="unknown workflow 'nonexistent'"):
        _expand_sub_workflows(steps, sub_workflow_config)


def test_sub_workflow_terminal_resolution_with_multi_terminal(tmp_path: Path) -> None:
    """When a sub-workflow has multiple terminal steps, wait_for the parent expands to all terminals."""
    data_root = tmp_path / "data"
    config_text = f"""
default_config: baseline
paths:
  home: {tmp_path}
  project: {tmp_path}
  config: {tmp_path}
  data_root: {data_root}
baseline:
  workflows:
    two_leaves:
      steps:
        - id: root
          invoke:
            kind: callable
            target: step.root
        - id: leaf_a
          invoke:
            kind: callable
            target: step.leaf_a
          wait_for: [root]
        - id: leaf_b
          invoke:
            kind: callable
            target: step.leaf_b
          wait_for: [root]
    after_leaves:
      steps:
        - id: base
          invoke:
            kind: workflow
            target: two_leaves
        - id: final
          invoke:
            kind: callable
            target: step.final
          wait_for: [base]
"""
    config_path = tmp_path / "multi_terminal.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    cfg = OmegaConfig.create(config_path)

    from genai_tk.workflow.resolver import load_workflow_spec

    spec = load_workflow_spec("after_leaves", cfg)
    steps_by_id = {s.id: s for s in spec.steps}
    # "final" wait_for [base] → base has two terminals: leaf_a and leaf_b
    assert "base.leaf_a" in steps_by_id["final"].wait_for
    assert "base.leaf_b" in steps_by_id["final"].wait_for
    assert "base" not in steps_by_id["final"].wait_for
