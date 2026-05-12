"""Unit tests for workflow configuration resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_tk.utils.config_mngr import OmegaConfig
from genai_tk.workflow.resolver import (
    WorkflowResolutionError,
    _expand_step_templates,
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
          uses: genai_tk.workflow.steps.LoadDocuments
          inputs:
            root_dir: ${{profile.root_dir}}
          outputs:
            documents: loaded_docs
        - id: ingest
          uses: genai_tk.workflow.steps.IngestDocuments
          needs: [load]
          params:
            retriever_name: ${{profile.retriever_name}}
    shared:
      steps:
        - id: only
          uses: genai_tk.workflow.steps.Shared
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
      uses: some.module.base_func
      inputs:
        base_dir: ${{profile.base_dir}}
        output_dir: ${{profile.output_dir}}
      params:
        batch_size: ${{profile.batch_size}}
        verbose: true
      concurrency: serial

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
          needs: [step_a]
          inputs:
            output_dir: ${{profile.custom_dir}}
          params:
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
            "uses": "module.func",
            "inputs": {"root": "${profile.root}"},
            "params": {"batch": 10},
            "concurrency": "serial",
        }
    }
    steps = [{"id": "step1", "ref": "my_template"}]
    expanded = _expand_step_templates(steps, templates)
    assert len(expanded) == 1
    assert expanded[0]["uses"] == "module.func"
    assert expanded[0]["inputs"] == {"root": "${profile.root}"}
    assert expanded[0]["params"] == {"batch": 10}
    assert expanded[0]["concurrency"] == "serial"
    assert "ref" not in expanded[0]


def test_expand_step_templates_step_overrides_template() -> None:
    templates = {
        "base": {
            "uses": "module.base",
            "inputs": {"dir": "/default", "extra": "keep"},
            "params": {"verbose": True},
            "concurrency": "serial",
        }
    }
    steps = [
        {
            "id": "step1",
            "ref": "base",
            "uses": "module.override",
            "inputs": {"dir": "/custom"},
            "params": {"verbose": False},
        }
    ]
    expanded = _expand_step_templates(steps, templates)
    assert expanded[0]["uses"] == "module.override"
    # dict fields are merged: step wins at key level, template keys are kept
    assert expanded[0]["inputs"] == {"dir": "/custom", "extra": "keep"}
    assert expanded[0]["params"] == {"verbose": False}


def test_expand_step_templates_unknown_ref_raises() -> None:
    with pytest.raises(WorkflowResolutionError, match="unknown step template 'missing'"):
        _expand_step_templates([{"id": "s", "ref": "missing"}], {})


def test_expand_step_templates_passthrough_without_ref() -> None:
    steps = [{"id": "s", "uses": "module.func"}]
    expanded = _expand_step_templates(steps, {})
    assert expanded == steps


def test_load_workflow_spec_expands_templates(template_config: OmegaConfig) -> None:
    from genai_tk.workflow.resolver import load_workflow_spec

    spec = load_workflow_spec("templated_workflow", template_config)
    assert len(spec.steps) == 2

    step_a = spec.steps[0]
    assert step_a.id == "step_a"
    assert step_a.uses == "some.module.base_func"
    assert step_a.concurrency == "serial"
    assert step_a.needs == []

    step_b = spec.steps[1]
    assert step_b.id == "step_b"
    assert step_b.uses == "some.module.base_func"
    assert step_b.needs == ["step_a"]
    # step_b overrides output_dir but keeps base_dir from template
    assert step_b.inputs["output_dir"] == "${profile.custom_dir}"
    assert step_b.inputs["base_dir"] == "${profile.base_dir}"
    # step_b overrides verbose=false
    assert step_b.params["verbose"] is False


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
