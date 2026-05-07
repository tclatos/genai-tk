"""Unit tests for workflow configuration resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_tk.utils.config_mngr import OmegaConfig
from genai_tk.workflow.resolver import (
    WorkflowResolutionError,
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
