"""Tests for RAG CLI parameter resolution helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_tk.extra.rag.commands_rag import _resolve_add_files_params
from genai_tk.utils.config_mngr import OmegaConfig
from genai_tk.workflow.resolver import WorkflowResolutionError


@pytest.fixture
def rag_workflow_config(tmp_path: Path) -> OmegaConfig:
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
    rag_add_files:
      steps:
        - id: ingest
          uses: genai_tk.workflow.steps.rag.AddFiles
    other_workflow:
      steps:
        - id: noop
          uses: genai_tk.workflow.steps.misc.Noop
  workflow_profiles:
    rag.docs:
      workflow: rag_add_files
      values:
        root_dir: {tmp_path}/docs
        retriever_name: docs_retriever
        batch_size: 4
        chunk_size: 256
        chunker: markdown
        recursive: true
        include_patterns: ['*.md']
        exclude_patterns: ['*_draft.md']
    wrong.profile:
      workflow: other_workflow
      values:
        root_dir: {tmp_path}/wrong
"""
    config_path = tmp_path / "rag_workflow.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    return OmegaConfig.create(config_path)


def test_resolve_add_files_params_legacy_mode() -> None:
    params, invocation = _resolve_add_files_params(
        root_dir="./docs",
        retriever_name=None,
        include=None,
        exclude=None,
        recursive=None,
        force=False,
        batch_size=None,
        chunk_size=None,
        chunker=None,
        workflow_config=None,
    )
    assert invocation is None
    assert params["root_dir"] == "./docs"
    assert params["retriever_name"] == "default"
    assert params["include"] == ["**/*"]
    assert params["batch_size"] == 10


def test_resolve_add_files_params_from_profile(rag_workflow_config: OmegaConfig) -> None:
    params, invocation = _resolve_add_files_params(
        root_dir=None,
        retriever_name=None,
        include=None,
        exclude=None,
        recursive=None,
        force=True,
        batch_size=None,
        chunk_size=None,
        chunker=None,
        workflow_config="rag.docs",
        config=rag_workflow_config,
    )
    assert invocation is not None
    assert invocation.workflow_name == "rag_add_files"
    assert params["root_dir"].endswith("/docs")
    assert params["retriever_name"] == "docs_retriever"
    assert params["batch_size"] == 4
    assert params["chunk_size"] == 256
    assert params["force"] is True


def test_resolve_add_files_params_profile_with_cli_overrides(rag_workflow_config: OmegaConfig) -> None:
    params, _ = _resolve_add_files_params(
        root_dir="/override/docs",
        retriever_name="override_retriever",
        include=["*.txt"],
        exclude=None,
        recursive=False,
        force=False,
        batch_size=9,
        chunk_size=777,
        chunker="recursive",
        workflow_config="rag.docs",
        config=rag_workflow_config,
    )
    assert params["root_dir"] == "/override/docs"
    assert params["retriever_name"] == "override_retriever"
    assert params["include"] == ["*.txt"]
    assert params["recursive"] is False
    assert params["batch_size"] == 9
    assert params["chunk_size"] == 777
    assert params["chunker"] == "recursive"


def test_resolve_add_files_rejects_wrong_workflow(rag_workflow_config: OmegaConfig) -> None:
    with pytest.raises(WorkflowResolutionError, match="expected 'rag_add_files'"):
        _resolve_add_files_params(
            root_dir=None,
            retriever_name=None,
            include=None,
            exclude=None,
            recursive=None,
            force=False,
            batch_size=None,
            chunk_size=None,
            chunker=None,
            workflow_config="wrong.profile",
            config=rag_workflow_config,
        )
