"""Unit tests for DeerFlow config bridge skills-directory handling.

A missing skills directory is a benign, optional condition — DeerFlow degrades
gracefully with no skills. These tests pin the contract that such a missing
directory is logged at DEBUG (not WARNING) yet still recorded in the structured
``ConfigSetupWarnings.missing_skill_directories`` so callers can surface it as a
soft hint if desired.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from loguru import logger

pytestmark = pytest.mark.unit


def _capture_logs():
    """Add a loguru sink collecting all records at DEBUG+; return (records, sink_id)."""
    records: list = []
    sink_id = logger.add(records.append, level="DEBUG", format="{message}")
    return records, sink_id


def _warning_messages(records: list) -> list[str]:
    """Return the formatted text of every WARNING-level captured record."""
    return [str(m) for m in records if m.record["level"].name == "WARNING"]


# ---------------------------------------------------------------------------
# load_skills_from_directories
# ---------------------------------------------------------------------------


def test_load_skills_missing_directory_logs_debug_not_warning(tmp_path: Path) -> None:
    """A missing skills dir is DEBUG-logged and recorded structurally, not WARNED."""
    from genai_tk.agents.deer_flow.config_bridge import (
        ConfigSetupWarnings,
        load_skills_from_directories,
    )

    missing = tmp_path / "no-such-skills"
    records, sink_id = _capture_logs()
    try:
        warnings = ConfigSetupWarnings()
        skills = load_skills_from_directories([str(missing)], warnings=warnings)
    finally:
        logger.remove(sink_id)

    assert skills == []
    assert warnings.missing_skill_directories == [f"Skill directory does not exist: {missing}"]
    assert not any("Skill directory does not exist" in m for m in _warning_messages(records))


# ---------------------------------------------------------------------------
# write_deer_flow_config
# ---------------------------------------------------------------------------


def test_write_config_missing_skills_directory_logs_debug_not_warning(tmp_path: Path) -> None:
    """write_deer_flow_config DEBUG-logs a missing skills dir and records it structurally."""
    from genai_tk.agents.deer_flow.config_bridge import ConfigSetupWarnings, write_deer_flow_config

    warnings = ConfigSetupWarnings()
    records, sink_id = _capture_logs()
    try:
        write_deer_flow_config(
            models=[
                {"name": "test-model", "display_name": "Test", "use": "langchain_openai:ChatOpenAI", "model": "gpt-4"},
            ],
            sandbox="local",
            config_dir=str(tmp_path),
            skills_path=str(tmp_path / "missing-skills"),
            warnings=warnings,
        )
    finally:
        logger.remove(sink_id)

    # Structured warning preserved (info not lost for callers that surface it).
    assert any("Skills directory not found" in m for m in warnings.missing_skill_directories)
    # No WARNING-level log emitted for the missing skills dir.
    assert not any("Skills directory not found" in m for m in _warning_messages(records))


# ---------------------------------------------------------------------------
# deepagents-layout discovery + staging
# ---------------------------------------------------------------------------


def _write_deepagents_skill(root: Path, name: str, body: str = "demo") -> Path:
    skill = root / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: demo\n---\n\n# {name}\n{body}", encoding="utf-8")
    return skill


def test_load_skills_discovers_deepagents_layout_as_custom(tmp_path: Path) -> None:
    """Directories with direct skill subdirs (no public/custom) map to custom/<name>."""
    from genai_tk.agents.deer_flow.config_bridge import load_skills_from_directories

    _write_deepagents_skill(tmp_path / "pkg-skills", "decision-navigation")
    _write_deepagents_skill(tmp_path / "pkg-skills", "document-graph-tools")

    skills = load_skills_from_directories([str(tmp_path / "pkg-skills")])

    assert sorted(skills) == ["custom/decision-navigation", "custom/document-graph-tools"]


def test_setup_stages_non_category_sources_into_deerflow_layout(tmp_path: Path) -> None:
    """Sources without public/custom categories are staged into one mounted root."""
    import yaml

    from genai_tk.agents.deer_flow.config_bridge import setup_deer_flow_config

    pkg_skills = tmp_path / "pkg-skills"
    _write_deepagents_skill(pkg_skills, "decision-navigation")
    classic_root = tmp_path / "classic-skills"
    classic = classic_root / "custom" / "my-skill"
    classic.mkdir(parents=True)
    (classic / "SKILL.md").write_text("---\nname: my-skill\ndescription: demo\n---\n\n# mine\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    config_path, _ext, _warnings = setup_deer_flow_config(
        skill_directories=[str(classic_root), str(pkg_skills)],
        config_dir=str(out_dir),
    )

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mounted_root = Path(cfg["skills"]["path"])
    # All sources staged: classic skills keep their category, deepagents ones land under custom/.
    assert (mounted_root / "custom" / "my-skill" / "SKILL.md").is_file()
    assert (mounted_root / "custom" / "decision-navigation" / "SKILL.md").is_file()
    assert not (mounted_root / "public").exists()  # public category only when a source ships it
