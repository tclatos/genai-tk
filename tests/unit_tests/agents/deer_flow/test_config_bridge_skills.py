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
