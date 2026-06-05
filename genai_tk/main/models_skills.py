"""Pydantic models for the skills management system."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SkillInfo(BaseModel):
    """Metadata for a single skill discovered on disk."""

    name: str
    description: str = ""
    path: Path
    source: Literal["bundled", "custom", "git", "skillssh"] = "custom"
    # dev = for developers building with genai-tk; agent = runtime capabilities
    category: Literal["dev", "agent", "project"] = "project"
    tags: list[str] = Field(default_factory=list)
    version: str = ""
    author: str = ""
    # Only for git/skillssh-sourced skills
    repo: str = ""
    git_ref: str = ""

    model_config = {"arbitrary_types_allowed": True}

    @property
    def display_source(self) -> str:
        if self.source == "bundled":
            return "[dim]bundled[/dim]"
        if self.source == "git":
            ref = f"@{self.git_ref[:7]}" if self.git_ref else ""
            return f"[blue]git{ref}[/blue]"
        if self.source == "skillssh":
            return "[magenta]skills.sh[/magenta]"
        return "[green]custom[/green]"


class SkillManifest(BaseModel):
    """Represents skills.yaml — the lockfile for installed skills."""

    skills: list[SkillInfo] = Field(default_factory=list)

    def get(self, name: str) -> SkillInfo | None:
        for s in self.skills:
            if s.name == name:
                return s
        return None

    def upsert(self, skill: SkillInfo) -> None:
        for i, s in enumerate(self.skills):
            if s.name == skill.name:
                self.skills[i] = skill
                return
        self.skills.append(skill)

    def remove(self, name: str) -> bool:
        before = len(self.skills)
        self.skills = [s for s in self.skills if s.name != name]
        return len(self.skills) < before
