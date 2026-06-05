"""Core logic for skill discovery, validation, installation, and creation."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import yaml
from loguru import logger

from genai_tk.main.models_skills import SkillInfo, SkillManifest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_FILENAME = "skills.yaml"

# Bundled skill roots shipped with genai-tk (relative to project root or
# resolved via genai_tk package path).
_BUNDLED_ROOTS_NAMES = ["langchain_examples", "public"]

# Map bundled subdirectory names to skill categories shown in `cli skills list`.
# Directories not in this map fall back to "project".
_DIR_CATEGORY: dict[str, str] = {
    "genai-tk": "dev",
    "copilot": "dev",
    "public": "agent",
    "langchain_examples": "agent",
}

SKILL_FRONTMATTER_REQUIRED = {"name", "description"}

SKILL_TEMPLATE = """\
---
name: {name}
description: {description}
tags: {tags}
version: "1.0"
author: ""
---

# {title}

## When to Use

{when_to_use}

## Workflow

1. Step one...
2. Step two...
3. Step three...

## Code Map

| Concern | Path |
|---------|------|
| Main logic | `<package>/...` |

## References

- [genai-tk docs](https://github.com/tclatos/genai-tk)
"""


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def parse_frontmatter(path: Path) -> dict:
    """Parse YAML frontmatter from a SKILL.md file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    match = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}


def discover_skills(roots: list[Path], source: str = "custom", category: str = "project") -> list[SkillInfo]:
    """Walk *roots* and return a SkillInfo for every SKILL.md found one level deep.

    Expected layout: <root>/<skill-name>/SKILL.md
    """
    skills: list[SkillInfo] = []
    for root in roots:
        if not root.is_dir():
            continue
        for skill_dir in sorted(root.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            fm = parse_frontmatter(skill_file)
            skills.append(
                SkillInfo(
                    name=fm.get("name", skill_dir.name),
                    description=fm.get("description", ""),
                    path=skill_dir,
                    source=source,  # type: ignore[arg-type]
                    category=category,  # type: ignore[arg-type]
                    tags=fm.get("tags") or [],
                    version=str(fm.get("version", "")),
                    author=str(fm.get("author", "")),
                )
            )
    return skills


def discover_all_skills(project_dir: Path) -> list[SkillInfo]:
    """Discover all skills in a project: bundled, custom, git/skillssh (from manifest)."""
    from importlib.resources import files as pkg_files

    skills: list[SkillInfo] = []

    # 1. Bundled skills shipped inside genai_tk package
    try:
        bundled_root = Path(str(pkg_files("genai_tk") / "skills"))
        if bundled_root.is_dir():
            for sub in sorted(bundled_root.iterdir()):
                if sub.is_dir():
                    cat = _DIR_CATEGORY.get(sub.name, "project")
                    skills.extend(discover_skills([sub], source="bundled", category=cat))
    except Exception as exc:
        logger.debug("Could not locate bundled skills: {}", exc)

    # 2. Project-local skills (custom + community)
    local_skills_root = project_dir / "skills"
    for sub_name in ("custom", "community"):
        sub = local_skills_root / sub_name
        if sub.is_dir():
            src = "skillssh" if sub_name == "community" else "custom"
            skills.extend(discover_skills([sub], source=src, category="project"))

    # 3. Manifest-tracked git/skillssh skills fill in the rest (already covered above,
    #    but we enrich with git_ref from the manifest).
    manifest = load_manifest(project_dir)
    for entry in manifest.skills:
        for s in skills:
            if s.name == entry.name and entry.git_ref:
                s.git_ref = entry.git_ref
                s.repo = entry.repo

    return skills


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_skill(path: Path) -> list[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors: list[str] = []
    skill_file = path / "SKILL.md" if path.is_dir() else path
    if not skill_file.exists():
        return [f"SKILL.md not found in {path}"]

    fm = parse_frontmatter(skill_file)
    for field in SKILL_FRONTMATTER_REQUIRED:
        if not fm.get(field):
            errors.append(f"Missing required frontmatter field: '{field}'")

    text = skill_file.read_text(encoding="utf-8")
    body = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()
    if not body:
        errors.append("SKILL.md body is empty (no content after frontmatter)")

    headings = [line for line in body.splitlines() if line.startswith("#")]
    if not headings:
        errors.append("SKILL.md has no headings — add at least one `## Section`")

    return errors


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def load_manifest(project_dir: Path) -> SkillManifest:
    manifest_path = project_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return SkillManifest()
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        skills_raw = raw.get("skills", [])
        skills = []
        for item in skills_raw:
            if isinstance(item, dict):
                skills.append(SkillInfo(**item))
        return SkillManifest(skills=skills)
    except Exception as exc:
        logger.warning("Could not parse {}: {}", manifest_path, exc)
        return SkillManifest()


def save_manifest(project_dir: Path, manifest: SkillManifest) -> None:
    manifest_path = project_dir / MANIFEST_FILENAME
    data = {
        "skills": [{k: v for k, v in s.model_dump().items() if v not in (None, "", [], {})} for s in manifest.skills]
    }
    # Convert Path objects to strings for YAML
    for entry in data["skills"]:
        if "path" in entry:
            entry["path"] = str(entry["path"])
    manifest_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def install_from_bundled(name: str, dest_root: Path) -> SkillInfo:
    """Copy a bundled genai-tk skill into dest_root/<name>/."""
    from importlib.resources import files as pkg_files

    bundled_root = Path(str(pkg_files("genai_tk") / "skills"))
    # Search all sub-dirs of bundled root for a matching skill
    matches: list[Path] = []
    for sub in bundled_root.rglob("SKILL.md"):
        skill_dir = sub.parent
        fm = parse_frontmatter(sub)
        skill_name = fm.get("name", skill_dir.name)
        if skill_name == name or skill_dir.name == name:
            matches.append(skill_dir)

    if not matches:
        raise ValueError(f"Bundled skill {name!r} not found. Run `cli skills search {name}` to find it.")

    src = matches[0]
    dest = dest_root / name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(str(src), str(dest))

    fm = parse_frontmatter(dest / "SKILL.md")
    return SkillInfo(
        name=fm.get("name", name),
        description=fm.get("description", ""),
        path=dest,
        source="bundled",
        tags=fm.get("tags") or [],
        version=str(fm.get("version", "")),
    )


def install_from_git(repo_url: str, dest_root: Path, ref: str = "", subpath: str = "") -> SkillInfo:
    """Shallow-clone a git repo and extract a skill into dest_root/."""
    import tempfile

    skill_name = Path(subpath).name if subpath else Path(repo_url.rstrip("/")).stem
    dest = dest_root / skill_name

    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["git", "clone", "--depth", "1"]
        if ref:
            cmd += ["--branch", ref]
        cmd += [repo_url, tmp]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr.strip()}")

        src = Path(tmp) / subpath if subpath else Path(tmp)

        # Find SKILL.md — could be at src/ or src/<skill-name>/
        if (src / "SKILL.md").exists():
            actual_src = src
        else:
            candidates = list(src.rglob("SKILL.md"))
            if not candidates:
                raise ValueError(f"No SKILL.md found in {repo_url} (subpath: {subpath or '/'})")
            actual_src = candidates[0].parent

        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(str(actual_src), str(dest))

        # Get the actual commit SHA
        sha_result = subprocess.run(["git", "-C", tmp, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        git_sha = sha_result.stdout.strip() if sha_result.returncode == 0 else ""

    fm = parse_frontmatter(dest / "SKILL.md")
    return SkillInfo(
        name=fm.get("name", skill_name),
        description=fm.get("description", ""),
        path=dest,
        source="git",
        tags=fm.get("tags") or [],
        version=str(fm.get("version", "")),
        repo=repo_url,
        git_ref=git_sha,
    )


def install_from_skillssh(owner_repo: str, dest_root: Path) -> list[SkillInfo]:
    """Install skills via `npx skills add <owner/repo>` and import the results.

    skills.sh CLI installs into the agent-specific config directory (e.g.,
    .cursor/rules/ for Cursor). We run it, then locate and copy any resulting
    SKILL.md files into dest_root/community/.
    """
    import tempfile

    # Check npx is available
    if not shutil.which("npx"):
        raise RuntimeError(
            "npx is not installed. Install Node.js to use skills.sh integration.\n  https://nodejs.org/en/download"
        )

    community_dir = dest_root / "community"
    community_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            ["npx", "--yes", "skills", "add", owner_repo],
            capture_output=True,
            text=True,
            cwd=tmp,
            env={**__import__("os").environ, "DISABLE_TELEMETRY": "1"},
        )
        if result.returncode != 0:
            raise RuntimeError(f"npx skills add failed:\n{result.stderr.strip() or result.stdout.strip()}")

        # Find any markdown files that look like skill files (have YAML frontmatter)
        installed: list[SkillInfo] = []
        for md_file in sorted(Path(tmp).rglob("*.md")):
            fm = parse_frontmatter(md_file)
            if not fm.get("name"):
                continue
            skill_name = fm.get("name", md_file.stem)
            dest = community_dir / skill_name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(md_file), str(dest / "SKILL.md"))
            installed.append(
                SkillInfo(
                    name=skill_name,
                    description=fm.get("description", ""),
                    path=dest,
                    source="skillssh",
                    tags=fm.get("tags") or [],
                    version=str(fm.get("version", "")),
                    repo=f"https://github.com/{owner_repo}",
                )
            )

    if not installed:
        raise ValueError(
            f"No skills with valid frontmatter found in {owner_repo}.\n"
            f"You can also install directly with: npx skills add {owner_repo}"
        )

    return installed


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def create_skill_skeleton(name: str, dest_root: Path, description: str = "", tags: list[str] | None = None) -> Path:
    """Scaffold a new SKILL.md in dest_root/<name>/SKILL.md."""
    skill_dir = dest_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    if skill_file.exists():
        raise FileExistsError(f"Skill already exists at {skill_file}")

    title = name.replace("-", " ").replace("_", " ").title()
    tags_str = str(tags or [])
    when_to_use = "Describe when an agent should use this skill."
    content = SKILL_TEMPLATE.format(
        name=name,
        description=description or f"TODO: describe what {name} does.",
        title=title,
        tags=tags_str,
        when_to_use=when_to_use,
    )
    skill_file.write_text(content, encoding="utf-8")
    return skill_dir


# ---------------------------------------------------------------------------
# Catalog (bundled skill listing)
# ---------------------------------------------------------------------------


def list_bundled_skills() -> list[SkillInfo]:
    """Return all skills shipped inside the genai_tk package."""
    from importlib.resources import files as pkg_files

    try:
        bundled_root = Path(str(pkg_files("genai_tk") / "skills"))
    except Exception:
        return []

    skills: list[SkillInfo] = []
    if bundled_root.is_dir():
        for sub in sorted(bundled_root.iterdir()):
            if sub.is_dir():
                found = discover_skills([sub], source="bundled")
                skills.extend(found)
    return skills
