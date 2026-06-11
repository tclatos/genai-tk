"""Project scaffolding — renders Jinja2 templates into a new project directory.

Supports multiple templates: agent-app, rag-app, workflow-app, minimal.
All templates share a common base (pyproject.toml, AGENTS.md, justfile, agent-support files),
and each template adds its own files on top.
"""

from __future__ import annotations

import re
from importlib.resources import files as pkg_files
from pathlib import Path

from rich.console import Console

console = Console()

_TEMPLATES_PKG = "templates/project"

_AGENT_APP_META: dict = {
    "label": "Agent App",
    "description": "AI agent application with tools, skills, and configurable profiles.",
    "run_command": "cli agent chat",
    "quickstart_run": "chat with default agent",
    "structure_entries": [
        "commands/         # CLI command groups",
        "tools/            # LangChain tools",
        "webapp/pages/     # Streamlit pages",
    ],
    "project_recipes": "# Launch agent chat\nrun:\n    uv run cli agent chat\n",
}


def _sanitize_package_name(name: str) -> str:
    """Convert a human-readable project name to a valid Python package name."""
    pkg = name.lower().strip()
    pkg = re.sub(r"[^a-z0-9]+", "_", pkg)
    pkg = pkg.strip("_")
    if not pkg or pkg[0].isdigit():
        pkg = "my_project"
    return pkg


class ProjectScaffolder:
    """Render Jinja2 templates into a target directory.

    Args:
        project_dir: Root directory of the new project (usually cwd).
        project_name: Human-readable name (e.g. "My AI Project").
        force: Overwrite existing files.
    """

    def __init__(
        self,
        project_dir: Path,
        project_name: str,
        *,
        force: bool = False,
    ) -> None:
        self.project_dir = project_dir
        self.project_name = project_name
        self.package_name = _sanitize_package_name(project_name)
        self.force = force
        self._written = 0
        self._skipped = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scaffold(self) -> int:
        """Generate the full project scaffold. Returns number of files written."""
        from jinja2 import Environment, FileSystemLoader

        tpl_root = pkg_files("genai_tk") / _TEMPLATES_PKG
        tpl_path = Path(str(tpl_root))
        meta = _AGENT_APP_META

        ctx = {
            "project_name": self.project_name,
            "package_name": self.package_name,
            "template_label": meta["label"],
            "template_description": meta["description"],
            "run_command": meta["run_command"],
            "quickstart_run": meta["quickstart_run"],
            "structure_entries": meta["structure_entries"],
            "project_recipes": meta["project_recipes"],
        }

        search_dirs = [
            str(tpl_path / "common"),
            str(tpl_path / "agent-app"),
        ]
        env = Environment(
            loader=FileSystemLoader(search_dirs),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # ── Common files (all templates) ─────────────────────────────
        common_map: dict[str, str] = {
            "pyproject.toml.j2": "pyproject.toml",
            "README.md.j2": "README.md",
            "__init__.py.j2": f"{self.package_name}/__init__.py",
            "AGENTS.md.j2": "AGENTS.md",
            "justfile.j2": "justfile",
            ".github/copilot-instructions.md.j2": ".github/copilot-instructions.md",
            "docs/SKILLS.md.j2": "docs/SKILLS.md",
            "docs/EXTENDING.md.j2": "docs/EXTENDING.md",
        }
        for tpl_name, out_rel in common_map.items():
            self._render_template(env, ctx, tpl_name, out_rel)

        # ── Template-specific files ───────────────────────────────────
        self._scaffold_template_files(env, ctx)

        # ── Skills directory structure ────────────────────────────────
        for sub in ("custom", "community", "bundled"):
            d = self.project_dir / "skills" / sub
            d.mkdir(parents=True, exist_ok=True)
            gitkeep = d / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.write_text("")
                self._written += 1

        # ── Copy genai-tk bundled skills into skills/genai-tk/ ────────
        self._copy_genai_tk_skills()

        # ── Package sub-package __init__.py files ─────────────────────
        for sub in self._get_package_subdirs():
            init_path = self.project_dir / self.package_name / sub / "__init__.py"
            self._write_file(init_path, "")

        if self._written:
            console.print(f"[green]✓ Scaffolded {self._written} file(s)[/green]")
        if self._skipped:
            console.print(f"[yellow]  ({self._skipped} file(s) already exist — use --force to overwrite)[/yellow]")

        # ── Remove uv-init stub files not needed in genai-tk projects ──
        for stub in ("main.py", "hello.py"):
            stub_path = self.project_dir / stub
            if stub_path.exists() and stub_path.read_text(encoding="utf-8").strip().startswith(
                ("def main()", "print(")
            ):
                stub_path.unlink()
                console.print(f"[dim]✓ Removed uv-init stub: {stub}[/dim]")

        # ── Config patches ─────────────────────────────────────────────
        self._patch_app_conf()
        self._patch_webapp_yaml()
        self._ensure_package_installed()

        return self._written

    # ------------------------------------------------------------------
    # Template-specific file maps
    # ------------------------------------------------------------------

    def _scaffold_template_files(self, env, ctx: dict) -> None:
        pkg = self.package_name
        file_map = {
            "commands/agent_commands.py.j2": f"{pkg}/commands/agent_commands.py",
            "tools/example_tool.py.j2": f"{pkg}/tools/example_tool.py",
            "config/langchain.yaml.j2": "config/agents/langchain.yaml",
            "skills/custom/getting-started/SKILL.md.j2": "skills/custom/getting-started/SKILL.md",
        }
        self._render_hello_agent_page(ctx, pkg)
        for tpl_name, out_rel in file_map.items():
            self._render_template(env, ctx, tpl_name, out_rel)

    def _render_hello_agent_page(self, ctx: dict, pkg: str) -> None:
        """Render webapp page + main/streamlit.py from the original templates (agent-app only)."""
        from jinja2 import Environment, FileSystemLoader

        tpl_root = Path(str(pkg_files("genai_tk") / _TEMPLATES_PKG))
        try:
            old_env = Environment(
                loader=FileSystemLoader(str(tpl_root)),
                keep_trailing_newline=True,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            for tpl_name, out_rel in {
                "webapp/pages/demos/hello_agent.py.j2": f"{pkg}/webapp/pages/demos/hello_agent.py",
                "main/streamlit.py.j2": f"{pkg}/main/streamlit.py",
            }.items():
                try:
                    tpl = old_env.get_template(tpl_name)
                    self._write_file(self.project_dir / out_rel, tpl.render(**ctx))
                except Exception:
                    pass
        except Exception:
            pass

    def _get_package_subdirs(self) -> list[str]:
        return ["commands", "tools", "webapp", "webapp/pages", "webapp/pages/demos", "main"]

    def _copy_genai_tk_skills(self) -> None:
        """Copy skills/genai-tk/ from the installed package into the project's skills/genai-tk/."""
        import shutil

        try:
            bundled_root = Path(str(pkg_files("genai_tk") / "skills" / "genai-tk"))
        except Exception:
            return
        if not bundled_root.is_dir():
            return

        dest = self.project_dir / "skills" / "genai-tk"
        if dest.exists() and not self.force:
            return

        shutil.copytree(bundled_root, dest, dirs_exist_ok=True)
        count = sum(1 for _ in dest.rglob("SKILL.md"))
        self._written += count
        console.print(f"[green]✓ Installed {count} genai-tk skill(s)[/green] → skills/genai-tk/")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_template(self, env, ctx: dict, tpl_name: str, out_rel: str) -> None:
        try:
            template = env.get_template(tpl_name)
        except Exception:
            return
        content = template.render(**ctx)
        target = self.project_dir / out_rel
        self._write_file(target, content)

    def _write_file(self, target: Path, content: str) -> None:
        if target.exists() and not self.force:
            self._skipped += 1
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self._written += 1

    def _patch_app_conf(self) -> None:
        app_conf = self.project_dir / "config" / "app_conf.yaml"
        if not app_conf.exists():
            return
        content = app_conf.read_text(encoding="utf-8")

        entry_class = f"{self.package_name}.commands.agent_commands.AgentCommands"
        if not entry_class:
            return

        entry = f"    - {entry_class}"
        if entry in content:
            return

        marker = "    - genai_tk.main.cli.register_commands"
        if marker in content:
            content = content.replace(marker, f"{marker}\n{entry}")
        else:
            content += f"\n{entry}\n"

        app_conf.write_text(content, encoding="utf-8")
        console.print(f"[green]✓ Registered {entry_class.split('.')[-1]} in[/green] [bold]{app_conf.name}[/bold]")

    def _patch_webapp_yaml(self) -> None:
        webapp_yaml = self.project_dir / "config" / "webapp.yaml"
        if not webapp_yaml.exists():
            return
        content = webapp_yaml.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("pages_dir:") and not stripped.startswith("#"):
                return

        new_nav = (
            f"  pages_dir: ${{paths.project}}/{self.package_name}/webapp/pages\n"
            f"  navigation:\n"
            f"    demos:\n"
            f"      - demos/hello_agent.py\n"
        )
        lines = content.splitlines()
        out_lines = []
        skip_section = False
        for line in lines:
            stripped = line.strip()
            if stripped in (
                "# -------------------------------------------------------------------------",
                "# Custom navigation (optional).",
                "# When both pages_dir and navigation are set, the webapp will load pages",
                "# from pages_dir using the structure defined below.",
                "# When left unset, only the built-in agent demos are shown.",
            ):
                skip_section = True
                continue
            if skip_section and (stripped.startswith("#") or stripped == ""):
                continue
            skip_section = False
            out_lines.append(line)

        content = "\n".join(out_lines).rstrip() + "\n\n" + new_nav
        webapp_yaml.write_text(content, encoding="utf-8")
        console.print(f"[green]✓ Configured webapp pages in[/green] [bold]{webapp_yaml.name}[/bold]")

    def _ensure_package_installed(self) -> None:
        import subprocess

        pyproject = self.project_dir / "pyproject.toml"
        if not pyproject.exists():
            return

        content = pyproject.read_text(encoding="utf-8")
        changed = False

        if "[tool.uv]" not in content:
            content += "\n[tool.uv]\npackage = true\n"
            changed = True
        elif "package = true" not in content:
            content = content.replace("[tool.uv]", "[tool.uv]\npackage = true")
            changed = True

        if "[build-system]" not in content:
            build_section = (
                "\n[build-system]\n"
                'requires = ["hatchling"]\n'
                'build-backend = "hatchling.build"\n'
                "\n[tool.hatch.build.targets.wheel]\n"
                f'packages = ["{self.package_name}"]\n'
            )
            content += build_section
            # Remove legacy setuptools block if present
            content = content.replace(f'\n[tool.setuptools.packages.find]\ninclude = ["{self.package_name}*"]\n', "")
            changed = True

        if "[project.optional-dependencies]" not in content:
            optional_deps = (
                "\n# Forward genai-tk optional extras — install with: uv sync --extra <name>\n"
                "[project.optional-dependencies]\n"
                'harnessing = ["genai-tk[harnessing]"]\n'
                'browser    = ["genai-tk[browser]"]\n'
                'nlp        = ["genai-tk[nlp]"]\n'
                'postgres   = ["genai-tk[postgres]"]\n'
                'streamlit  = ["genai-tk[streamlit]"]\n'
                'baml       = ["genai-tk[baml]"]\n'
                'chromadb   = ["genai-tk[chromadb]"]\n'
            )
            # Insert before [build-system] if present, otherwise append
            if "[build-system]" in content:
                content = content.replace("[build-system]", optional_deps + "\n[build-system]", 1)
            else:
                content += optional_deps
            changed = True

        if changed:
            pyproject.write_text(content, encoding="utf-8")
            console.print("[green]✓ Enabled package mode in pyproject.toml[/green]")

        console.print("[cyan]Running uv sync...[/cyan]")
        result = subprocess.run(
            ["uv", "sync", "--no-install-workspace"],
            cwd=str(self.project_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            console.print("[green]✓ uv sync complete[/green]")
        else:
            console.print(f"[yellow]uv sync failed — run manually:[/yellow] {result.stderr.strip()[:200]}")
