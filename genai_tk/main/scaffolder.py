"""Project scaffolding — renders Jinja2 templates into a new project directory.

Supports multiple templates: agent-app, rag-app, workflow-app, minimal.
All templates share a common base (pyproject.toml, AGENTS.md, justfile, agent-support files),
and each template adds its own files on top.
"""

from __future__ import annotations

import re
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Literal

from rich.console import Console

console = Console()

_TEMPLATES_PKG = "templates/project"

Template = Literal["agent-app", "rag-app", "workflow-app", "minimal"]

TEMPLATE_META: dict[str, dict] = {
    "agent-app": {
        "label": "Agent App",
        "description": "AI agent application with tools, skills, and configurable profiles.",
        "run_command": "cli agent chat",
        "quickstart_run": "chat with default agent",
        "structure_entries": [
            "commands/         # CLI command groups",
            "tools/            # LangChain tools",
            "webapp/pages/     # Streamlit pages",
        ],
        "project_recipes": "# Launch agent chat\nrun:\n    uv run cli agent chat\n\n# Launch the webapp\nwebapp:\n    uv run cli webapp\n",
    },
    "rag-app": {
        "label": "RAG Pipeline",
        "description": "Retrieval-Augmented Generation pipeline with document ingestion and querying.",
        "run_command": "cli rag query <question>",
        "quickstart_run": "cli rag query 'your question'",
        "structure_entries": [
            "commands/         # CLI command groups",
            "loaders/          # document loaders",
            "data/             # raw / processed documents",
        ],
        "project_recipes": '# Ingest documents\ningest:\n    uv run cli rag ingest data/raw\n\n# Query\nquery question="What is this about?":\n    uv run cli rag query "{{question}}"\n',
    },
    "workflow-app": {
        "label": "Workflow App",
        "description": "Multi-step data pipeline orchestrated with YAML workflow definitions.",
        "run_command": "cli workflow run example",
        "quickstart_run": "cli workflow run example --dry-run",
        "structure_entries": [
            "commands/         # CLI command groups",
            "workflows/steps/  # workflow step functions",
            "config/workflows/ # workflow YAML definitions",
        ],
        "project_recipes": '# Dry-run a workflow\ndry-run workflow="example":\n    uv run cli workflow run {{workflow}} --dry-run\n\n# Run a workflow\nrun workflow="example":\n    uv run cli workflow run {{workflow}}\n',
    },
    "minimal": {
        "label": "Minimal",
        "description": "Config, justfile, and AGENTS.md only — no example code.",
        "run_command": "cli --help",
        "quickstart_run": "explore available commands",
        "structure_entries": [],
        "project_recipes": "# Launch agent chat\nrun:\n    uv run cli agents langchain --chat\n",
    },
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
        template: One of agent-app | rag-app | workflow-app | minimal.
        force: Overwrite existing files.
    """

    def __init__(
        self,
        project_dir: Path,
        project_name: str,
        *,
        template: Template = "agent-app",
        force: bool = False,
    ) -> None:
        self.project_dir = project_dir
        self.project_name = project_name
        self.package_name = _sanitize_package_name(project_name)
        self.template = template
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
        meta = TEMPLATE_META[self.template]

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
            str(tpl_path / self.template),
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
            ".cursor/rules/genai-tk.mdc.j2": ".cursor/rules/genai-tk.mdc",
            ".windsurfrules.j2": ".windsurfrules",
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
            console.print(
                f"[green]✓ Scaffolded {self._written} file(s)[/green] (template: [bold]{self.template}[/bold])"
            )
        if self._skipped:
            console.print(f"[yellow]  ({self._skipped} file(s) already exist — use --force to overwrite)[/yellow]")

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

        if self.template == "agent-app":
            file_map = {
                "commands/agent_commands.py.j2": f"{pkg}/commands/agent_commands.py",
                "tools/example_tool.py.j2": f"{pkg}/tools/example_tool.py",
                "config/langchain.yaml.j2": "config/agents/langchain.yaml",
                "skills/custom/getting-started/SKILL.md.j2": "skills/custom/getting-started/SKILL.md",
            }
            self._render_hello_agent_page(ctx, pkg)

        elif self.template == "rag-app":
            file_map = {
                "commands/rag_commands.py.j2": f"{pkg}/commands/rag_commands.py",
            }
            for sub in ("raw", "processed"):
                d = self.project_dir / "data" / sub
                d.mkdir(parents=True, exist_ok=True)
                readme = d / "README.md"
                if not readme.exists():
                    readme.write_text(f"# {sub.title()} Data\n\nPlace your {sub} documents here.\n")
                    self._written += 1

        elif self.template == "workflow-app":
            file_map = {
                "workflows/steps/example_step.py.j2": f"{pkg}/workflows/steps/example_step.py",
                "config/pipeline.yaml.j2": "config/workflows/pipeline.yaml",
            }
            for sub in ("workflows", "workflows/steps"):
                init = self.project_dir / pkg / sub / "__init__.py"
                self._write_file(init, "")

        else:  # minimal
            file_map = {}

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
        base = ["commands"]
        if self.template == "agent-app":
            return base + ["tools", "webapp", "webapp/pages", "webapp/pages/demos", "main"]
        if self.template == "rag-app":
            return base + ["loaders"]
        if self.template == "workflow-app":
            return base + ["workflows", "workflows/steps"]
        return []

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

        command_classes = {
            "agent-app": f"{self.package_name}.commands.agent_commands.AgentCommands",
            "rag-app": f"{self.package_name}.commands.rag_commands.RagCommands",
            # workflow-app: genai_tk.workflow.commands.WorkflowCommands (already registered in app_conf)
            # handles `cli workflow run` — no project-level WorkflowCommands needed.
            "workflow-app": None,
            "minimal": None,
        }
        entry_class = command_classes.get(self.template)
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
        if self.template != "agent-app":
            return
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

        discovery_marker = "[tool.setuptools.packages.find]"
        if discovery_marker not in content:
            content += f'\n{discovery_marker}\ninclude = ["{self.package_name}*"]\n'
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
