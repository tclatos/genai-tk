"""Project scaffolding — renders Jinja2 templates into a new project directory."""

from __future__ import annotations

import re
from importlib.resources import files as pkg_files
from pathlib import Path

from rich.console import Console

console = Console()

_TEMPLATES_PKG = "templates/project"


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

    def __init__(self, project_dir: Path, project_name: str, *, force: bool = False) -> None:
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
        # Jinja2 needs a real filesystem path
        tpl_path = Path(str(tpl_root))
        env = Environment(
            loader=FileSystemLoader(str(tpl_path)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        ctx = {
            "project_name": self.project_name,
            "package_name": self.package_name,
        }

        # Map of template path → output path (relative to project_dir)
        file_map: dict[str, str] = {
            "pyproject.toml.j2": "pyproject.toml",
            "README.md.j2": "README.md",
            "__init__.py.j2": f"{self.package_name}/__init__.py",
            "main/streamlit.py.j2": f"{self.package_name}/main/streamlit.py",
            "commands/example_commands.py.j2": f"{self.package_name}/commands/example_commands.py",
            "chains/joke_chain.py.j2": f"{self.package_name}/chains/joke_chain.py",
            "webapp/pages/demos/hello_agent.py.j2": f"{self.package_name}/webapp/pages/demos/hello_agent.py",
            "AGENTS.md.j2": "AGENTS.md",
            ".github/copilot-instructions.md.j2": ".github/copilot-instructions.md",
        }

        for tpl_name, out_rel in file_map.items():
            self._render_template(env, ctx, tpl_name, out_rel)

        # Create __init__.py files for sub-packages (empty)
        for sub in ("main", "commands", "chains", "webapp", "webapp/pages", "webapp/pages/demos"):
            init_path = self.project_dir / self.package_name / sub / "__init__.py"
            if sub != "":  # root __init__ already rendered from template
                self._write_file(init_path, "")

        if self._written:
            console.print(
                f"[green]✓ Scaffolded {self._written} file(s) into[/green] "
                f"[bold]{self.project_dir / self.package_name}[/bold]"
            )
        if self._skipped:
            console.print(f"[yellow]  ({self._skipped} file(s) already exist — use --force to overwrite)[/yellow]")

        # Patch config files to reference the generated package
        self._patch_app_conf()
        self._patch_webapp_yaml()
        self._patch_makefile_streamlit_entry()
        self._ensure_package_installed()

        return self._written

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_template(self, env, ctx: dict, tpl_name: str, out_rel: str) -> None:
        template = env.get_template(tpl_name)
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
        """Add the example commands registration to config/app_conf.yaml."""
        app_conf = self.project_dir / "config" / "app_conf.yaml"
        if not app_conf.exists():
            return
        content = app_conf.read_text(encoding="utf-8")
        entry = f"    - {self.package_name}.commands.example_commands.ExampleCommands"
        if entry in content:
            return
        # Append after the last existing command registration
        marker = "    - genai_tk.main.cli.register_commands"
        if marker in content:
            content = content.replace(marker, f"{marker}\n{entry}")
        else:
            # Fallback: append to cli.commands section
            content += f"\n{entry}\n"
        app_conf.write_text(content, encoding="utf-8")
        console.print(f"[green]✓ Registered ExampleCommands in[/green] [bold]{app_conf}[/bold]")

    def _patch_webapp_yaml(self) -> None:
        """Set pages_dir and navigation in webapp.yaml to reference the generated pages."""
        webapp_yaml = self.project_dir / "config" / "webapp.yaml"
        if not webapp_yaml.exists():
            return
        content = webapp_yaml.read_text(encoding="utf-8")
        # Skip if an active (non-commented) pages_dir is already set
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("pages_dir:") and not stripped.startswith("#"):
                return  # Already configured

        # Use ${paths.project} so the path resolves correctly regardless of CWD
        new_nav = (
            f"  pages_dir: ${{paths.project}}/{self.package_name}/webapp/pages\n"
            f"  navigation:\n"
            f"    demos:\n"
            f"      - demos/hello_agent.py\n"
        )

        # Strip all the commented-out navigation block and append active config
        lines = content.splitlines()
        out_lines = []
        skip_section = False
        for line in lines:
            stripped = line.strip()
            # Start skipping from the separator comment that precedes pages_dir
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
        console.print(f"[green]✓ Configured webapp pages in[/green] [bold]{webapp_yaml}[/bold]")

    def _patch_makefile_streamlit_entry(self) -> None:
        """Set STREAMLIT_ENTRY in the Makefile to the generated entry point."""
        makefile = self.project_dir / "Makefile"
        if not makefile.exists():
            return
        content = makefile.read_text(encoding="utf-8")
        old = "STREAMLIT_ENTRY ?="
        new = f"STREAMLIT_ENTRY ?= {self.package_name}/main/streamlit.py"
        if old in content and new not in content:
            content = content.replace(old, new, 1)
            makefile.write_text(content, encoding="utf-8")
            console.print(f"[green]✓ Set STREAMLIT_ENTRY in[/green] [bold]{makefile}[/bold]")

    def _ensure_package_installed(self) -> None:
        """Run 'uv sync' so the generated package is importable by the CLI."""
        import subprocess

        pyproject = self.project_dir / "pyproject.toml"
        if not pyproject.exists():
            return

        content = pyproject.read_text(encoding="utf-8")
        changed = False

        # Add package=true so uv treats project as an installable package
        if "[tool.uv]" not in content:
            content += "\n[tool.uv]\npackage = true\n"
            changed = True
        elif "package = true" not in content:
            content = content.replace("[tool.uv]", "[tool.uv]\npackage = true")
            changed = True

        # Add explicit package discovery to prevent setuptools from picking up config/
        discovery_marker = "[tool.setuptools.packages.find]"
        if discovery_marker not in content:
            content += f'\n{discovery_marker}\ninclude = ["{self.package_name}*"]\n'
            changed = True

        if changed:
            pyproject.write_text(content, encoding="utf-8")
            console.print("[green]✓ Enabled package mode in[/green] [bold]pyproject.toml[/bold]")

        console.print("[cyan]Running uv sync to install the project package...[/cyan]")
        result = subprocess.run(
            ["uv", "sync", "--no-install-workspace"],
            cwd=str(self.project_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            console.print("[green]✓ Package installed (uv sync)[/green]")
        else:
            console.print(f"[yellow]uv sync failed — run it manually:[/yellow] {result.stderr.strip()[:200]}")
