"""CLI commands for managing skills — `cli skills`.

Skills are SKILL.md files that give agents procedural knowledge on demand.
This command group lets you list, add, create, validate, and search skills.

Usage:
    cli skills list                                   # show all skills
    cli skills add deep-research                      # install bundled skill
    cli skills add --git https://github.com/... --path retrieval
    cli skills add --skillssh langchain-ai/langchain-skills
    cli skills create my-domain                       # scaffold a new skill
    cli skills validate --all                         # lint all skills
    cli skills info deep-research                     # show skill metadata
    cli skills search rag                             # find skills by keyword
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from genai_tk.cli.base import CliTopCommand
from genai_tk.main.models_skills import SkillManifest

console = Console()


def _project_dir() -> Path:
    return Path.cwd()


def _skills_custom_dir(project_dir: Path) -> Path:
    d = project_dir / "skills" / "custom"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _skills_community_dir(project_dir: Path) -> Path:
    d = project_dir / "skills" / "community"
    d.mkdir(parents=True, exist_ok=True)
    return d


class SkillsCommands(CliTopCommand):
    """Manage agent skills — discover, install, create, and validate SKILL.md files."""

    description: str = "Manage agent skills (list, add, create, validate)"

    def get_description(self) -> tuple[str, str]:
        return "skills", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:

        @cli_app.command("list")
        def list_skills(
            source: Annotated[
                Optional[str],
                typer.Option("--source", "-s", help="Filter by source: bundled, custom, git, skillssh"),
            ] = None,
            category: Annotated[
                Optional[str],
                typer.Option("--category", "-c", help="Filter by category: dev, agent, project"),
            ] = None,
        ) -> None:
            """List all discovered skills (bundled + custom + community)."""
            from genai_tk.main.skills_manager import discover_all_skills

            project_dir = _project_dir()
            skills = discover_all_skills(project_dir)

            if source:
                skills = [s for s in skills if s.source == source]
            if category:
                skills = [s for s in skills if s.category == category]

            if not skills:
                console.print("[yellow]No skills found.[/yellow]")
                if not source and not category:
                    console.print("  Run [bold]cli skills add <name>[/bold] to install skills.")
                return

            _CAT_LABELS = {
                "dev": "Dev Skills  [dim](for building with genai-tk)[/dim]",
                "agent": "Agent Skills  [dim](runtime capabilities for agents)[/dim]",
                "project": "Project Skills  [dim](custom / community)[/dim]",
            }

            from itertools import groupby

            sorted_skills = sorted(skills, key=lambda x: (x.category, x.name))
            groups = {k: list(v) for k, v in groupby(sorted_skills, key=lambda x: x.category)}

            total = len(skills)
            console.print(f"\n[bold]Skills[/bold] [dim]({total} found)[/dim]\n")

            for cat_key in ("dev", "agent", "project"):
                group = groups.get(cat_key)
                if not group:
                    continue
                label = _CAT_LABELS.get(cat_key, cat_key)
                table = Table(title=label, show_lines=False, border_style="dim", title_justify="left")
                table.add_column("Name", style="bold cyan", no_wrap=True)
                table.add_column("Description")
                table.add_column("Source", style="dim")
                table.add_column("Tags", style="dim")
                for s in group:
                    tags_str = ", ".join(s.tags) if s.tags else ""
                    table.add_row(s.name, s.description or "[dim]—[/dim]", s.display_source, tags_str)
                console.print(table)
                console.print()

            console.print(
                "[dim]Filter:[/dim]  cli skills list --category dev  |  --category agent  |  --category project"
            )
            console.print(
                "[dim]Add:[/dim]    cli skills add <name>  |  "
                "cli skills add --git <url>  |  cli skills add --skillssh <owner/repo>"
            )

        @cli_app.command("add")
        def add_skill(
            name: Annotated[
                Optional[str],
                typer.Argument(help="Name of a bundled skill to install."),
            ] = None,
            git: Annotated[
                Optional[str],
                typer.Option("--git", help="Git repository URL to install from."),
            ] = None,
            ref: Annotated[
                Optional[str],
                typer.Option("--ref", help="Git ref (tag or SHA) to pin. Only with --git."),
            ] = None,
            path: Annotated[
                Optional[str],
                typer.Option("--path", help="Subdirectory inside the git repo containing the SKILL.md."),
            ] = None,
            skillssh: Annotated[
                Optional[str],
                typer.Option(
                    "--skillssh",
                    help="Install from skills.sh registry (owner/repo). Wraps `npx skills add`.",
                ),
            ] = None,
            force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite if already installed.")] = False,
        ) -> None:
            """Install a skill from bundled catalog, git repo, or skills.sh registry.

            \\b
            Examples:
                cli skills add deep-research
                cli skills add --git https://github.com/langchain-ai/langchain-skills --path retrieval
                cli skills add --skillssh langchain-ai/langchain-skills
            """
            from genai_tk.main.skills_manager import (
                load_manifest,
                save_manifest,
            )

            project_dir = _project_dir()
            manifest = load_manifest(project_dir)

            if skillssh:
                _add_from_skillssh(skillssh, project_dir, manifest)
                save_manifest(project_dir, manifest)
                return

            if git:
                _add_from_git(git, ref or "", path or "", project_dir, manifest, force)
                save_manifest(project_dir, manifest)
                return

            if name:
                _add_bundled(name, project_dir, manifest, force)
                save_manifest(project_dir, manifest)
                return

            console.print(
                "[red]Specify a skill name, --git <url>, or --skillssh <owner/repo>.[/red]\n"
                "  [dim]cli skills add deep-research[/dim]\n"
                "  [dim]cli skills add --git https://github.com/langchain-ai/langchain-skills --path retrieval[/dim]\n"
                "  [dim]cli skills add --skillssh langchain-ai/langchain-skills[/dim]"
            )
            raise typer.Exit(1)

        @cli_app.command("create")
        def create_skill(
            name: Annotated[str, typer.Argument(help="Skill name (used as directory name and frontmatter name).")],
            description: Annotated[
                Optional[str],
                typer.Option("--description", "-d", help="One-line description of the skill."),
            ] = None,
            tags: Annotated[
                Optional[str],
                typer.Option("--tags", help="Comma-separated tags (e.g. 'rag,search,langchain')."),
            ] = None,
            interactive: Annotated[
                bool,
                typer.Option("--interactive", "-i", help="Prompt for description and tags interactively."),
            ] = False,
        ) -> None:
            """Scaffold a new SKILL.md in skills/custom/<name>/."""
            from genai_tk.main.skills_manager import create_skill_skeleton

            project_dir = _project_dir()
            custom_dir = _skills_custom_dir(project_dir)

            desc = description
            tag_list: list[str] = [t.strip() for t in tags.split(",")] if tags else []

            if interactive or (not desc):
                try:
                    import questionary

                    if not desc:
                        desc = questionary.text(
                            f"One-line description for '{name}':",
                            instruction="(what does this skill enable an agent to do?)",
                        ).ask()
                    if not tag_list:
                        raw_tags = questionary.text(
                            "Tags (comma-separated, optional):",
                            default="",
                        ).ask()
                        tag_list = [t.strip() for t in raw_tags.split(",") if t.strip()] if raw_tags else []
                except ImportError:
                    if not desc:
                        desc = typer.prompt(f"Description for '{name}'", default=f"Skill for {name}")

            try:
                skill_dir = create_skill_skeleton(name, custom_dir, description=desc or "", tags=tag_list)
                console.print(f"[green]✓ Created skill:[/green] {skill_dir / 'SKILL.md'}")
                console.print(f"\n[bold]Next:[/bold] Edit [cyan]{skill_dir / 'SKILL.md'}[/cyan] to add your content.")
                console.print(
                    "Then enable it in your agent profile:\n"
                    "  [dim]skill_directories:\n"
                    "    - ${paths.project}/skills/custom[/dim]"
                )
            except FileExistsError as exc:
                console.print(f"[red]{exc}[/red]  Use --force or choose a different name.")
                raise typer.Exit(1) from exc

        @cli_app.command("validate")
        def validate_skills(
            name: Annotated[
                Optional[str],
                typer.Argument(help="Skill name to validate (omit to validate all local skills)."),
            ] = None,
            all_skills: Annotated[
                bool, typer.Option("--all", "-a", help="Validate all skills including bundled.")
            ] = False,
        ) -> None:
            """Validate SKILL.md structure and frontmatter."""
            from genai_tk.main.skills_manager import discover_all_skills, validate_skill

            project_dir = _project_dir()
            skills = discover_all_skills(project_dir)

            if name:
                skills = [s for s in skills if s.name == name]
                if not skills:
                    console.print(f"[red]Skill '{name}' not found.[/red]")
                    raise typer.Exit(1)
            elif not all_skills:
                # Default: only custom + community
                skills = [s for s in skills if s.source in ("custom", "git", "skillssh")]

            if not skills:
                console.print("[yellow]No skills to validate.[/yellow]")
                return

            errors_found = 0
            for s in skills:
                errs = validate_skill(s.path)
                if errs:
                    errors_found += 1
                    console.print(f"[red]✗[/red] [bold]{s.name}[/bold] ({s.path})")
                    for e in errs:
                        console.print(f"    [red]•[/red] {e}")
                else:
                    console.print(f"[green]✓[/green] {s.name}")

            if errors_found:
                console.print(f"\n[red]{errors_found} skill(s) have validation errors.[/red]")
                raise typer.Exit(1)
            else:
                console.print(f"\n[green]All {len(skills)} skill(s) are valid.[/green]")

        @cli_app.command("info")
        def skill_info(
            name: Annotated[str, typer.Argument(help="Skill name to show info for.")],
        ) -> None:
            """Show full metadata and content for a skill."""
            from genai_tk.main.skills_manager import discover_all_skills

            skills = discover_all_skills(_project_dir())
            matches = [s for s in skills if s.name == name]
            if not matches:
                console.print(f"[red]Skill '{name}' not found.[/red]")
                console.print("  Run [bold]cli skills list[/bold] to see available skills.")
                raise typer.Exit(1)

            s = matches[0]
            console.print(f"\n[bold cyan]{s.name}[/bold cyan]")
            console.print(f"  Description: {s.description or '[dim]—[/dim]'}")
            console.print(f"  Source:      {s.display_source}")
            console.print(f"  Path:        {s.path}")
            if s.version:
                console.print(f"  Version:     {s.version}")
            if s.author:
                console.print(f"  Author:      {s.author}")
            if s.tags:
                console.print(f"  Tags:        {', '.join(s.tags)}")
            if s.repo:
                console.print(f"  Repo:        {s.repo}")
            if s.git_ref:
                console.print(f"  Git ref:     {s.git_ref}")

            skill_file = s.path / "SKILL.md"
            if skill_file.exists():
                console.print(f"\n[dim]─── {skill_file} ───[/dim]")
                from rich.markdown import Markdown

                console.print(Markdown(skill_file.read_text(encoding="utf-8")))

        @cli_app.command("search")
        def search_skills(
            query: Annotated[str, typer.Argument(help="Search term (matches name, description, tags).")],
        ) -> None:
            """Search installed skills and suggest community sources."""
            from genai_tk.main.skills_manager import list_bundled_skills

            q = query.lower()

            # Search bundled
            bundled = list_bundled_skills()
            hits = [
                s for s in bundled if q in s.name.lower() or q in s.description.lower() or any(q in t for t in s.tags)
            ]

            if hits:
                table = Table(title=f"Skills matching '{query}'", border_style="dim")
                table.add_column("Name", style="bold cyan")
                table.add_column("Description")
                table.add_column("Source", style="dim")
                for s in hits:
                    table.add_row(s.name, s.description, s.display_source)
                console.print(table)
                console.print(f"\n  [dim]Install with:[/dim] cli skills add {hits[0].name}")
            else:
                console.print(f"[yellow]No local skills match '{query}'.[/yellow]")

            # Always suggest community sources
            console.print("\n[bold]Community skill sources:[/bold]")
            console.print(
                "  [cyan]cli skills add --skillssh langchain-ai/langchain-skills[/cyan]  "
                "[dim](LangChain official)[/dim]"
            )
            console.print("  [dim]Browse:[/dim] https://www.skills.sh/topic/agent-workflows")
            console.print("  [dim]Install any:[/dim] cli skills add --skillssh <owner/repo>")
            console.print("  [dim]Or directly:[/dim] npx skills add <owner/repo>")


# ---------------------------------------------------------------------------
# Internal helpers (keep the command functions readable)
# ---------------------------------------------------------------------------


def _add_bundled(name: str, project_dir: Path, manifest: SkillManifest, force: bool) -> None:
    from genai_tk.main.skills_manager import install_from_bundled, list_bundled_skills

    dest_root = project_dir / "skills" / "bundled"
    dest_root.mkdir(parents=True, exist_ok=True)

    existing = manifest.get(name)
    if existing and not force:
        console.print(
            f"[yellow]Skill '{name}' already installed at {existing.path}.[/yellow]  Use --force to reinstall."
        )
        return

    # Check it exists
    available = {s.name for s in list_bundled_skills()}
    if name not in available:
        console.print(f"[red]Bundled skill '{name}' not found.[/red]")
        console.print(f"  Available: {', '.join(sorted(available)) or 'none'}")
        console.print("  For community skills: [bold]cli skills add --skillssh langchain-ai/langchain-skills[/bold]")
        raise typer.Exit(1)

    console.print(f"[cyan]Installing bundled skill '{name}'...[/cyan]")
    skill = install_from_bundled(name, dest_root)
    manifest.upsert(skill)
    console.print(f"[green]✓ Installed '{skill.name}'[/green] → {skill.path}")


def _add_from_git(
    repo_url: str, ref: str, subpath: str, project_dir: Path, manifest: SkillManifest, force: bool
) -> None:
    from genai_tk.main.skills_manager import install_from_git

    community_dir = _skills_community_dir(project_dir)
    skill_name = Path(subpath).name if subpath else Path(repo_url.rstrip("/")).stem

    if manifest.get(skill_name) and not force:
        console.print(f"[yellow]Skill '{skill_name}' already installed.[/yellow]  Use --force to reinstall.")
        return

    console.print(f"[cyan]Installing skill from git: {repo_url}[/cyan]")
    if ref:
        console.print(f"  ref: {ref}")
    if subpath:
        console.print(f"  path: {subpath}")

    try:
        skill = install_from_git(repo_url, community_dir, ref=ref, subpath=subpath)
        manifest.upsert(skill)
        console.print(f"[green]✓ Installed '{skill.name}'[/green] → {skill.path}")
        if skill.git_ref:
            console.print(f"  [dim]pinned @ {skill.git_ref}[/dim]")
    except Exception as exc:
        console.print(f"[red]Install failed:[/red] {exc}")
        raise typer.Exit(1) from exc


def _add_from_skillssh(owner_repo: str, project_dir: Path, manifest: SkillManifest) -> None:
    from genai_tk.main.skills_manager import install_from_skillssh

    console.print(f"[cyan]Installing skills from skills.sh: {owner_repo}[/cyan]")
    console.print("  [dim](running npx skills add — requires Node.js)[/dim]")

    try:
        skills = install_from_skillssh(owner_repo, project_dir / "skills")
        for s in skills:
            manifest.upsert(s)
            console.print(f"[green]✓ Installed '{s.name}'[/green] → {s.path}")
        console.print(f"\n[green]{len(skills)} skill(s) installed from {owner_repo}[/green]")
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(
            f"\n[yellow]Alternative:[/yellow] Install directly with npx:\n"
            f"  [bold]npx skills add {owner_repo}[/bold]\n"
            f"Then copy resulting .md files into [bold]skills/community/<skill-name>/SKILL.md[/bold]"
        )
        raise typer.Exit(1) from exc
    except ValueError as exc:
        console.print(f"[yellow]Warning:[/yellow] {exc}")
