"""CLI commands for information and utility operations.

This module provides command-line interface commands for:
- Displaying system information (config, models, tools)
- Listing directory contents with pattern matching
- Managing command registry
- MCP tools inspection

The commands are registered with a Typer CLI application and provide:
- Configuration display and validation
- Model and component listing
- Directory listing with glob pattern support
- Path resolution utilities
"""

import os
from typing import Annotated

import typer
from typer import Option

from genai_tk.main.cli import CliTopCommand
from genai_tk.utils.config_mngr import global_config


class InfoCommands(CliTopCommand):
    """Information and listing commands."""

    description: str = "Information and listing commands."

    def get_description(self) -> tuple[str, str]:
        return "info", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command("config")
        def config() -> None:
            """
            Display current configuration, available LLM tags, and API keys.
            """

            from langsmith import utils as ls_utils
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table

            from genai_tk.core.embeddings_factory import EmbeddingsFactory
            from genai_tk.core.embeddings_store import EmbeddingsStore
            from genai_tk.core.llm_factory import PROVIDER_INFO, LlmFactory

            console = Console()

            console.print(
                Panel(f"[bold blue]Selected configuration:[/bold blue] {global_config().selected_config}", expand=False)
            )

            # Default models info — resolve from config without instantiating (avoids validation errors)
            llm_models_config = global_config().get("llm.models", {})
            default_llm_id = llm_models_config.get("default", "—") if llm_models_config else "—"
            try:
                default_embeddings = EmbeddingsFactory(embeddings=None)
                default_embeddings_id = str(default_embeddings.embeddings_id)
            except Exception:
                default_embeddings_id = str(global_config().get("embeddings.models.default", "—"))
            try:
                default_vector_store = EmbeddingsStore.create_from_config("default")
                default_vector_id = str(default_vector_store.backend)
            except Exception:
                default_vector_id = str(global_config().get("vector_store.default", "—"))

            models_table = Table(title="Default Components", show_header=True, header_style="bold magenta")
            models_table.add_column("Type", style="cyan")
            models_table.add_column("Model ID", style="green")

            models_table.add_row("LLM", str(default_llm_id))
            models_table.add_row("Embeddings", default_embeddings_id)
            models_table.add_row("Vector-store", default_vector_id)

            console.print(models_table)

            # LLM Tags info with enhanced details
            tags_table = Table(
                title="🏷️  LLM Tags (Use these with --llm option)", show_header=True, header_style="bold magenta"
            )
            tags_table.add_column("Tag", style="cyan", width=15)
            tags_table.add_column("LLM ID", style="green", width=25)
            tags_table.add_column("Provider", style="blue", width=12)
            tags_table.add_column("Status", style="yellow", width=12)
            tags_table.add_column("Usage Example", style="dim white", width=30)

            # Get all LLM tags from config under llm.models.*
            llm_models_config = global_config().get("llm.models", {})
            tag_count = 0
            # Handle both regular dict and OmegaConf DictConfig
            if llm_models_config and hasattr(llm_models_config, "items"):
                for tag, llm_id in llm_models_config.items():
                    if tag != "default":  # Skip the default entry as it's shown above
                        # Check if the LLM ID is available (has API keys and module)
                        is_available = llm_id in LlmFactory.known_items()
                        status = "[green]✓ available[/green]" if is_available else "[red]✗ unavailable[/red]"

                        # Extract provider from LLM ID (part after @ separator)
                        provider = "unknown"
                        if isinstance(llm_id, str) and "@" in llm_id:
                            provider = llm_id.split("@")[1]

                        # Create usage example
                        example = f"--llm {tag}"

                        tags_table.add_row(tag, str(llm_id), provider, status, example)
                        tag_count += 1

            if tag_count == 0:
                tags_table.add_row(
                    "[dim]No tags configured[/dim]",
                    "[dim]N/A[/dim]",
                    "[dim]N/A[/dim]",
                    "[dim]N/A[/dim]",
                    "[dim]Configure in config file[/dim]",
                )

            console.print(tags_table)

            # Add helpful usage information
            if tag_count > 0:
                console.print(
                    Panel(
                        "[bold cyan]💡 Usage Tips:[/bold cyan]\n"
                        "• Use tags with [bold]--llm[/bold] option: [green]uv run cli llm 'Hello' --llm fast_model[/green]\n"
                        "• Tags are easier to remember than full LLM IDs\n"
                        "• Configure more tags in your configuration file under [bold]llm.models[/bold]",
                        title="How to use LLM Tags",
                        border_style="cyan",
                        expand=False,
                    )
                )

            # API keys info
            keys_table = Table(title="Available API Keys", show_header=True, header_style="bold magenta")
            keys_table.add_column("Provider", style="cyan")
            keys_table.add_column("Environment Variable", style="green")
            keys_table.add_column("Status", style="yellow")

            for provider, provider_info in PROVIDER_INFO.items():
                key_name = provider_info.api_key_env_var
                if key_name:
                    status = "[green]✓ set[/green]" if key_name in os.environ else "[red]✗ not set[/red]"
                    keys_table.add_row(provider, key_name, status)

            console.print(keys_table)

            # KV Store info
            from genai_tk.extra.kv_store_registry import KvStoreRegistry

            kv_registry = KvStoreRegistry()
            try:
                available_stores = kv_registry.get_available_stores()

                kv_stores_table = Table(title="🗄️  Available KV Stores", show_header=True, header_style="bold magenta")
                kv_stores_table.add_column("Store ID", style="cyan", width=15)
                kv_stores_table.add_column("Type", style="green", width=20)
                kv_stores_table.add_column("Configuration", style="blue", width=30)
                kv_stores_table.add_column("Status", style="yellow", width=15)

                for store_id in available_stores:
                    try:
                        # Try to get configuration details for each store
                        config_info = global_config().get(f"kv_store.{store_id}", {})

                        # Handle different configuration formats
                        if hasattr(config_info, "get") and "type" in config_info:
                            # New format with explicit type
                            store_type = str(config_info["type"])
                            path_info = config_info.get("path", "N/A")
                            # Truncate long paths for display
                            if isinstance(path_info, str) and len(path_info) > 25:
                                path_info = f"...{path_info[-22:]}"
                            config_display = f"path: {path_info}"
                        elif hasattr(config_info, "get") and "path" in config_info:
                            # Legacy format - infer type
                            path_info = str(config_info["path"])
                            if store_id == "sql" or ("postgresql://" in path_info or "sqlite://" in path_info):
                                store_type = "SQLStore"
                            else:
                                store_type = "LocalFileStore"
                            # Truncate long paths for display
                            if len(path_info) > 25:
                                path_info = f"...{path_info[-22:]}"
                            config_display = f"path: {path_info}"
                        else:
                            # Handle special cases like OmegaConf objects
                            config_str = str(config_info)
                            if "LocalFileStore" in config_str:
                                store_type = "LocalFileStore"
                            elif "SQLStore" in config_str:
                                store_type = "SQLStore"
                            else:
                                store_type = "Unknown"

                            # Try to extract path from string representation
                            if "path" in config_str:
                                import re

                                path_match = re.search(r"'path':\s*'([^']+)'", config_str)
                                if path_match:
                                    path_info = path_match.group(1)
                                    if len(path_info) > 25:
                                        path_info = f"...{path_info[-22:]}"
                                    config_display = f"path: {path_info}"
                                else:
                                    config_display = config_str[:30] + ("..." if len(config_str) > 30 else "")
                            else:
                                config_display = config_str[:30] + ("..." if len(config_str) > 30 else "")

                        # Test if store can be created (indicates proper configuration)
                        try:
                            kv_registry.get(store_id)
                            status = "[green]✓ available[/green]"
                        except Exception as e:
                            error_msg = str(e)
                            if len(error_msg) > 20:
                                error_msg = f"{error_msg[:17]}..."
                            status = "[red]✗ error[/red]"

                        kv_stores_table.add_row(store_id, store_type, config_display, status)

                    except Exception as e:
                        # Handle individual store errors
                        kv_stores_table.add_row(store_id, "Error", str(e)[:25], "[red]✗ error[/red]")

                if not available_stores:
                    kv_stores_table.add_row(
                        "[dim]No stores configured[/dim]",
                        "[dim]N/A[/dim]",
                        "[dim]Configure in config file[/dim]",
                        "[dim]N/A[/dim]",
                    )

                console.print(kv_stores_table)

            except Exception as e:
                console.print(f"[yellow]Warning: Could not load KV store information: {e}[/yellow]")

            console.print(f"LangSmith Tracing: {ls_utils.tracing_is_enabled()}")

        @cli_app.command("models")
        def models() -> None:
            """
            List the known LLMs, embeddings models, and vector stores.
            """
            import os
            from collections import Counter

            from rich.columns import Columns
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text

            from genai_tk.core.embeddings_factory import EmbeddingsFactory
            from genai_tk.core.embeddings_store import EmbeddingsStore
            from genai_tk.core.llm_factory import LlmFactory
            from genai_tk.core.providers import PROVIDER_INFO

            console = Console()

            # --- LLM providers table ---
            llm_items_dict = LlmFactory.known_items_dict()
            provider_counts: Counter[str] = Counter(info.provider for info in llm_items_dict.values())

            prov_table = Table(show_header=True, header_style="bold blue", box=None, padding=(0, 2))
            prov_table.add_column("Provider", style="cyan", no_wrap=True)
            prov_table.add_column("Type", style="white", no_wrap=True)
            prov_table.add_column("API Key", no_wrap=True)
            prov_table.add_column("Models", style="dim white", justify="right", no_wrap=True)

            for provider_id in sorted(provider_counts):
                pinfo = PROVIDER_INFO.get(provider_id)
                type_str = (
                    Text("gateway", style="dim cyan") if pinfo and pinfo.gateway else Text("direct", style="dim green")
                )
                if pinfo and pinfo.api_key_env_var:
                    key_name = pinfo.api_key_env_var
                    key_set = bool(os.environ.get(key_name))
                    key_str = Text(f"✓ {key_name}", style="green") if key_set else Text(f"✗ {key_name}", style="red")
                else:
                    key_str = Text("not required", style="dim")
                prov_table.add_row(provider_id, type_str, key_str, str(provider_counts[provider_id]))

            llm_panel = Panel(prov_table, title="[bold blue]LLM Providers[/bold blue]", border_style="blue")

            # --- Embeddings & Vector stores as compact bullet lists ---
            embeddings_items = EmbeddingsFactory.known_items()
            vector_items = EmbeddingsStore.known_items()
            embeddings_content = Columns([f"• {item}" for item in embeddings_items], equal=True, expand=True)
            vector_content = Columns([f"• {item}" for item in vector_items], equal=True, expand=True)
            embeddings_panel = Panel(
                embeddings_content, title="[bold green]Embeddings[/bold green]", border_style="green"
            )
            vector_panel = Panel(
                vector_content, title="[bold magenta]Vector Stores[/bold magenta]", border_style="magenta"
            )

            console.print(llm_panel)
            console.print()
            console.print(Columns([embeddings_panel, vector_panel], equal=True, expand=True))

        @cli_app.command("llm-profile")
        def llm_profile(
            model_id: Annotated[
                str,
                typer.Argument(
                    help=(
                        "Model ID from registry (e.g. gpt_41mini@openai, gpt_oss120@openrouter) "
                        "or a raw provider model name (e.g. gpt-4o-mini)"
                    )
                ),
            ],
        ) -> None:
            """Display the LangChain ModelProfile for a model.

            Looks up profile data (context window, max output tokens, capability flags) from
            the models.dev database via the LangChain ModelProfile registry, then overlays
            any overrides defined in the local llm.yaml configuration.

            Examples:
                ```bash
                uv run cli info llm-profile gpt_41mini@openai
                uv run cli info llm-profile gpt_oss120@openrouter
                uv run cli info llm-profile gpt-4o-mini
                ```
            """
            from rich.console import Console, Group
            from rich.panel import Panel
            from rich.rule import Rule
            from rich.table import Table
            from rich.text import Text

            from genai_tk.core.llm_factory import LlmFactory, lookup_model_entry, resolve_model
            from genai_tk.core.providers import PROVIDER_INFO

            console = Console()

            # --- Resolve LlmInfo: exact match first, then fuzzy ---
            llm_info = None
            fuzzy_alternatives: list[tuple[str, float]] = []
            resolved_canonical: str | None = None

            if "@" in model_id:
                # Step 1: exact match in known items (exceptions + registry)
                llm_info = LlmFactory.known_items_dict().get(model_id)

                # Step 2: fuzzy resolution for compact aliases like haiku45@anthropic
                if llm_info is None:
                    compact, _, provider_id = model_id.rpartition("@")
                    try:
                        from genai_tk.core.llm_factory import LlmInfo

                        canon, _fp, fuzzy_alternatives = resolve_model(compact, provider_id)
                        resolved_canonical = canon
                        best_score = fuzzy_alternatives[0][1] if fuzzy_alternatives else 0.0
                        if best_score < 0.6:
                            top3 = [name for name, _ in fuzzy_alternatives[:3]]
                            console.print(
                                f"[yellow]⚠ Low-confidence resolution: '{model_id}' → '{canon}@{provider_id}' "
                                f"(score {best_score:.2f}). Did you mean one of: {top3}?[/yellow]"
                            )
                        llm_info = LlmFactory.known_items_dict().get(f"{canon}@{provider_id}")
                        if llm_info is None:
                            llm_info = LlmInfo(id=model_id, provider=provider_id, model=canon)
                    except ValueError as e:
                        console.print(f"[yellow]Could not fuzzy-resolve '{model_id}': {e}[/yellow]")

            # --- Determine model_name / provider for profile lookup ---
            if llm_info is not None:
                lc_model_name = llm_info.model
                lc_provider = llm_info.provider
            elif "@" in model_id:
                model_part, _, provider_part = model_id.rpartition("@")
                lc_model_name = model_part
                lc_provider = provider_part
            else:
                lc_model_name = model_id
                lc_provider = "openai"

            profile = lookup_model_entry(lc_model_name, lc_provider)

            # --- Header ---
            console.print()
            console.print(Rule(f"[bold cyan]{model_id}[/bold cyan]", style="cyan"))
            console.print()

            # --- Left table: llm.yaml registry info ---
            reg_table = Table(
                title="[bold blue]llm.yaml[/bold blue]",
                show_header=True,
                header_style="bold blue",
                box=None,
                padding=(0, 1),
            )
            reg_table.add_column("Field", style="dim", no_wrap=True, min_width=12)
            reg_table.add_column("Value", style="white", min_width=26)
            if llm_info is not None:
                reg_table.add_row("ID", f"[cyan]{llm_info.id}[/cyan]")
                reg_table.add_row("Provider", llm_info.provider)
                reg_table.add_row("Model", llm_info.model)
                caps_yaml = ", ".join(llm_info.capabilities) if llm_info.capabilities else "[dim]—[/dim]"
                reg_table.add_row("Capabilities", caps_yaml)
                reg_table.add_row("Max tokens", str(llm_info.max_tokens) if llm_info.max_tokens else "[dim]—[/dim]")
                reg_table.add_row(
                    "Context", str(llm_info.context_window) if llm_info.context_window else "[dim]—[/dim]"
                )
            else:
                reg_table.add_row("ID", f"[dim]{model_id}[/dim]")
                reg_table.add_row("Provider", lc_provider)
                reg_table.add_row("Model", lc_model_name)
                reg_table.add_row("Capabilities", "[dim]—[/dim]")
                reg_table.add_row("Max tokens", "[dim]—[/dim]")
                reg_table.add_row("Context", "[dim]—[/dim]")

            # Append effective values to left column when there's a profile
            left_renderables: list = [reg_table]
            if llm_info is not None and profile is not None:
                eff_table = Table(
                    title="[bold magenta]Effective[/bold magenta]",
                    show_header=True,
                    header_style="bold magenta",
                    box=None,
                    padding=(0, 1),
                )
                eff_table.add_column("Field", style="dim", no_wrap=True, min_width=12)
                eff_table.add_column("Value", style="white", min_width=26)
                eff_table.add_row("Capabilities", ", ".join(llm_info.effective_capabilities) or "[dim]—[/dim]")
                eff_table.add_row(
                    "Context",
                    f"{llm_info.effective_context_window:,}" if llm_info.effective_context_window else "[dim]—[/dim]",
                )
                eff_table.add_row(
                    "Max tokens",
                    f"{llm_info.effective_max_tokens:,}" if llm_info.effective_max_tokens else "[dim]—[/dim]",
                )
                left_renderables.append(eff_table)

            # --- Right table: models.dev capabilities ---
            prof_table = Table(
                title="[bold green]models.dev[/bold green]",
                show_header=True,
                header_style="bold green",
                box=None,
                padding=(0, 1),
            )
            prof_table.add_column("Capability / Limit", style="dim", no_wrap=True, min_width=20)
            prof_table.add_column("Value", style="white", min_width=10, no_wrap=True)

            if profile:
                prof_table.add_row(
                    "Context window",
                    f"[yellow]{profile.context:,}[/yellow]" if profile.context else "[dim]—[/dim]",
                )
                prof_table.add_row(
                    "Max output tokens",
                    f"[yellow]{profile.output:,}[/yellow]" if profile.output else "[dim]—[/dim]",
                )
                bool_fields = [
                    ("has_vision", "Vision / image"),
                    ("has_thinking", "Reasoning"),
                    ("has_structured_outputs", "Structured output"),
                    ("has_pdf", "PDF"),
                    ("has_audio", "Audio"),
                    ("has_video", "Video"),
                    ("tool_call", "Tool calling"),
                    ("open_weights", "Open weights"),
                ]
                for attr, label in bool_fields:
                    val = getattr(profile, attr)
                    prof_table.add_row(label, "[green]✓[/green]" if val else "[dim]✗[/dim]")
                if profile.cost_input is not None:
                    prof_table.add_row("Cost in ($/M tok)", f"{profile.cost_input}")
                if profile.cost_output is not None:
                    prof_table.add_row("Cost out ($/M tok)", f"{profile.cost_output}")
            else:
                prof_table.add_row("[dim italic]no entry in models.dev[/dim italic]", "")

            # --- Middle table: provider info ---
            prov_table = Table(
                title="[bold yellow]Provider[/bold yellow]",
                show_header=True,
                header_style="bold yellow",
                box=None,
                padding=(0, 1),
            )
            prov_table.add_column("Field", style="dim", no_wrap=True, min_width=10)
            prov_table.add_column("Value", style="white", min_width=22, no_wrap=True)

            prov_info = PROVIDER_INFO.get(lc_provider)
            if prov_info:
                # Gateway flag
                prov_table.add_row(
                    "Type",
                    "[cyan]gateway[/cyan]" if prov_info.gateway else "[white]direct[/white]",
                )
                # API key availability
                key_var = prov_info.api_key_env_var
                if key_var == "":
                    key_status = "[dim]not required[/dim]"
                elif key_var in os.environ:
                    key_status = f"[green]✓[/green] [dim]{key_var}[/dim]"
                else:
                    key_status = f"[red]✗[/red] [dim]{key_var}[/dim]"
                prov_table.add_row("API key", key_status)
                # Module
                prov_table.add_row("Module", f"[dim]{prov_info.module}[/dim]")
                prov_table.add_row("Class", prov_info.langchain_class)
                # API base (only if overridden)
                if prov_info.api_base:
                    prov_table.add_row("API base", f"[dim]{prov_info.api_base}[/dim]")
                # LiteLLM prefix
                if prov_info.litellm_prefix:
                    prov_table.add_row("LiteLLM prefix", f"[dim]{prov_info.litellm_prefix}[/dim]")
            else:
                prov_table.add_row("[dim italic]unknown provider[/dim italic]", "")

            # --- Side-by-side: Group(reg + eff) | prov | prof ---
            side_grid = Table.grid(padding=(0, 4))
            side_grid.add_column()
            side_grid.add_column()
            side_grid.add_column()
            side_grid.add_row(Group(*left_renderables), prov_table, prof_table)
            console.print(side_grid)

            # --- Fuzzy Resolution (compact, full width below) ---
            if fuzzy_alternatives:
                console.print()
                fuzz_table = Table(
                    title=f"[bold yellow]Fuzzy Resolution[/bold yellow]  [dim]{model_id}[/dim]",
                    show_header=True,
                    header_style="bold yellow",
                    box=None,
                    padding=(0, 1),
                )
                fuzz_table.add_column("#", style="dim", width=2)
                fuzz_table.add_column("Canonical name", style="cyan")
                fuzz_table.add_column("Score", style="white", width=5)
                fuzz_table.add_column("", style="green", width=1)
                for rank, (cname, score) in enumerate(fuzzy_alternatives[:6], start=1):
                    selected = "[bold green]✓[/bold green]" if cname == resolved_canonical else ""
                    fuzz_table.add_row(str(rank), cname, f"{score:.2f}", selected)
                console.print(fuzz_table)

            # --- No profile fallback ---
            if not profile:
                console.print()
                no_profile_text = Text()
                no_profile_text.append("No entry in models.dev for ", style="yellow")
                no_profile_text.append(f"{lc_model_name!r}", style="bold")
                no_profile_text.append(f" (provider: {lc_provider}). ", style="dim")
                no_profile_text.append("Run ", style="dim")
                no_profile_text.append("cli info llm-refresh --reload", style="bold")
                no_profile_text.append(" to refresh, or add caps to llm.yaml.", style="dim")
                console.print(Panel(no_profile_text, border_style="yellow", expand=False))
                if llm_info is not None and llm_info.effective_capabilities:
                    console.print(
                        f"[cyan]Effective capabilities (yaml override):[/cyan] "
                        f"{', '.join(llm_info.effective_capabilities)}"
                    )

        @cli_app.command("llm-refresh")
        def llm_refresh(
            apply: bool = typer.Option(
                False, "--apply", help="Write changes to llm.yaml, removing redundant/outdated overrides"
            ),
            reload: bool = typer.Option(
                False, "--reload", help="Re-download models.dev database before running the review"
            ),
        ) -> None:
            """Inspect and refresh llm.yaml exceptions against the local models.dev database.

            For each exception entry, looks up the models.dev ModelEntry and reports whether
            yaml overrides are redundant (database already provides same data),
            outdated (database has more capabilities than yaml lists), or intentional
            (yaml value differs from database — kept as override).

            Use ``--reload`` to download the latest models.dev database before running.

            Examples:
                ```bash
                uv run cli info llm-refresh                    # report only
                uv run cli info llm-refresh --reload           # refresh DB then report
                uv run cli info llm-refresh --apply            # update llm.yaml
                uv run cli info llm-refresh --reload --apply   # both
                ```
            """
            import os
            from pathlib import Path

            from rich.console import Console
            from rich.rule import Rule
            from rich.table import Table
            from rich.text import Text
            from ruamel.yaml import YAML

            from genai_tk.core.llm_factory import lookup_model_entry
            from genai_tk.core.models_db import get_models_db, invalidate_models_db
            from genai_tk.core.providers import PROVIDER_INFO

            console = Console()

            console.print()
            console.print(Rule("[bold bright_blue]llm.yaml Exception Review[/bold bright_blue]", style="bright_blue"))

            if reload:
                console.print("\n[cyan]Reloading models.dev database…[/cyan]")
                invalidate_models_db()
                get_models_db().fetch()
                invalidate_models_db()  # reload from freshly saved file
                get_models_db()
                console.print("[green]✓ Database updated.[/green]")

            # DB summary
            db = get_models_db()
            provider_count = len(db._providers)
            model_count = len(db._index)
            console.print(
                f"\n[dim]models.dev local DB: [bold]{model_count:,}[/bold] models across "
                f"[bold]{provider_count}[/bold] providers[/dim]"
            )

            llm_yaml_path = Path(__file__).parent.parent.parent / "config" / "basic" / "providers" / "llm.yaml"

            ruamel = YAML()
            ruamel.preserve_quotes = True
            with open(llm_yaml_path, encoding="utf-8") as f:
                ruamel_data = ruamel.load(f)

            exceptions = ruamel_data.get("llm", {}).get("exceptions", [])

            table = Table(
                show_header=True,
                header_style="bold magenta",
                show_lines=True,
            )
            table.add_column("model_id", style="cyan", width=22, no_wrap=True)
            table.add_column("Provider", style="white", width=10, no_wrap=True)
            table.add_column("Key", width=22, no_wrap=True)
            table.add_column("Field", style="white", width=16)
            table.add_column("YAML value", style="dim white", width=26)
            table.add_column("Profile value", style="dim white", width=26)
            table.add_column("Action", width=12)

            # (entry_idx, field_name) for fields that --apply should delete
            removals: list[tuple[int, str]] = []

            for idx, entry in enumerate(exceptions):
                model_id = entry.get("model_id", "?")

                # Find first provider/model with a profile
                profile = None
                profile_provider = ""
                first_provider_id = ""
                for provider_item in entry.get("providers", []):
                    if not isinstance(provider_item, dict):
                        continue
                    for provider, model in provider_item.items():
                        if not isinstance(model, str):
                            continue
                        if not first_provider_id:
                            first_provider_id = provider
                        p = lookup_model_entry(model, provider)
                        if p:
                            profile = p
                            profile_provider = f"{provider}/{model}"
                            if not first_provider_id:
                                first_provider_id = provider
                            break
                    if profile:
                        break

                # Provider info columns (shown on first row of each entry)
                pinfo = PROVIDER_INFO.get(first_provider_id) if first_provider_id else None
                if pinfo:
                    ptype_str = (
                        Text("gateway", style="dim cyan") if pinfo.gateway else Text("direct", style="dim green")
                    )
                    if pinfo.api_key_env_var:
                        key_set = bool(os.environ.get(pinfo.api_key_env_var))
                        key_str = (
                            Text(f"✓ {pinfo.api_key_env_var}", style="green")
                            if key_set
                            else Text(f"✗ {pinfo.api_key_env_var}", style="red")
                        )
                    else:
                        key_str = Text("not required", style="dim")
                else:
                    ptype_str = Text(first_provider_id or "?", style="dim")
                    key_str = Text("?", style="dim")

                yaml_caps = list(entry.get("capabilities", []) or [])
                yaml_max_tokens = entry.get("max_tokens")
                yaml_context_window = entry.get("context_window")
                has_any_override = bool(yaml_caps or yaml_max_tokens is not None or yaml_context_window is not None)

                if profile:
                    profile_caps = profile.capabilities
                    profile_max_out = profile.output
                    profile_max_in = profile.context

                    row_model = model_id
                    row_ptype = ptype_str
                    row_key = key_str

                    # --- capabilities ---
                    if yaml_caps:
                        yaml_set = set(yaml_caps)
                        prof_set = set(profile_caps)
                        caps_yaml_str = ", ".join(sorted(yaml_set))
                        caps_prof_str = ", ".join(sorted(prof_set)) if prof_set else "—"
                        if yaml_set == prof_set:
                            action = Text("REMOVE", style="bold green")
                            removals.append((idx, "capabilities"))
                        elif yaml_set.issubset(prof_set):
                            action = Text("OUTDATED", style="bold red")
                            removals.append((idx, "capabilities"))
                        else:
                            action = Text("OVERRIDE", style="bold yellow")
                        table.add_row(
                            row_model, row_ptype, row_key, "capabilities", caps_yaml_str, caps_prof_str, action
                        )
                        row_model = ""
                        row_ptype = ""
                        row_key = ""

                    # --- max_tokens ---
                    if yaml_max_tokens is not None:
                        prof_str = f"{profile_max_out:,}" if profile_max_out else "—"
                        yaml_str = f"{yaml_max_tokens:,}"
                        if profile_max_out and yaml_max_tokens == profile_max_out:
                            action = Text("REMOVE", style="bold green")
                            removals.append((idx, "max_tokens"))
                        else:
                            action = Text("OVERRIDE", style="bold yellow")
                        table.add_row(row_model, row_ptype, row_key, "max_tokens", yaml_str, prof_str, action)
                        row_model = ""
                        row_ptype = ""
                        row_key = ""

                    # --- context_window ---
                    if yaml_context_window is not None:
                        prof_str = f"{profile_max_in:,}" if profile_max_in else "—"
                        yaml_str = f"{yaml_context_window:,}"
                        if profile_max_in and yaml_context_window == profile_max_in:
                            action = Text("REMOVE", style="bold green")
                            removals.append((idx, "context_window"))
                        else:
                            action = Text("OVERRIDE", style="bold yellow")
                        table.add_row(row_model, row_ptype, row_key, "context_window", yaml_str, prof_str, action)
                        row_model = ""
                        row_ptype = ""
                        row_key = ""

                    if not has_any_override:
                        table.add_row(
                            model_id,
                            ptype_str,
                            key_str,
                            "—",
                            "—",
                            f"from {profile_provider}",
                            Text("OK", style="dim green"),
                        )
                else:
                    # No profile available — all yaml values are necessary
                    table.add_row(
                        model_id,
                        ptype_str,
                        key_str,
                        "all",
                        ", ".join(yaml_caps) or "—",
                        "[dim]no profile[/dim]",
                        Text("KEEP", style="dim"),
                    )

            console.print()
            console.print(table)

            if not removals:
                console.print(
                    "\n[green]✓ Nothing to remove — all overrides are intentional or no profile available.[/green]"
                )
                return

            total = len(removals)
            console.print(f"\n[bold]{total}[/bold] redundant/outdated field(s) found.")

            if not apply:
                console.print("[dim]Run with [bold]--apply[/bold] to remove them from llm.yaml.[/dim]")
                return

            # Apply: remove fields using ruamel (preserves all comments and formatting)
            for entry_idx, field in removals:
                entry = exceptions[entry_idx]
                if field in entry:
                    del entry[field]

            with open(llm_yaml_path, "w", encoding="utf-8") as f:
                ruamel.dump(ruamel_data, f)

            console.print(f"[bold green]✓ Removed {total} field(s) from {llm_yaml_path.name}[/bold green]")

        @cli_app.command("mcp-tools")
        def mcp_tools(
            filter: Annotated[list[str] | None, Option("--filter", "-f", help="Filter tools by server names")] = None,
        ) -> None:
            """Display information about available MCP tools.

            Shows the list of tools from MCP servers along with their descriptions.
            Can be filtered by server names.
            """
            import asyncio

            from rich.console import Console
            from rich.table import Table

            from genai_tk.core.mcp_client import get_mcp_tools_info

            async def display_tools():
                tools_info = await get_mcp_tools_info(filter)
                if not tools_info:
                    print("No MCP tools found.")
                    return

                console = Console()
                for server_name, tools in tools_info.items():
                    table = Table(
                        title=f"Server: {server_name}",
                        show_header=True,
                        header_style="bold magenta",
                        row_styles=["", "dim"],
                    )
                    table.add_column("Tool", style="cyan", no_wrap=True)
                    table.add_column("Description", style="green")

                    for tool_name, description in tools.items():
                        table.add_row(tool_name, description)

                    console.print(table)
                    print()  # Add space between tables

            asyncio.run(display_tools())

        @cli_app.command("commands")
        def commands() -> None:
            """Display all registered CLI commands and subcommands.

            Shows a hierarchical view of all available commands in the CLI,
            including their help text. Similar to --help but provides a complete overview.

            Example:
                uv run cli info commands
            """
            from genai_tk.cli.command_tree import display_command_tree

            display_command_tree(
                cli_app,
                title="[bold cyan]📋 CLI Command Structure[/bold cyan]",
            )

        @cli_app.command("ls")
        def ls(
            target_dir: Annotated[
                str,
                typer.Argument(
                    help=("Directory to list. Supports config variables like ${paths.data_root}"),
                ),
            ],
            include_patterns: Annotated[
                list[str] | None,
                typer.Option(
                    "--include",
                    "-i",
                    help="Glob patterns to include (e.g., '*.py', 'test_*.py'). Default: ['*']",
                ),
            ] = None,
            exclude_patterns: Annotated[
                list[str] | None,
                typer.Option(
                    "--exclude",
                    "-e",
                    help="Glob patterns to exclude (e.g., '__pycache__', '*.pyc')",
                ),
            ] = None,
            recursive: bool = typer.Option(False, "--recursive", "-r", help="List directories recursively"),
            show_hidden: bool = typer.Option(False, "--all", "-a", help="Include hidden files (starting with .)"),
            long_format: bool = typer.Option(False, "--long", "-l", help="Use long listing format with details"),
        ) -> None:
            """List directory contents with pattern matching and path resolution.

            This command lists files and directories with support for:
            - Configuration variable substitution (e.g., ${paths.data_root})
            - Include/exclude glob patterns
            - Recursive directory traversal
            - Hidden file filtering

            Primarily used to test path resolving and pattern matching capabilities.

            Examples:
                ```bash
                # List all files in a directory
                cli info ls ./src

                # Use config variables
                cli info ls '${paths.data_root}'

                # List only Python files recursively
                cli info ls ./src --include '*.py' --recursive

                # Exclude test and cache files
                cli info ls ./src --recursive \\
                    --include '*.py' \\
                    --exclude 'test_*.py' --exclude '__pycache__'

                # Long format with all files
                cli info ls ./src --long --all

                # Multiple include patterns
                cli info ls ./docs --include '*.md' --include '*.rst' --recursive
                ```
            """
            from pathlib import Path

            from loguru import logger
            from rich.console import Console
            from rich.table import Table

            from genai_tk.utils.file_patterns import resolve_entries

            console = Console()

            # Use resolve_entries to handle all pattern matching consistently (files and directories)
            try:
                files = resolve_entries(
                    target_dir,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                    recursive=recursive,
                    case_sensitive=False,
                    include_files=True,
                    include_directories=True,
                )
            except Exception as e:
                logger.error(f"Failed to resolve entries: {e}")
                raise typer.Exit(1) from e

            # Filter out hidden files if needed
            if not show_hidden:
                files = [f for f in files if not f.name.startswith(".")]

            # Convert to Path objects and sort
            all_entries = sorted([Path(f) for f in files])

            if not all_entries:
                logger.warning(f"No files or directories found matching patterns in {target_dir}")
                return

            # Get resolved directory for display
            from genai_tk.utils.file_patterns import resolve_config_path

            resolved_dir = resolve_config_path(target_dir)
            target_path = Path(resolved_dir)

            # Display results
            if long_format:
                # Long format with details
                table = Table(
                    title=f"📂 Directory listing: {resolved_dir}",
                    show_header=True,
                    header_style="bold magenta",
                )
                table.add_column("Type", style="cyan", width=8, no_wrap=True)
                table.add_column("Name", style="green")
                table.add_column("Size", style="yellow", justify="right")
                table.add_column("Modified", style="blue")

                for entry in all_entries:
                    import datetime

                    # Determine entry type
                    if entry.is_dir():
                        entry_type = "📁 DIR"
                        size = "-"
                    elif entry.is_file():
                        entry_type = "📄 FILE"
                        try:
                            size = f"{entry.stat().st_size:,}"
                        except OSError:
                            size = "?"
                    else:
                        entry_type = "❓ OTHER"
                        size = "?"

                    # Get modification time
                    try:
                        mtime = datetime.datetime.fromtimestamp(entry.stat().st_mtime)
                        mod_time = mtime.strftime("%Y-%m-%d %H:%M")
                    except OSError:
                        mod_time = "?"

                    table.add_row(
                        entry_type,
                        entry.name,
                        size,
                        mod_time,
                    )

                console.print(table)
                console.print(f"\n[bold]Total:[/bold] {len(all_entries)} entries")

            else:
                # Simple format - just names
                console.print(f"[bold cyan]📂 Directory:[/bold cyan] {resolved_dir}\n")

                for entry in all_entries:
                    # Show relative path if recursive, otherwise just name
                    if recursive:
                        try:
                            display_name = str(entry.relative_to(target_path))
                        except ValueError:
                            display_name = entry.name
                    else:
                        display_name = entry.name

                    # Add indicator for directories
                    if entry.is_dir():
                        console.print(f"[cyan]{display_name}/[/cyan]")
                    else:
                        console.print(display_name)

                console.print(f"\n[bold]Total:[/bold] {len(all_entries)} entries")
