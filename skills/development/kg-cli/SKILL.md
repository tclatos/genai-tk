---
name: kg-cli
description: Add or modify the genai-graph CLI command groups — cli kg (create/info/schema/agent/cypher/query/view), cli docgraph (run/build/list/toc/cat/search/tui), and cli neo4j (analyze/convert/subset/import/query/info). Use when editing genai_graph/core/commands_*.py or genai_graph/neo4j_import/commands.py, adding a KG CLI subcommand, or wiring a command to the workflow engine.
---

# GenAI Graph CLI

## Read First

- `docs/workflows.md` — CLI Reference (the authoritative command flags)
- `docs/document-graph.md` — `cli docgraph` reference
- `docs/graph_construction.md` — `cli kg` reference
- `genai_graph/core/commands_ekg.py` — `KgCommands` (`cli kg`)
- `genai_graph/core/commands_docgraph.py` — `DocGraphCommands` (`cli docgraph`)
- `genai_graph/neo4j_import/commands.py` — `Neo4jCommands` (`cli neo4j`)
- `config/app_conf.yaml` — `cli.commands` registration list

## Relationship to genai-tk

genai-graph command groups are `genai_tk.cli.base.CliTopCommand` subclasses, registered
through the same dynamic `cli.commands` config mechanism genai-tk uses (see
`genai-tk/cli-and-scaffolding`). Each implements `get_description()` (returns the command
name + help) and `register_sub_commands(cli_app)` (registers Typer subcommands). The `cli`
entry point itself comes from genai-tk.

## Command groups

### `cli kg` — `KgCommands` (`core/commands_ekg.py`)

```bash
cli kg create                          # default workflow profile
cli kg create my_graph                 # specific profile
cli kg create my_graph --force parquet # rebuild import caches
cli kg create my_graph --force all     # full clean rebuild
cli kg create my_graph --delete-first
cli kg create my_graph --dry-run
cli kg create --all                    # every kg_* profile

cli kg info                            # DB stats + artifact links
cli kg schema                          # node/relationship schema (--regen to rebuild)
cli kg schema --regen --kg my_graph
cli kg cypher "MATCH (n) RETURN labels(n), count(*)"
cli kg query "Which companies have the most projects?"   # text-to-Cypher
cli kg agent "..."                     # KG LangChain agent (see kg-query)
cli kg view                            # open the HTML graph visualization
```

`cli kg create <name>` wraps `cli workflow run <profile>`; CLI flags always override YAML
defaults. `--force <stage>`, `--delete-first`, `--no-export-html`, `--dry-run`, `--all` are
the main flags (see `kg-workflows` for force stages).

### `cli docgraph` — `DocGraphCommands` (`core/commands_docgraph.py`)

```bash
cli docgraph build ./docs --db ./data/kg/tree.db            # markdownize + document graph
cli docgraph build ./RFQ.zip --db ./data/kg/tree.db --profile fast
cli docgraph build ./docs --db ./data/kg/tree.db --force md # re-run markdown conversion

cli docgraph run --workflow rainbow_extract                # project entity-extraction workflow
cli docgraph run -w rainbow_extract -s ./some_file.pptx    # ad-hoc source override
cli docgraph run -w rainbow_extract --dry-run

cli docgraph list --db ./data/kg/tree.db
cli docgraph toc <filename-or-hash> --db ./data/kg/tree.db
cli docgraph cat <filename-or-hash> --db ./data/kg/tree.db
cli docgraph search "keyword" --db ./data/kg/tree.db
cli docgraph tui --db ./data/kg/tree.db
```

`cli docgraph build` always markdownizes first then ingests; `cli docgraph run` runs a
project's `docgraph_build`-based workflow against configured or ad-hoc sources. See
`kg-document-graph`.

### `cli neo4j` — `Neo4jCommands` (`neo4j_import/commands.py`)

```bash
cli neo4j analyze export.jsonl -o schema.cypher
cli neo4j convert export.jsonl ./ladybug_import
cli neo4j subset export.jsonl subset.jsonl --max-nodes 20 --max-rels 20
cli neo4j import export.jsonl --db path/to/ladybug.db -f
cli neo4j query "MATCH (n) RETURN labels(n), count(*)" --db path/to/ladybug.db
cli neo4j info --db path/to/ladybug.db
```

See `kg-neo4j-import`. All path arguments accept `${paths.*}` config variables.

## Add or modify a command

1. Edit the owning `*Commands` class — add a `@cli_app.command("name")` Typer function inside
   `register_sub_commands`.
2. Resolve YAML config variables in path args with
   `genai_tk.config_mgmt.file_patterns.resolve_config_path`.
3. Reuse existing primitives — `create_backend_from_config`, `get_kg_manager()`,
   `resolve_workflow_invocation` + `execute_workflow`, `create_kg_cypher_tool`,
   `document_graph_tools` — rather than re-implementing them.
4. Keep heavy imports inside the command function (lazy), not at module top, so `cli --help`
   stays fast.
5. Register a new top-level group by adding its `CliTopCommand` subclass to
   `config/app_conf.yaml` under `cli.commands`.

## Commands

```bash
uv run cli --help
uv run cli kg --help
uv run cli docgraph --help
uv run cli neo4j --help
uv run just test
```

## Avoid

- Do not hardcode command groups in a custom `cli` entry point — use the `cli.commands`
  config registration.
- Do not import heavy optional dependencies at module import time in a commands module.
- Do not re-implement workflow execution in a command — call `kg_create_step` /
  `docgraph_build_step` / `resolve_workflow_invocation`.
- Do not bypass `resolve_config_path` for path arguments that should support `${paths.*}`.

## Complements

- `genai-tk/cli-and-scaffolding` — `CliTopCommand`, dynamic registration, scaffolding.
- `kg-workflows` — the step functions `cli kg create` / `cli docgraph run` wrap.
- `kg-query` — primitives behind `cli kg query` / `cli kg agent` / `cli kg cypher`.
- `kg-neo4j-import` / `kg-document-graph` — the domains behind `cli neo4j` / `cli docgraph`.
