# GenAI Toolkit — development justfile
# Standalone: no external genai-* project dependencies.
# Downstream projects import shared recipes via `tk.just`.

set dotenv-load
set dotenv-path := "~/.env"
set shell := ["sh", "-eu", "-c"]
set script-interpreter := ['uv', 'run', '--script']
set positional-arguments

pkg_name := "genai_tk"
streamlit_entry := "genai_tk/webapp/main/streamlit.py"
deer_flow_repo := "https://github.com/bytedance/deer-flow.git"
deer_flow_dir := "ext/deer-flow"

# Docker overrides for genai-tk itself (all extras, with Node.js)
app_name     := "genai-tk"
docker_tag   := "latest"
extras       := "all,monitoring"
install_node := "true"
extra_copy   := "skills"
docker_pkg   := pkg_name
docker_port  := "8501"
dockerfile   := "deploy/Dockerfile"

import 'deploy/docker.just'

# List available recipes
[group('General')]
[doc('Show all available recipes')]
default:
    @just --list --unsorted

# ─── Install ────────────────────────────────────────────────────────────────

[group('Install')]
[doc('Check if uv is installed, install if missing')]
check-uv:
    @if command -v uv >/dev/null 2>&1; then \
        echo "uv is already installed"; \
    else \
        echo "Installing uv..."; \
        curl -LsSf https://astral.sh/uv/install.sh | sh; \
    fi

[group('Install')]
[doc('Install package (no dev dependencies)')]
install: check-uv
    uv sync --no-dev

[group('Install')]
[doc('Install with development dependencies')]
install-dev: check-uv
    uv sync

# ─── Scaffolding ─────────────────────────────────────────────────────────────

[group('Scaffolding')]
[doc('Scaffold a new genai-tk project at <path>  e.g: just new ../my-app')]
new path source="git+https://github.com/tclatos/genai-tk@main" editable="false": check-uv
    #!/usr/bin/env sh
    set -eu

    target="{{ path }}"
    src="{{ source }}"
    mode="{{ editable }}"
    repo="{{ justfile_directory() }}"

    # If the target already exists, confirm deletion before recreating.
    if [ -e "$target" ]; then
        printf 'Path already exists: %s\nDelete it and recreate? [y/N] ' "$target"
        read -r ans
        case "$ans" in
            y|Y|yes) ;;
            *) echo "Aborted."; exit 1 ;;
        esac
        rm -rf "$target"
    fi

    mkdir -p "$target"
    cd "$target"
    abs="$(pwd)"
    name="$(basename "$abs")"

    echo "Scaffolding genai-tk project at: $abs"
    echo "Project name: $name   |   genai-tk source: $src (editable: $mode)"
    echo

    # Warn if the target is inside this repo (a uv workspace root): the new
    # project would become a workspace member rather than a standalone app.
    case "$abs" in
        "$repo"|"$repo"/*)
            echo "⚠️  Target is inside the genai-tk repo ($repo)."
            echo "    This repo is a uv workspace root, so 'uv init' creates a workspace member"
            echo "    rather than a standalone project. For a standalone app, scaffold outside"
            echo "    the repo, e.g: just new ../$name"
            echo
            ;;
    esac

    # 1. Bare uv project (no README, no .python-version, no main.py stub).
    uv init --bare --no-readme --no-pin-python

    # 2. Add genai-tk. Default: from git. For a fast local-dev link, override:
    #       just new ../my-app /home/tcl/prj/genai-tk true
    if [ "$mode" = "true" ]; then
        uv add --editable "$src"
    else
        uv add "$src"
    fi

    # 3. Scaffold config/, package, skills, justfile, AGENTS.md … (runs uv sync).
    uv run cli init --name "$name"

    # 4. Final sync so the generated package itself is installed.
    uv sync

    # 5. Verify the CLI loads in the new project.
    echo
    echo "Verifying the scaffolded project…"
    uv run cli --help >/dev/null
    echo "✓ uv run cli works"

    # Bonus smoke test: a fake-model LLM call (no API key needed). Non-fatal.
    if uv run cli core llm -i "tell me a joke" -m parrot_local@fake >/dev/null 2>&1; then
        echo "✓ fake-model LLM call works"
    else
        echo "ℹ fake-model smoke test skipped (non-fatal)"
    fi

    echo
    echo "✓ Done. Project ready at: $abs"
    echo "Next: cd $abs && just"

# ─── Code Quality ───────────────────────────────────────────────────────────

[group('Code Quality')]
[doc('Format and lint code with ruff')]
lint:
    uv run ruff format .
    uv run ruff check --select I --fix .
    uv run ruff check --fix --exclude {{ pkg_name }}/wip {{ pkg_name }}

[group('Code Quality')]
[doc('Format code with ruff (imports + style)')]
fmt:
    uv run ruff format .
    uv run ruff check --select I --fix .

[group('Code Quality')]
[doc('Ruff check without auto-fix (CI-safe)')]
quality:
    @echo "Checking {{ pkg_name }} (excluding .venv and wip)..."
    uv run ruff check --exclude .venv --exclude {{ pkg_name }}/wip .

[group('Code Quality')]
[doc('Run fmt + lint + test sequentially')]
check: fmt lint test

# ─── Testing ────────────────────────────────────────────────────────────────

[group('Testing')]
[doc('Run unit tests only')]
test-unit *args:
    uv run pytest tests/unit_tests/ -v {{ args }}

[group('Testing')]
[doc('Run unit + integration tests (no LLM/API keys required)')]
test *args:
    uv run pytest tests/unit_tests/ tests/integration_tests/ -v {{ args }}

[group('Testing')]
[doc('Run integration tests with real LLM API calls')]
test-full *args:
    uv run pytest tests/integration_tests/ --include-real-models -v {{ args }}

[group('Testing')]
[doc('Run all tests including real LLM calls')]
test-all *args:
    uv run pytest tests/unit_tests/ tests/integration_tests/ --include-real-models -m 'not slow' -v {{ args }}

# Delegated to cli test — these need config-path resolution, marker logic, or notebook execution:

[group('Testing')]
[doc('Run eval tests  (--real for LLM-judged, --deerflow for DeerFlow suite)')]
test-evals *args:
    uv run cli test evals {{ args }}

[group('Testing')]
[doc('Run tests matching a pattern across all test dirs  e.g: just test-select rag')]
test-select pattern *args:
    uv run cli test select '{{ pattern }}' {{ args }}

[group('Testing')]
[doc('Execute Jupyter notebooks as tests')]
test-notebooks *args:
    uv run cli test notebooks {{ args }}

[group('Testing')]
[doc('Run pytest with custom args  e.g: just pytest -k my_test -v')]
pytest *args:
    uv run pytest {{ args }}

[group('Testing')]
[doc('Quick smoke-test: verify basic package imports')]
test-install:
    echo "Testing {{ pkg_name }} imports..."
    uv run python -c "import genai_tk.core;  print('ok genai_tk.core')"
    uv run python -c "import genai_tk.extra; print('ok genai_tk.extra')"
    uv run python -c "import genai_tk.utils; print('ok genai_tk.utils')"
    printf '\033[3m\033[36mExpected output: '"'"'Human: Tell me a joke '"'"'\033[0m\n'
    uv run cli core llm -i "tell me a joke" -m parrot_local@fake

# ─── Skills ─────────────────────────────────────────────────────────────────

[group('Skills')]
[doc('List all available skills')]
skills:
    uv run cli skills list

[group('Skills')]
[doc('Validate all skills')]
lint-skills:
    uv run cli skills validate --all

# ─── Web Interface ──────────────────────────────────────────────────────────

[group('Web')]
[doc('Launch built-in agent demo webapp (Streamlit)')]
webapp:
    uv run python -m streamlit run "{{ streamlit_entry }}"

# ─── Monitoring ─────────────────────────────────────────────────────────────

[group('Monitoring')]
[doc('Show monitoring backend status (active backends, keys, log file)')]
monitoring-status:
    uv run cli monitoring status

[group('Monitoring')]
[doc('Enable monitoring for this project (writes .genai_tk state file)')]
monitoring-start backends="":
    uv run cli monitoring start {{ backends }}

[group('Monitoring')]
[doc('Disable monitoring for this project (clears .genai_tk state entry)')]
monitoring-stop:
    uv run cli monitoring stop

[group('Monitoring')]
[doc('Open monitoring backend UI in browser  (monitoring-open langfuse|langsmith)')]
monitoring-open backend="langfuse":
    uv run cli monitoring open {{ backend }}

[group('Monitoring')]
[doc('Open the latest trace in the browser  (monitoring-open-trace langfuse|langsmith)')]
monitoring-open-trace backend="langfuse":
    uv run cli monitoring open {{ backend }} --trace

[group('Monitoring')]
[doc('Tail the last 30 entries from the local JSONL trace log')]
monitoring-tail n="30":
    uv run cli monitoring tail --n {{ n }}

[group('Monitoring')]
[doc('Clear the local JSONL trace log (with confirmation)')]
monitoring-clear:
    uv run cli monitoring clear

# ─── LangFuse server (Docker) ────────────────────────────────────────────────

[private]
_compose_file := `python3 -c "from pathlib import Path; p = Path('deploy/docker-compose.langfuse.yaml'); print(p if p.exists() else '')" 2>/dev/null || echo "deploy/docker-compose.langfuse.yaml"`

[group('LangFuse')]
[doc('Start self-hosted LangFuse via Docker Compose')]
langfuse-server-start:
    docker compose -f deploy/docker-compose.langfuse.yaml up -d

[group('LangFuse')]
[doc('Stop self-hosted LangFuse Docker Compose')]
langfuse-server-stop:
    docker compose -f deploy/docker-compose.langfuse.yaml down

[group('LangFuse')]
[doc('Show LangFuse Docker Compose service status')]
langfuse-server-status:
    docker compose -f deploy/docker-compose.langfuse.yaml ps

# ─── Deer-flow ──────────────────────────────────────────────────────────────

[group('DeerFlow')]
[doc('Install DeerFlow harness and harnessing extras')]
deer-flow-install:
    uv sync --extra harnessing
    @echo ""
    @echo "✓ DeerFlow (deerflow-harness) installed via harnessing extra."
    @echo "  Verify with: cli agents deerflow --list"

# ─── Maintenance ────────────────────────────────────────────────────────────

[group('Maintenance')]
[doc('Clean Python bytecode and cache files')]
clean:
    uv cache prune
    find . \( -name "*.py[co]" -o -name "__pycache__" \
             -o -name ".ruff_cache" -o -name ".mypy_cache" \
             -o -name ".pytest_cache" \) \
        -exec rm -rf {} + 2>/dev/null || true

[group('Maintenance')]
[doc('Clear Jupyter notebook outputs')]
clean-notebooks:
    find . -path "./.venv" -prune -o -name "*.ipynb" -print \
        | while read -r nb; do \
            echo "Cleaning: $nb"; \
            uv run --with nbconvert python -m nbconvert --clear-output --inplace "$nb"; \
          done

[group('Maintenance')]
[confirm("This will modify ~/.bash_history. Continue?")]
[doc('Remove duplicates and noise from ~/.bash_history')]
clean-history:
    [ -f ~/.bash_history ] \
        && awk '!/^(ls|cat|hgrep|h|cd|p|m|ll|pwd|code|mkdir|export|rmdir|uv tree|make|just)( |$)/ \
              && !seen[$0]++' ~/.bash_history > ~/.bash_history_unique \
        && mv ~/.bash_history_unique ~/.bash_history \
        && echo "Done. Run 'history -c; history -r' to reload." \
        || echo "No ~/.bash_history found"

[group('Maintenance')]
[doc('Demo script: prints a message via the sh library')]
[script]
hello *args:
    # /// script
    # requires-python = ">=3.11"
    # dependencies=["sh"]
    # ///

    import sys, sh
    args = sys.argv[1:]
    print(sh.echo(f"Hello from Python! {' '.join(args)}"), end='')
