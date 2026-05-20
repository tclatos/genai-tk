# GenAI Toolkit — development justfile
# Standalone: no external genai-* project dependencies.
# Downstream projects import shared recipes via `tk.just`.

set dotenv-load
set shell := ["bash", "-euc"]
set script-interpreter := ['uv', 'run', '--script']
set positional-arguments
set tempdir := "/dev/shm"

pkg_name := "genai_tk"
streamlit_entry := "genai_tk/webapp/main/streamlit.py"
deer_flow_repo := "https://github.com/bytedance/deer-flow.git"
deer_flow_dir := "ext/deer-flow"

# List available recipes
[doc('Show all available recipes')]
default:
    @just --list --unsorted

# ─── Install ────────────────────────────────────────────────────────────────

[doc('Check if uv is installed, install if missing')]
check-uv:
    @command -v uv >/dev/null 2>&1 && echo "uv is already installed" \
        || { echo "Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }

[doc('Install package (no dev dependencies)')]
install: check-uv
    uv sync --no-dev

[doc('Install with development dependencies')]
install-dev: check-uv
    uv sync

# ─── Code Quality ───────────────────────────────────────────────────────────

[doc('Format and lint code with ruff')]
lint:
    uv run ruff format .
    uv run ruff check --select I --fix .
    uv run ruff check --fix --exclude {{ pkg_name }}/wip {{ pkg_name }}

[doc('Format code with ruff (imports + style)')]
fmt:
    uv run ruff format .
    uv run ruff check --select I --fix .

[doc('Ruff check without auto-fix (CI-safe)')]
quality:
    @echo "Checking {{ pkg_name }} (excluding .venv and wip)..."
    uv run ruff check --exclude .venv --exclude {{ pkg_name }}/wip .

[doc('Run fmt + lint + test sequentially')]
check: fmt lint test

# ─── Testing ────────────────────────────────────────────────────────────────

[doc('Run unit tests only')]
test-unit *args:
    uv run pytest tests/unit_tests/ {{ args }}

[doc('Run unit + integration tests (no LLM/API keys required)')]
test *args:
    uv run pytest tests/unit_tests/ tests/integration_tests/ {{ args }}

[doc('Run integration tests with real LLM API calls')]
test-full *args:
    uv run pytest tests/integration_tests/ --include-real-models {{ args }}

[doc('Run all tests including real LLM calls')]
test-all *args:
    uv run pytest tests/unit_tests/ tests/integration_tests/ --include-real-models -m 'not slow' {{ args }}

# Delegated to cli test — these need config-path resolution, marker logic, or notebook execution:

[doc('Run eval tests  (--real for LLM-judged, --deerflow for DeerFlow suite)')]
test-evals *args:
    uv run cli test evals {{ args }}

[doc('Run tests matching a pattern across all test dirs  e.g: just test-select rag')]
test-select pattern *args:
    uv run cli test select '{{ pattern }}' {{ args }}

[doc('Execute Jupyter notebooks as tests')]
test-notebooks *args:
    uv run cli test notebooks {{ args }}

[doc('Run pytest with custom args  e.g: just pytest -k my_test -v')]
pytest *args:
    uv run pytest {{ args }}

[doc('Quick smoke-test: verify basic package imports')]
test-install:
    echo "Testing {{ pkg_name }} imports..."
    uv run python -c "import genai_tk.core;  print('ok genai_tk.core')"
    uv run python -c "import genai_tk.extra; print('ok genai_tk.extra')"
    uv run python -c "import genai_tk.utils; print('ok genai_tk.utils')"
    echo -e "\033[3m\033[36mExpected output: 'Human: Tell me a joke on bears'\033[0m"
    echo bears | uv run cli core run joke -m parrot_local@fake

# ─── Web Interface ──────────────────────────────────────────────────────────

[doc('Launch built-in agent demo webapp (Streamlit)')]
webapp:
    uv run python -m streamlit run "{{ streamlit_entry }}"

# ─── Deer-flow ──────────────────────────────────────────────────────────────

[doc('Clone/update Deer-flow and install backend + Python deps')]
deer-flow-install:
    [ -d "{{ deer_flow_dir }}" ] \
        && (echo "Updating Deer-flow..." && cd {{ deer_flow_dir }} && git pull --rebase) \
        || (echo "Cloning Deer-flow..." && mkdir -p ext && git clone --depth 1 {{ deer_flow_repo }} {{ deer_flow_dir }})
    echo "Installing Deer-flow backend..."
    uv run python scripts/install_deer_flow_backend.py
    uv sync --group deer-flow
    @echo ""
    @echo "✓ Deer-flow installed."
    @echo "  Add to your .env:  DEER_FLOW_PATH=$(pwd)/{{ deer_flow_dir }}"

# ─── Maintenance ────────────────────────────────────────────────────────────

[doc('Clean Python bytecode and cache files')]
clean:
    uv cache prune
    find . \( -name "*.py[co]" -o -name "__pycache__" \
             -o -name ".ruff_cache" -o -name ".mypy_cache" \
             -o -name ".pytest_cache" \) \
        -exec rm -rf {} + 2>/dev/null || true

[doc('Clear Jupyter notebook outputs')]
clean-notebooks:
    find . -path "./.venv" -prune -o -name "*.ipynb" -print \
        | while read -r nb; do \
            echo "Cleaning: $nb"; \
            uv run --with nbconvert python -m nbconvert --clear-output --inplace "$nb"; \
          done

[confirm("This will modify ~/.bash_history. Continue?")]
[doc('Remove duplicates and noise from ~/.bash_history')]
clean-history:
    [ -f ~/.bash_history ] \
        && awk '!/^(ls|cat|hgrep|h|cd|p|m|ll|pwd|code|mkdir|export|rmdir|uv tree|make|just)( |$)/ \
              && !seen[$0]++' ~/.bash_history > ~/.bash_history_unique \
        && mv ~/.bash_history_unique ~/.bash_history \
        && echo "Done. Run 'history -c; history -r' to reload." \
        || echo "No ~/.bash_history found"

[script]
hello:
    print("Hello from Python!")

[script]
goodbye:
    # /// script
    # requires-python = ">=3.11"
    # dependencies=["sh"]
    # ///
    import sh
    print(sh.echo("Goodbye from Python!"), end='')
