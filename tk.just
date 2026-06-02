# tk.just — shared genai-tk recipes for downstream projects.
# Import in your project justfile:  import '../genai-tk/tk.just'
# Or copy this file as tk.just in your project root.
#
# Requires `pkg_name` variable set before import.

# ─── Install ────────────────────────────────────────────────────────────────

[doc('Check if uv is installed, install if missing')]
check-uv:
    @command -v uv >/dev/null 2>&1 && echo "uv is already installed" \
        || { echo "Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }

[doc('Install package and dependencies')]
install: check-uv
    uv sync

[doc('Install with development dependencies (all extras)')]
install-dev: check-uv
    uv sync --all-extras

# ─── Code Quality ───────────────────────────────────────────────────────────

[doc('Format code with ruff (imports + style)')]
fmt:
    uv run ruff format .
    uv run ruff check --select I --fix .

[doc('Lint code with ruff (fix safe issues)')]
lint:
    uv run ruff check --fix {{ pkg_name }}

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
# ─── Web Interface ──────────────────────────────────────────────────────────────

[doc('Launch Streamlit webapp (entry point resolved from genai-tk package)')]
webapp:
    entry=$(uv run python -c 'import pathlib, genai_tk; print(pathlib.Path(genai_tk.__file__).parent / "webapp/main/streamlit.py")') && uv run python -m streamlit run "$entry"
# ─── Maintenance ────────────────────────────────────────────────────────────

[doc('Sync with remote, stash changes, rebase, upgrade genai-tk')]
rebase:
    git fetch origin
    git stash
    git rebase origin/main
    uv sync --upgrade-package genai-tk

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

[doc('Remove duplicates and noise from ~/.bash_history')]
[confirm("This will modify ~/.bash_history. Continue?")]
clean-history:
    [ -f ~/.bash_history ] \
        && awk '!/^(ls|cat|hgrep|h|cd|p|m|ll|pwd|code|mkdir|export|rmdir|uv tree|make|just)( |$)/ \
              && !seen[$0]++' ~/.bash_history > ~/.bash_history_unique \
        && mv ~/.bash_history_unique ~/.bash_history \
        && echo "Done. Run 'history -c; history -r' to reload." \
        || echo "No ~/.bash_history found"

# ─── Info ───────────────────────────────────────────────────────────────────

[doc('Show DEV_PYTHONPATH and genai-tk availability')]
show-dev-path:
    @echo "DEV_PYTHONPATH : ${DEV_PYTHONPATH:-../genai-tk:.:$(pwd)}"
    @[ -d ../genai-tk ] \
        && echo "genai-tk source: EXISTS (local checkout)" \
        || echo "genai-tk source: NOT FOUND (using .venv)"
