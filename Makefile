# cSpell: disable
# GenAI Toolkit -- development and testing Makefile.
# Standalone: no external genai-* project dependencies.
# Downstream projects vendor the shared targets via tk_makefile.mk.

##############################
##  Settings
##############################
PKG_NAME = genai_tk

MAKEFLAGS += --warn-undefined-variables
SHELL     := bash -euo pipefail -c

##############################
##  .env discovery (walk up to find .env)
##############################
ENV_FILE := $(shell \
	if   [ -f ".env" ];       then echo "$(CURDIR)/.env"; \
	elif [ -f "../.env" ];    then echo "$(CURDIR)/../.env"; \
	elif [ -f "../../.env" ]; then echo "$(CURDIR)/../../.env"; \
	else echo ""; fi)
ifneq ($(ENV_FILE),)
include $(ENV_FILE)
else
$(warning .env file not found in current or parent directories)
endif

all: help

##############################
##  Guards
##############################
.PHONY: .uv .pre-commit

.uv:  ## Check that uv is installed
	@uv -V || echo 'Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh'

.pre-commit: .uv  ## Check that pre-commit is installed
	@uv run pre-commit -V || uv pip install pre-commit

##############################
##  Install
##############################
.PHONY: check-uv install install-dev

check-uv:  ## Check if uv is installed, install if missing
	@if command -v uv >/dev/null 2>&1; then \
		echo "uv is already installed"; \
	else \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		. $$HOME/.local/bin/env; \
	fi

install: check-uv  ## Install package (no dev dependencies)
	uv sync --no-dev

install-dev: check-uv  ## Install with development dependencies
	uv sync


##############################
##  Testing Install
##############################
.PHONY: test-install

test-install:  ## Quick smoke-test: call a fake LLM via the CLI
	@if [ -z "$(PYTHONPATH)" ]; then \
		echo -e "\033[33mWarning: PYTHONPATH is not set.\033[0m"; \
	else \
		echo -e "\033[32mPYTHONPATH=$(PYTHONPATH)\033[0m"; \
	fi
 ## Quick smoke-test: verify basic package imports
	@echo "Testing $(PKG_NAME) imports..."
	@uv run python -c "import genai_tk.core;  print('ok genai_tk.core')"  || echo "FAIL genai_tk.core"
	@uv run python -c "import genai_tk.extra; print('ok genai_tk.extra')" || echo "FAIL genai_tk.extra"
	@uv run python -c "import genai_tk.utils; print('ok genai_tk.utils')" || echo "FAIL genai_tk.utils"
# test Echo LLM
	@echo -e "\033[3m\033[36mExpected output: '"Human: Tell me a joke on {'topic': 'bears\\n'}"'\032[0m"
	echo bears | PYTHONPATH=$(DEV_PYTHONPATH) uv run cli core run joke -m parrot_local@fake | \
		while IFS= read -r line; do echo -e "\033[32m$$line\033[0m"; done

##############################
##  Code Quality
##############################
.PHONY: fmt lint quality check


lint:  ## format (imports + style) and lint code with ruff (fix safe issues)
	uv run ruff format .
	uv run ruff check --select I --fix .
	uv run ruff check --fix --exclude $(PKG_NAME)/wip $(PKG_NAME) 



##############################
##  Testing
##############################
# Fine-grained suites are available via:  uv run cli test <sub-command> [opts]
# e.g.  uv run cli test evals --real
#       uv run cli test evals --deerflow --timeout 360
#       uv run cli test full
.PHONY: test test-unit test-integration test-evals test-full test-install pytest

test:  ## Run unit + integration tests
	uv run pytest tests/unit_tests/ tests/integration_tests/

test-unit:  ## Run unit tests only
	uv run pytest tests/unit_tests/

test-integration:  ## Run integration tests only
	uv run pytest tests/integration_tests/

pytest:  ## Run pytest with custom args: make pytest ARGS="-k my_test -v"
	uv run pytest $(ARGS)



##############################
##  Maintenance
##############################
.PHONY: clean clean-notebooks clean-history

clean:  ## Clean Python bytecode and cache files
	uv cache prune
	find . \( -name "*.py[co]" -o -name "__pycache__" \
	         -o -name ".ruff_cache" -o -name ".mypy_cache" \
	         -o -name ".pytest_cache" \) \
		-exec rm -rf {} + 2>/dev/null || true

clean-notebooks:  ## Clear Jupyter notebook outputs
	@find . -path "./.venv" -prune -o -name "*.ipynb" -print | while read -r nb; do \
		echo "Cleaning: $$nb"; \
		uv run --with nbconvert python -m nbconvert --clear-output --inplace "$$nb"; \
	done

clean-history:  ## Remove duplicates and noise from ~/.bash_history
	@if [ -f ~/.bash_history ]; then \
		awk '!/^(ls|cat|hgrep|h|cd|p|m|ll|pwd|code|mkdir|export|rmdir|uv tree|make)( |$$)/ \
		      && !seen[$$0]++' ~/.bash_history > ~/.bash_history_unique && \
		mv ~/.bash_history_unique ~/.bash_history; \
		echo "Done. Run 'history -c; history -r' to reload."; \
	else \
		echo "No ~/.bash_history found"; \
	fi

##############################
##  Help
##############################
.PHONY: help

help:
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*?##/ \
		{ printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 }' \
		$(MAKEFILE_LIST) | sort -u


##############################
##  Deer-flow Integration
##############################
DEER_FLOW_REPO = https://github.com/bytedance/deer-flow.git
DEER_FLOW_DIR  = ext/deer-flow

.PHONY: deer-flow-install

##############################
##  Web Interface
##############################
STREAMLIT_ENTRY ?= genai_tk/webapp/main/streamlit.py

.PHONY: webapp

webapp:  ## Launch built-in agent demo webapp (Streamlit)
	uv run streamlit run "$(STREAMLIT_ENTRY)"

deer-flow-install:  ## Clone/update Deer-flow and install backend + Python deps
	@if [ -d "$(DEER_FLOW_DIR)" ]; then \
		echo "Updating Deer-flow..."; \
		cd $(DEER_FLOW_DIR) && git pull --rebase; \
	else \
		echo "Cloning Deer-flow..."; \
		mkdir -p ext; \
		git clone --depth 1 $(DEER_FLOW_REPO) $(DEER_FLOW_DIR); \
	fi
	@echo "Installing Deer-flow backend..."
	@uv run python scripts/install_deer_flow_backend.py
	@uv sync --group deer-flow
	@echo ""
	@echo "✓ Deer-flow installed."
	@echo ""
	@echo "  Add to your .env (or export before running):"
	@echo "    DEER_FLOW_PATH=$(CURDIR)/$(DEER_FLOW_DIR)"
	@echo ""
	@echo "  The value is also set automatically when you use 'make' targets."
