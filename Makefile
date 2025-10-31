# Makefile for GenAI Toolkit (genai-tk)
# Provides commands for development, testing, and maintenance

.PHONY: help install install-dev fmt lint test test-unit test-integration clean check

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install the package and dependencies"
	@echo "  install-dev  - Install with development dependencies"
	@echo "  fmt          - Format code with ruff"
	@echo "  lint         - Lint code with ruff"
	@echo "  test         - Run all tests"
	@echo "  test-unit    - Run unit tests only"
	@echo "  test-integration - Run integration tests only"  
	@echo "  test-install - Quick test of package installation"
	@echo "  clean        - Clean Python cache files"
	@echo "  check        - Run format, lint, and test"

install: ## Install the package
	uv sync --no-dev

install-dev: ## Install with development dependencies  
	uv sync

fmt: ## Format code with ruff
	uv run ruff format .
	uv run ruff check --select I --fix .

lint: ## Lint code with ruff
	uv run ruff check --fix genai_tk

test: ## Run all tests
	uv run pytest tests/unit_tests/ tests/integration_tests/

test-unit: ## Run unit tests only
	uv run pytest tests/unit_tests/

test-integration: ## Run integration tests only
	uv run pytest tests/integration_tests/

test-install: ## Quick test of package installation - tests basic imports
	@echo "Testing genai_tk package imports..."
	@uv run python -c "import genai_tk.core; print('✓ genai_tk.core imported successfully')" || echo "✗ Failed to import genai_tk.core"
	@uv run python -c "import genai_tk.extra; print('✓ genai_tk.extra imported successfully')" || echo "✗ Failed to import genai_tk.extra" 
	@uv run python -c "import genai_tk.utils; print('✓ genai_tk.utils imported successfully')" || echo "✗ Failed to import genai_tk.utils"
	@echo "✓ Basic package structure test completed"

clean: ## Clean Python cache files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ruff_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	find . -type d -name ".pytest_cache" -delete

check: fmt lint test ## Run format, lint, and test