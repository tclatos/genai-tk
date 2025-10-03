# Development Guidelines for GenAI Toolkit

## Build/Lint/Test Commands

**Package Management:** Uses `uv` for dependency management
- `make install-dev` - Install with development dependencies
- `make fmt` - Format code with ruff (includes import sorting)
- `make lint` - Lint code with ruff
- `make test` - Run all tests (unit + integration)
- `make test-unit` - Run unit tests only
- `make test-integration` - Run integration tests only
- `make check` - Run format, lint, and test sequentially

**Single Test Execution:**
- `uv run pytest tests/unit_tests/core/test_llm_factory.py::test_basic_call -v`
- `uv run pytest tests/unit_tests/ -k "test_name_pattern" -v`

## Code Style Guidelines

**Formatting & Linting:**
- Line length: 120 characters
- Use ruff for formatting and linting
- Import sorting: isort rules (handled by ruff)
- Docstring convention: Google style

**Type System:**
- Python 3.12+ required
- Use type hints extensively (pydantic models preferred)
- Runtime type checking with beartype
- Return types encouraged but not strictly required (ANN201/ANN202 ignored)

**Naming Conventions:**
- Classes: PascalCase (e.g., `LlmFactory`, `OmegaConfig`)
- Functions/variables: snake_case (e.g., `get_llm`, `llm_config`)
- Constants: UPPER_SNAKE_CASE (e.g., `APPLICATION_CONFIG_FILE`)
- Private members: underscore prefix (e.g., `_internal_method`)

**Import Organization:**
- Standard library imports first
- Third-party imports second
- Local imports third
- Prefer explicit imports over star imports
- profer absolute imports

**Error Handling:**
- Use structured logging with loguru
- Raise specific exceptions with descriptive messages
- Use pydantic for data validation and error handling
- Implement proper error boundaries in agent workflows

**Testing:**
- Use pytest with asyncio support
- Test files: `test_*.py` or `*_test.py`
- Place unit tests in `tests/unit_tests/`
- Place integration tests in `tests/integration_tests/`
- Use faker for test data generation

**Configuration:**
- YAML-based configuration with OmegaConf
- Environment variable substitution supported
- Singleton pattern for global config access
- Separate configs for different environments (dev, prod, etc.)