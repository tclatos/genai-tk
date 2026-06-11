"""Optional feature registry for genai-tk.

Maps feature names to their PyPI packages, importable module names, and install commands.
Use this module to gate imports at call sites and to generate helpful error messages.

Example:
    ```python
    from genai_tk.config_mgmt.features import require_feature, is_available

    # At the top of a command or page that needs browser automation:
    require_feature("browser")

    # In a conditional block:
    if is_available("nlp"):
        import spacy

        ...
    ```

Pytest integration — mark tests with ``@pytest.mark.requires_feature("<name>")``.
The conftest plugin in ``tests/conftest.py`` skips those tests automatically when
the feature packages are absent.
"""

from __future__ import annotations

from importlib.util import find_spec

from pydantic import BaseModel


class FeatureInfo(BaseModel):
    """Metadata for one optional feature."""

    description: str
    """Human-readable description shown in error messages."""

    packages: list[str]
    """PyPI package names required by this feature."""

    check_modules: list[str]
    """Importable module names used to probe whether the feature is installed."""

    install_cmd: str
    """The recommended install command shown when the feature is missing."""


# ---------------------------------------------------------------------------
# Registry — one entry per [project.optional-dependencies] extra in pyproject.toml.
# Keep in sync manually (or use `cli features check` once that command exists).
# ---------------------------------------------------------------------------
FEATURES: dict[str, FeatureInfo] = {
    "harnessing": FeatureInfo(
        description="Agent sandbox, DeerFlow harness, DeepAgents CLI, SmolAgents",
        packages=["smolagents", "deepagents", "deepagents-cli", "agent-sandbox", "opensandbox", "deerflow-harness"],
        check_modules=["smolagents", "deepagents", "agent_sandbox", "opensandbox", "deerflow"],
        install_cmd='uv sync --extra harnessing  # or: uv add "genai-tk[harnessing]"',
    ),
    "browser": FeatureInfo(
        description="Browser automation (Playwright)",
        packages=["playwright"],
        check_modules=["playwright"],
        install_cmd='uv sync --extra browser  # or: uv add "genai-tk[browser]"',
    ),
    "nlp": FeatureInfo(
        description="NLP with spaCy, English language models, and Presidio PII detection",
        packages=["spacy", "en-core-web-sm", "en-core-web-lg", "presidio-analyzer", "presidio-anonymizer"],
        check_modules=["spacy"],
        install_cmd='uv sync --extra nlp  # or: uv add "genai-tk[nlp]"',
    ),
    "postgres": FeatureInfo(
        description="PostgreSQL vector store (pgvector / langchain-postgres)",
        packages=["langchain-postgres", "psycopg", "psycopg2-binary"],
        check_modules=["langchain_postgres", "psycopg"],
        install_cmd='uv sync --extra postgres  # or: uv add "genai-tk[postgres]"',
    ),
    "streamlit": FeatureInfo(
        description="Streamlit web interface",
        packages=["streamlit"],
        check_modules=["streamlit"],
        install_cmd='uv sync --extra streamlit  # or: uv add "genai-tk[streamlit]"',
    ),
    "baml": FeatureInfo(
        description="BAML structured data extraction",
        packages=["baml-py"],
        check_modules=["baml_lib"],
        install_cmd='uv sync --extra baml  # or: uv add "genai-tk[baml]"',
    ),
    "chromadb": FeatureInfo(
        description="ChromaDB vector database",
        packages=["chromadb", "langchain-chroma"],
        check_modules=["chromadb"],
        install_cmd='uv sync --extra chromadb  # or: uv add "genai-tk[chromadb]"',
    ),
}


def is_available(feature: str) -> bool:
    """Return True if all probe modules for *feature* are importable.

    Args:
        feature: Feature key (e.g. ``"browser"``, ``"nlp"``).

    Returns:
        True when every ``check_modules`` entry can be located by the import system.

    Raises:
        ValueError: If *feature* is not registered in :data:`FEATURES`.
    """
    info = _get_info(feature)
    return all(find_spec(m) is not None for m in info.check_modules)


def require_feature(feature: str, context: str = "") -> None:
    """Raise :class:`ImportError` with install instructions if *feature* is missing.

    Call this at the top of any function or module that depends on an optional feature.
    The raised error message includes the exact ``uv`` command needed to install it.

    Args:
        feature: Feature key (e.g. ``"browser"``, ``"nlp"``).
        context: Optional description of the caller shown in the error (e.g. ``"cli agents sandbox"``).

    Raises:
        ImportError: When one or more probe modules for *feature* are not importable.
        ValueError: If *feature* is not registered in :data:`FEATURES`.

    Example:
        ```python
        require_feature("browser", context="cli agents sandbox")
        from playwright.async_api import async_playwright
        ```
    """
    info = _get_info(feature)
    missing_modules = [m for m in info.check_modules if find_spec(m) is None]
    if missing_modules:
        ctx = f" (required by: {context})" if context else ""
        missing_pkgs = ", ".join(info.packages)
        raise ImportError(
            f"Optional feature '{feature}'{ctx} is not installed.\n"
            f"  Description : {info.description}\n"
            f"  Packages    : {missing_pkgs}\n"
            f"  Install with: {info.install_cmd}"
        )


def available_features() -> list[str]:
    """Return names of all features currently installed.

    Returns:
        Sorted list of feature keys whose packages are all present.
    """
    return sorted(k for k in FEATURES if is_available(k))


def missing_features() -> list[str]:
    """Return names of all features that are NOT currently installed.

    Returns:
        Sorted list of feature keys with at least one missing package.
    """
    return sorted(k for k in FEATURES if not is_available(k))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_info(feature: str) -> FeatureInfo:
    info = FEATURES.get(feature)
    if info is None:
        valid = ", ".join(sorted(FEATURES))
        raise ValueError(f"Unknown feature '{feature}'. Valid features: {valid}")
    return info
