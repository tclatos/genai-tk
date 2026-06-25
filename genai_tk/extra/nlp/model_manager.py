"""SpaCy model management — download, install, and locate models.

Provides utilities to manage spaCy models with automatic download and
path management via the global configuration.

Example:
    ```python
    from genai_tk.extra.nlp.model_manager import SpaCyModelManager

    # Set up model (downloads if needed)
    SpaCyModelManager.setup_spacy_model("en_core_web_sm")

    # Check availability
    if SpaCyModelManager.is_model_installed("fr_core_news_sm"):
        ...
    ```
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from loguru import logger

from genai_tk.config_mgmt.config_mngr import global_config


class SpaCyModelManager:
    """Manages spaCy model installation and configuration."""

    @staticmethod
    def get_model_path(model_name: str) -> Path:
        """Get the path where the spaCy model should be stored."""
        path = global_config().get_dir_path("paths.models", create_if_not_exists=True)
        return path / "spacy_models" / model_name

    @staticmethod
    def is_model_installed(model_name: str) -> bool:
        """Check if the spaCy model is installed (globally or in custom directory)."""
        import importlib.util

        # Check global availability first (fast)
        if importlib.util.find_spec(model_name) is not None:
            return True
        # Fallback: check custom directory
        model_path = SpaCyModelManager.get_model_path(model_name)
        return model_path.exists()

    @staticmethod
    def download_model(model_name: str) -> Path:
        """Download the spaCy model if not already present."""
        model_path = SpaCyModelManager.get_model_path(model_name)

        if model_path.exists():
            return model_path

        logger.info("Downloading spaCy model '{}' to {}", model_name, model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["python", "-m", "spacy", "download", model_name],
            check=True,
        )

        # Create a symlink in our models directory for consistency
        import spacy

        global_model_path = None
        try:
            global_model_path = spacy.util.get_package_path(model_name)
            if global_model_path.exists() and not model_path.exists():
                model_path.symlink_to(global_model_path)
        except Exception as e:
            logger.warning("Could not create symlink for model {}: {}", model_name, e)
            if global_model_path and global_model_path.exists():
                import shutil

                shutil.copytree(global_model_path, model_path, dirs_exist_ok=True)

        return model_path

    @staticmethod
    def setup_spacy_model(model_name: str) -> None:
        """Set up the spaCy model by downloading it if needed.

        Args:
            model_name: spaCy model name (e.g. ``"en_core_web_sm"``, ``"fr_core_news_sm"``).

        Raises:
            ImportError: If spaCy is not installed.
            RuntimeError: If the model cannot be set up after download.
        """
        try:
            import spacy
        except ImportError as e:
            raise ImportError("spaCy is not installed. Install the NLP extra with: uv sync --extra nlp") from e

        # Try global load first
        try:
            spacy.load(model_name)
            logger.debug("spaCy model '{}' is available globally", model_name)
            return
        except OSError:
            pass

        # Try custom path
        model_path = SpaCyModelManager.get_model_path(model_name)
        if model_path.exists():
            try:
                spacy.load(model_path)
                logger.debug("spaCy model '{}' loaded from {}", model_name, model_path)
                return
            except Exception:
                # Broken install — clean up and re-download
                import shutil

                if model_path.is_symlink():
                    model_path.unlink()
                elif model_path.is_dir():
                    shutil.rmtree(model_path)

        # Download and verify
        SpaCyModelManager.download_model(model_name)
        try:
            spacy.load(model_name)
            logger.info("spaCy model '{}' downloaded and loaded successfully", model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load spaCy model '{model_name}' after download. "
                f"Try manually: python -m spacy download {model_name}"
            ) from e

    @staticmethod
    def require_model(model_name: str, language: str | None = None) -> None:
        """Verify that *model_name* is loadable; raise a helpful error if not.

        Unlike :meth:`setup_spacy_model`, this does NOT attempt to download — it only
        checks availability and raises with install instructions.

        Args:
            model_name: spaCy model to check.
            language: Optional language hint for the error message.

        Raises:
            ImportError: If the model is not installed.
        """
        import importlib.util

        if importlib.util.find_spec("spacy") is None:
            raise ImportError("spaCy is not installed. Install the NLP extra with: uv sync --extra nlp")

        import spacy

        try:
            spacy.load(model_name)
        except OSError:
            # Check custom path
            model_path = SpaCyModelManager.get_model_path(model_name)
            if model_path.exists():
                try:
                    spacy.load(model_path)
                    return
                except Exception:
                    pass

            lang_hint = f" for language '{language}'" if language else ""
            raise ImportError(
                f"spaCy model '{model_name}' is not installed{lang_hint}. "
                f"Install it with: python -m spacy download {model_name}"
            ) from None
