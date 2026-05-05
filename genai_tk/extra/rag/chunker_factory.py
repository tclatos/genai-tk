"""Factory for creating TextSplitter instances from YAML configuration.

Follows the same pattern as RetrieverFactory: loads chunker definitions from
config/rag.yaml, validates them with Pydantic, and instantiates the configured
TextSplitter classes.

Supports both explicit chunker selection and automatic selection based on file extension.

Example:
    ```python
    from genai_tk.extra.rag.chunker_factory import ChunkerFactory

    # Create a named chunker from config
    splitter = ChunkerFactory.create("markdown")
    docs = splitter.split_documents([Document(page_content=text)])

    # Auto-select chunker based on file extension
    from upath import UPath

    splitter = ChunkerFactory.create_for_file(UPath("document.md"))
    docs = splitter.split_documents([Document(page_content=text)])
    ```
"""

from __future__ import annotations

from langchain_text_splitters import TextSplitter
from loguru import logger
from upath import UPath

from genai_tk.utils.config_mngr import global_config, import_from_qualified


class ChunkerFactory:
    """Factory for creating TextSplitter instances from YAML configuration."""

    @classmethod
    def create(cls, config_tag: str) -> TextSplitter:
        """Create a TextSplitter from a named configuration.

        Loads the chunker definition from config.chunkers[config_tag],
        imports the class, and instantiates it with the configured parameters.

        Args:
            config_tag: Name of the chunker configuration (e.g., "markdown", "recursive").

        Returns:
            Configured TextSplitter instance.

        Raises:
            KeyError: If config_tag not found in config.chunkers.
            ImportError: If the configured class cannot be imported.
            Exception: If instantiation fails (e.g., invalid parameters).
        """
        config = global_config()
        chunker_config = config.get_dict(f"chunkers.{config_tag}")

        if not chunker_config:
            raise KeyError(
                f"Chunker '{config_tag}' not found in config.chunkers. "
                f"Available: {list(config.get_dict('chunkers', {}).keys())}"
            )

        class_path = chunker_config.get("class")
        params = chunker_config.get("params", {})

        if not class_path:
            raise ValueError(f"Chunker '{config_tag}' missing 'class' field")

        logger.debug("Creating chunker '{}' from class {}", config_tag, class_path)

        # Import the class
        splitter_cls = import_from_qualified(class_path)

        # Instantiate with parameters
        try:
            return splitter_cls(**params)
        except Exception as exc:
            raise RuntimeError(f"Failed to instantiate chunker '{config_tag}' with class {class_path}: {exc}") from exc

    @classmethod
    def create_for_file(
        cls,
        path: UPath | str,
        chunker_name: str = "auto",
    ) -> TextSplitter:
        """Create a TextSplitter for a file, optionally with auto-detection.

        If chunker_name is "auto", selects the chunker based on file extension
        using config.chunker_auto_map.

        Args:
            path: File path (str or UPath).
            chunker_name: Chunker configuration name.
                If "auto", uses file extension to select from chunker_auto_map.

        Returns:
            Configured TextSplitter instance.

        Raises:
            KeyError: If chunker_name not found in config, or if "auto" is selected
                but no mapping exists for the file extension.
        """
        if isinstance(path, str):
            path = UPath(path)

        if chunker_name == "auto":
            config = global_config()
            auto_map = config.get_dict("chunker_auto_map", {})

            # Try file extension first
            ext = path.suffix.lower()
            if ext in auto_map:
                selected_chunker = auto_map[ext]
                logger.debug("Auto-selected chunker '{}' for extension '{}'", selected_chunker, ext)
            elif ".default" in auto_map:
                selected_chunker = auto_map[".default"]
                logger.debug("Using default chunker for extension '{}' (not in auto_map)", ext)
            else:
                raise KeyError(
                    f"No chunker mapping for extension '{ext}' and no '.default' fallback. "
                    f"Configure in config.chunker_auto_map"
                )

            return cls.create(selected_chunker)

        return cls.create(chunker_name)
