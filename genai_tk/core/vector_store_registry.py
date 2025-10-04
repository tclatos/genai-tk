"""Vector Store Management and Factory System.

This module provides a comprehensive interface for creating, managing, and
interacting with vector stores across multiple storage backends. It supports
advanced document indexing, retrieval, and vector database operations.

Key Features:
- Multi-backend vector store support (Chroma, In-Memory, Sklearn, PgVector)
- Flexible document indexing and deduplication
- Configurable retrieval strategies
- Seamless integration with embedding models
- Advanced search and filtering capabilities
- Generic configuration system with YAML override support
- Hybrid search support for PostgreSQL (vector + full-text search)

Supported Backends:
- Chroma (persistent and in-memory)
- InMemoryVectorStore
- SKLearnVectorStore
- PgVector (with hybrid search support)

Design Patterns:
- Factory Method for vector store creation
- Configurable retrieval strategies
- Generic configuration via dict parameter
- Singleton-like access to vector stores

Example:
    ```python
    # Configuration-based factory creation (recommended)
    factory = VectorStoreRegistry.create_from_config("default")

    # PgVector example from configuration
    pg_factory = VectorStoreRegistry.create_from_config("postgres")

    # Add documents to the store
    factory.add_documents([
        Document(page_content="First document"),
        Document(page_content="Second document")
    ])

    # Perform similarity search
    results = factory.get().similarity_search("query")

    # Configuration in baseline.yaml:
    # vector_store_registry:
    #   default:
    #     backend: Chroma
    #     embeddings: default
    #     config:
    #       storage: '::memory::'  # In-memory storage
    #
    #   persistent:
    #     backend: Chroma
    #     embeddings: default
    #     config:
    #       storage: /path/to/storage  # Persistent storage
    #
    #   postgres:
    #     backend: PgVector
    #     embeddings: default
    #     config:
    #       postgres_url: postgresql://user:pass@localhost:5432/db
    ```
"""

import os
from collections.abc import Iterable
from typing import Annotated, Any, Literal, get_args

from devtools import debug
from langchain.embeddings.base import Embeddings
from langchain.indexes import IndexingResult, SQLRecordManager, index
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore

try:
    from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig
except ImportError:
    HybridSearchConfig = None  # Optional dependency
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from genai_tk.core.embeddings_factory import EmbeddingsFactory
from genai_tk.utils.config_mngr import global_config, global_config_reload

# List of known Vector Stores (created as Literal so can be checked by MyPy)
VECTOR_STORE_ENGINE = Literal["Chroma", "InMemory", "Sklearn", "PgVector"]


class VectorStoreRegistry(BaseModel):
    """Factory for creating and managing vector stores with advanced configuration.

    Provides a flexible and powerful interface for creating vector stores with
    support for multiple backends, document indexing, and advanced retrieval
    strategies.

    Important: This class should only be instantiated through the create_from_config
    class method. Direct instantiation is not allowed.

    Attributes:
        backend: Identifier for the vector store backend
        embeddings_factory: Factory for creating embedding models
        table_name_prefix: Prefix for generated table/collection names
        config: Dictionary of vector store specific configuration that overrides YAML values
        index_document: Flag to enable document deduplication and indexing
        collection_metadata: Optional metadata for the collection

    Example:
        >>> # Configuration-based instantiation (recommended)
        >>> factory = VectorStoreRegistry.create_from_config("default")
        >>> factory = VectorStoreRegistry.create_from_config("local_indexed")
        >>> factory.add_documents([Document(page_content="example")])
        >>>
        >>> # Hybrid search example (PostgreSQL)
        >>> hybrid_factory = VectorStoreRegistry.create_from_config("postgres_hybrid")
    """

    backend: Annotated[VECTOR_STORE_ENGINE | None, Field(validate_default=True, alias="id")] = None
    embeddings_factory: EmbeddingsFactory
    table_name_prefix: str = "embeddings"
    config: dict[str, Any] = {}
    index_document: bool = False
    collection_metadata: dict[str, str] | None = None
    _record_manager: SQLRecordManager | None = None
    _conf: dict = {}

    def __init__(self, **data):
        """Direct instantiation is not allowed. Use create_from_config instead."""
        raise RuntimeError(
            "VectorStoreRegistry cannot be instantiated directly. "
            "Use VectorStoreRegistry.create_from_config(config_tag) instead."
        )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context) -> None:
        """Post-initialization logic for VectorStoreRegistry.

        Handles backward compatibility and storage path normalization.
        """
        # Handle Chroma storage: if backend is Chroma and no storage is configured,
        # default to in-memory storage
        if self.backend == "Chroma":
            storage = self.config.get("storage") or self.config.get("chroma_path")
            if not storage:
                self.config["storage"] = "::memory::"
            elif "chroma_path" in self.config and "storage" not in self.config:
                # Migrate from old chroma_path to new storage
                self.config["storage"] = self.config.pop("chroma_path")

    @computed_field
    @property
    def table_name(self) -> str:
        """Generate a name by combining collection and embeddings ID.

        Returns:
            Unique collection name to prevent conflicts
        """
        assert self.embeddings_factory
        embeddings_id = self.embeddings_factory.short_name()
        return f"{self.table_name_prefix}_{embeddings_id}"

    @computed_field
    def description(self) -> str:
        """Generate a detailed description of the vector store configuration.

        Returns:
            Comprehensive configuration description string
        """
        r = f"{str(self.backend)}/{self.table_name}"
        if self.backend == "Chroma":
            storage = self.config.get("storage", "::memory::")
            if storage == "::memory::":
                r += " => 'in-memory'"
            else:
                r += " => 'on disk'"
        if self.index_document and self._record_manager:
            r += f" indexer: {self._record_manager}"
        return r

    @staticmethod
    def known_items() -> list[str]:
        """List all supported vector store backends.

        Returns:
            List of supported vector store engine identifiers
        """
        return list(get_args(VECTOR_STORE_ENGINE))

    @classmethod
    def create_from_config(cls, config_tag: str) -> "VectorStoreRegistry":
        """Create a VectorStoreRegistry from configuration.

        Args:
            config_tag: Configuration tag to look up (e.g., 'default', 'local_indexed')

        Returns:
            Configured VectorStoreRegistry instance

        Raises:
            ValueError: If configuration tag is not found or invalid
            KeyError: If required configuration keys are missing

        Example:
            ```python
            # Configuration in baseline.yaml:
            # vector_store_registry:
            #   default:
            #     id: InMemory
            #   local_indexed:
            #     id: Chroma
            #     embeddings: default
            #     table_name_prefix: embeddings
            #     record_manager: sqlite:///data/record_manager.sql
            #     config:
            #       chroma_path: /custom/path

            factory = VectorStoreRegistry.create_from_config("default")
            factory = VectorStoreRegistry.create_from_config("local_indexed")
            ```
        """
        config_key = f"vector_store_registry.{config_tag}"
        try:
            config_dict = global_config().get_dict(config_key)
        except (ValueError, KeyError) as e:
            raise ValueError(f"Configuration for vector store factory '{config_tag}' not found") from e

        # Extract required backend field (support both 'backend' and legacy 'id')
        backend = config_dict.get("backend") or config_dict.get("id")
        if not backend:
            raise KeyError(f"Missing required 'backend' (or legacy 'id') field in configuration '{config_tag}'")

        if "id" in config_dict and "backend" not in config_dict:
            logger.warning(f"Configuration '{config_tag}' uses deprecated 'id' field. Use 'backend' instead.")

        # Resolve embeddings factory
        embeddings_factory = cls._resolve_embeddings_from_config(config_dict, config_tag)

        # Extract other configuration parameters
        table_name_prefix = config_dict.get("table_name_prefix", "embeddings")
        index_document = config_dict.get("index_document", False)
        collection_metadata = config_dict.get("collection_metadata")
        record_manager = config_dict.get("record_manager")
        config_overrides = config_dict.get("config", {})

        # Handle legacy chroma_path in config_overrides
        if "chroma_path" in config_overrides and "storage" not in config_overrides:
            logger.warning(f"Configuration '{config_tag}' uses deprecated 'chroma_path'. Use 'storage' instead.")
            config_overrides["storage"] = config_overrides.pop("chroma_path")

        # Handle record manager for indexing
        if record_manager and not index_document:
            index_document = True
            logger.info(f"Enabling document indexing because record_manager is specified in config '{config_tag}'")

        factory_args = {
            "backend": backend,
            "embeddings_factory": embeddings_factory,
            "table_name_prefix": table_name_prefix,
            "config": config_overrides,
            "index_document": index_document,
        }

        if collection_metadata is not None:
            factory_args["collection_metadata"] = collection_metadata

        # Create instance using model_construct to bypass __init__ restriction
        instance = cls.model_construct(**factory_args)
        # Run post-init manually since model_construct doesn't call it
        instance.model_post_init(None)
        return instance

    @staticmethod
    def _resolve_embeddings_from_config(config_dict: dict[str, Any], config_tag: str) -> EmbeddingsFactory:
        """Resolve embeddings factory from configuration.

        Args:
            config_dict: Configuration dictionary for the vector store
            config_tag: Configuration tag for error messaging

        Returns:
            Configured EmbeddingsFactory instance

        Raises:
            ValueError: If both embeddings_id and embeddings are specified, or if neither is found
        """
        embeddings_id = config_dict.get("embeddings_id")
        embeddings_tag = config_dict.get("embeddings")  # This can be either a tag or an id

        if embeddings_id and embeddings_tag:
            raise ValueError(
                f"Configuration '{config_tag}' cannot specify both 'embeddings_id' and 'embeddings'. "
                f"Use 'embeddings' for tags (e.g., 'default') or 'embeddings_id' for specific IDs."
            )

        if embeddings_id:
            return EmbeddingsFactory(embeddings_id=embeddings_id)
        elif embeddings_tag:
            # First try as embeddings_tag, if that fails try as embeddings_id
            try:
                return EmbeddingsFactory(embeddings_tag=embeddings_tag)
            except ValueError:
                # If tag resolution fails, try treating it as a direct embeddings_id
                try:
                    return EmbeddingsFactory(embeddings_id=embeddings_tag)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid embeddings specification '{embeddings_tag}' in config '{config_tag}'. "
                        f"Not found as tag or embeddings_id."
                    ) from e
        else:
            # Use default embeddings if neither is specified
            return EmbeddingsFactory()

    @field_validator("backend", mode="before")
    def check_known(cls, backend: str | None) -> str:
        """Validate and normalize the vector store backend identifier.

        Args:
            backend: Vector store backend identifier

        Returns:
            Validated vector store backend identifier

        Raises:
            ValueError: If an unknown vector store backend is specified
        """
        if backend is None:
            backend = global_config().get_str("vector_store.default")
        # Handle legacy Chroma_in_memory by converting to Chroma
        if backend == "Chroma_in_memory":
            logger.warning("Chroma_in_memory is deprecated. Use backend='Chroma' with storage='::memory::' instead")
            backend = "Chroma"
        if backend not in VectorStoreRegistry.known_items():
            raise ValueError(f"Unknown Vector Store: {backend}")
        return backend

    def get(self) -> VectorStore:
        """Create and configure a vector store based on the specified backend.

        Returns:
            Configured vector store instance

        Raises:
            ValueError: If an unsupported vector store backend is specified
        """
        embeddings = self.embeddings_factory.get()
        vector_store = None
        if self.backend == "Chroma":
            vector_store = self._create_chroma_vector_store(embeddings)
        elif self.backend == "InMemory":
            from langchain_core.vectorstores import InMemoryVectorStore

            vector_store = InMemoryVectorStore(
                embedding=embeddings,
            )
        elif self.backend == "Sklearn":
            from langchain_community.vectorstores import SKLearnVectorStore

            vector_store = SKLearnVectorStore(
                embedding=embeddings,
            )
        elif self.backend == "PgVector":
            vector_store = self._create_pg_vector_store()
        else:
            raise ValueError(f"Unknown vector store: {self.backend}")

        logger.debug(f"get vector store  : {self.description}")
        if self.index_document:
            # NOT TESTED
            db_url = global_config().get_str("vector_store.record_manager")
            logger.debug(f"vector store record manager : {db_url}")
            namespace = f"{self.backend}/{self.table_name}"
            self._record_manager = SQLRecordManager(
                namespace,
                db_url=db_url,
            )
            self._record_manager.create_schema()
        assert vector_store
        return vector_store

    def add_documents(self, docs: Iterable[Document]) -> IndexingResult | list[str]:
        """Add documents to the vector store with optional deduplication.

        Args:
            docs: Iterable of documents to add to the store

        Returns:
            Indexing result or list of document IDs

        Notes:
            Supports two modes of document addition:
            1. Direct addition without indexing
            2. Indexed addition with deduplication
        """
        if not self.index_document:
            return self.get().add_documents(list(docs))
        else:
            vector_store = self.get()
            assert self._record_manager

            info = index(
                docs,
                self._record_manager,
                vector_store,
                cleanup="incremental",
                source_id_key="source",
            )
            return info

    def document_count(self):
        """Count the number of documents in the vector store.

        Returns:
            Number of documents in the store

        Raises:
            NotImplementedError: For unsupported vector store backends
        """
        if self.backend == "Chroma":
            return self.get()._collection.count()  # type: ignore
        else:
            raise NotImplementedError(f"Don't know how to get collection count for {self.get()}")

    def _create_chroma_vector_store(self, embeddings: Embeddings) -> VectorStore:
        """Create and configure a Chroma vector store."""
        from langchain_chroma import Chroma

        storage = self.config.get("storage", "::memory::")

        if storage == "::memory::":
            persist_directory = None
        else:
            # Handle both absolute paths and paths that need to be resolved via global config
            if os.path.isabs(storage):
                persist_directory = storage
            else:
                # Try to resolve as config path first, fallback to literal path
                try:
                    store_path = global_config().get_dir_path("vector_store.storage", create_if_not_exists=True)
                    persist_directory = str(store_path)
                except (ValueError, KeyError):
                    persist_directory = storage

        return Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory,
            collection_name=self.table_name,
            collection_metadata=self.collection_metadata,
        )

    def _create_pg_vector_store(self) -> VectorStore:
        """Create and configure a PgVector store."""
        from genai_tk.extra.pgvector_factory import create_pg_vector_store

        return create_pg_vector_store(
            embeddings_factory=self.embeddings_factory,
            table_name=self.table_name,
            config=self.config,
            conf=self._conf,
        )

    def delete_collection(self) -> None:
        """Delete the vector store collection.

        Raises:
            NotImplementedError: For unsupported vector store backends
        """
        if self.backend == "PgVector":
            from langchain_postgres import PGEngine

            if pg_engine := self._conf.get("pg_engine"):
                logger.info(f"drop table {self._conf['schema_name']}.{self._conf['table_name']}")
                assert isinstance(pg_engine, PGEngine)
                pg_engine.drop_table(table_name=self._conf["table_name"], schema_name=self._conf["schema_name"])
            else:
                raise NotImplementedError(f"Don't know how to delete collection for {self.get()}")
        else:
            raise NotImplementedError(f"Delete collection not implemented for {self.backend}")


def search_one(vc: VectorStore, query: str) -> list[Document]:
    """Perform a similarity search to find the single most relevant document.

    Args:
        vc: Vector store to search in
        query: Search query string

    Returns:
        List containing the most similar document
    """
    return vc.similarity_search(query, k=1)


if __name__ == "__main__":
    """Quick test script for hybrid search functionality."""

    from langchain_core.documents import Document

    from genai_tk.core.embeddings_factory import EmbeddingsFactory

    # Test configuration
    os.environ["POSTGRES_USER"] = "tcl"
    os.environ["POSTGRES_PASSWORD"] = "tcl"
    global_config_reload()

    postgres_url = global_config().get_dsn("vector_store.postgres_url", driver="asyncpg")

    print("🧪 Testing hybrid search with PostgreSQL...")

    # Create embeddings factory
    embeddings_factory = EmbeddingsFactory(embeddings_id="embeddings_768_fake")
    embeddings_factory_cached = EmbeddingsFactory(embeddings_id="embeddings_768_fake", cache_embeddings=True)

    # Note: This example shows legacy usage patterns for testing purposes
    # In production code, use VectorStoreRegistry.create_from_config() instead

    # Create test configuration temporarily
    import tempfile

    import yaml

    test_config = {
        "vector_store_registry": {
            "test_hybrid": {
                "backend": "PgVector",
                "embeddings": "fake",
                "table_name_prefix": "test_embeddings_1",
                "config": {
                    "postgres_url": str(postgres_url),
                    "metadata_columns": [{"name": "test_metadata", "data_type": "TEXT"}],
                    "hybrid_search": True,
                    "hybrid_search_config": {
                        "tsv_lang": "pg_catalog.english",
                        "fusion_function_parameters": {
                            "primary_results_weight": 0.5,
                            "secondary_results_weight": 0.5,
                        },
                    },
                },
            },
            "test_cache": {
                "backend": "PgVector",
                "embeddings": "fake",
                "table_name_prefix": "test_embeddings_2",
                "config": {
                    "postgres_url": str(postgres_url),
                },
            },
        }
    }

    # Write temporary config and reload
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(test_config, f)
        temp_config_path = f.name

    # Reload config with test configuration
    global_config_reload(additional_configs=[temp_config_path])

    # Create factories using config
    factory = VectorStoreRegistry.create_from_config("test_hybrid")

    print("🧪 Testing with cache_embedding...")
    cache_factory = VectorStoreRegistry.create_from_config("test_cache")

    print("📄 Adding test documents with cached embeddings...")
    cache_test_docs = [
        Document(page_content="Cached embedding test document 1"),
        Document(page_content="Cached embedding test document 2"),
    ]
    cache_factory.add_documents(cache_test_docs)
    print("✅ Successfully added documents with cached embeddings")

    try:
        # Add test documents
        test_docs = [
            Document(page_content="PostgreSQL is a powerful open-source database system"),
            Document(page_content="Hybrid search combines vector similarity and full-text search"),
            Document(page_content="GIN indexes are used for full-text search in PostgreSQL"),
            Document(page_content="LangChain provides excellent vector store integration"),
        ]

        print("📄 Adding test documents...")
        factory.add_documents(test_docs)

        # Perform hybrid search
        print("🔍 Performing hybrid search...")
        results = factory.get().similarity_search(
            "database search",
            k=2,
            hybrid_search_config=HybridSearchConfig(
                tsv_column="content_tsv",
                fusion_function_parameters={"primary_results_weight": 0.5, "secondary_results_weight": 0.5},
            ),
        )

        print(f"✅ Found {len(results)} results:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content}")

    except Exception as e:
        print(f"❌ Error: {e}")
        debug(e)
        raise e
        print("💡 Make sure PostgreSQL is running and POSTGRES_URL is set correctly")

    finally:
        # Clean up
        try:
            os.unlink(temp_config_path)
            print("🧹 Cleaned up temporary config file")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up temp config - {e}")
