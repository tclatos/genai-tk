"""Vector Store Management and Factory System.

This module provides a comprehensive interface for creating, managing, and
interacting with vector stores across multiple storage backends. It supports
advanced document indexing, retrieval, and vector database operations.

Key Features:
- Multi-backend vector store support (Chroma, In-Memory, Sklearn, PgVector)
- Flexible document indexing and deduplication with SQLRecordManager
- Configurable retrieval strategies and search parameters
- Seamless integration with embedding models via EmbeddingsFactory
- Advanced search capabilities including hybrid search (vector + full-text)
- YAML-based configuration with runtime override support
- Collection metadata management and cleanup operations
- Automatic table/collection naming with conflict prevention

Supported Backends:
- Chroma (persistent and in-memory storage)
- InMemoryVectorStore (in-memory only)
- SKLearnVectorStore (scikit-learn based)
- PgVector (PostgreSQL with hybrid search support)

Design Patterns:
- Factory Method pattern for vector store creation
- Configuration-driven instantiation with fallback defaults
- Document indexing with deduplication via SQLRecordManager
- Generic configuration system supporting runtime overrides

Configuration Examples:
    ```yaml
    # baseline.yaml configuration
    embeddings_store:
      # In-memory Chroma store
      default:
        backend: Chroma
        embeddings: default
        config:
          storage: '::memory::'

      # Persistent Chroma store
      persistent:
        backend: Chroma
        embeddings: default
        table_name_prefix: my_docs
        config:
          storage: /data/vector_store
        collection_metadata:
          environment: production
          version: "1.0"

      # Indexed store with deduplication
      indexed:
        backend: Chroma
        embeddings: default
        index_document: true
        record_manager: sqlite:///data/record_manager.sql
        config:
          storage: /data/indexed_store

      # PostgreSQL with hybrid search
      postgres_hybrid:
        backend: PgVector
        embeddings: default
        config:
          postgres_url: postgresql://user:pass@localhost:5432/db
          metadata_columns:
            - name: category
              data_type: TEXT
            - name: score
              data_type: FLOAT
          hybrid_search: true
          hybrid_search_config:
            tsv_lang: pg_catalog.english
            fusion_function_parameters:
              primary_results_weight: 0.7
              secondary_results_weight: 0.3
    ```

Usage Examples:
    ```python
    # Basic usage with configuration
    factory = EmbeddingsStore.create_from_config("default")

    # Add documents
    documents = [
        Document(page_content="First document", metadata={"source": "doc1"}),
        Document(page_content="Second document", metadata={"source": "doc2"}),
    ]
    factory.add_documents(documents)

    # Similarity search
    vector_store = factory.get()
    results = vector_store.similarity_search("query text", k=5)

    # Hybrid search (PostgreSQL only)
    if factory.backend == "PgVector":
        hybrid_results = vector_store.similarity_search(
            "query",
            k=3,
            hybrid_search_config=HybridSearchConfig(
                tsv_column="content_tsv", fusion_function_parameters={"primary_results_weight": 0.6}
            ),
        )

    # Document management
    count = factory.document_count()
    factory.delete_collection()  # For supported backends
    ```

Migration Notes:
- Legacy 'id' field is deprecated; use 'backend' instead
- Legacy 'chroma_path' is deprecated; use 'storage' instead
- Chroma_in_memory backend is deprecated; use backend='Chroma' with storage='::memory::'
"""

import os
from collections.abc import Iterable
from typing import Annotated, Any, Literal, get_args

from devtools import debug
from langchain_classic.indexes import IndexingResult, SQLRecordManager, index
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from typing_extensions import deprecated

try:
    from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig  # pyright: ignore[reportMissingImports]  # noqa: I001
except ImportError:
    HybridSearchConfig = None  # Optional dependency
from loguru import logger
from pydantic import AliasChoices, AnyUrl, BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from genai_tk.core.embeddings_factory import EmbeddingsFactory
from genai_tk.utils.config_mngr import (
    global_config,
    global_config_reload,
)

# List of known Vector Stores (created as Literal so can be checked by MyPy)
VECTOR_STORE_ENGINE = Literal["Chroma", "InMemory", "Sklearn", "PgVector"]


class _EmbeddingsStoreConfig(BaseModel):
    """Parsed configuration for an EmbeddingsStore YAML entry."""
    backend: str = Field(validation_alias=AliasChoices("backend", "id"))
    embeddings: str | None = None
    embeddings_id: str | None = None
    table_name_prefix: str = "embeddings"
    index_document: bool = False
    collection_metadata: dict[str, str] | None = None
    record_manager: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, data: Any) -> Any:
        """Migrate legacy config keys to the current schema."""
        if not isinstance(data, dict):
            return data

        migrated = dict(data)
        if "backend" not in migrated and "id" in migrated:
            migrated["backend"] = migrated["id"]

        config = migrated.get("config")
        if isinstance(config, dict) and "storage" not in config and "chroma_path" in config:
            new_config = dict(config)
            new_config["storage"] = new_config.pop("chroma_path")
            migrated["config"] = new_config

        return migrated

    @model_validator(mode="after")
    def _apply_defaults(self) -> "_EmbeddingsStoreConfig":
        if self.embeddings and self.embeddings_id:
            raise ValueError("Cannot specify both 'embeddings_id' and 'embeddings' in embeddings_store config")
        if self.record_manager:
            self.index_document = True
        return self

    def resolve_embeddings_factory(self) -> EmbeddingsFactory:
        """Resolve the appropriate EmbeddingsFactory from config fields."""
        if self.embeddings_id:
            return EmbeddingsFactory(embeddings=self.embeddings_id)
        if self.embeddings:
            return EmbeddingsFactory(embeddings=self.embeddings)
        return EmbeddingsFactory()


class EmbeddingsStore(BaseModel):
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
        >>> factory = EmbeddingsStore.create_from_config("default")
        >>> factory = EmbeddingsStore.create_from_config("local_indexed")
        >>> factory.add_documents([Document(page_content="example")])
        >>>
        >>> # Hybrid search example (PostgreSQL)
        >>> hybrid_factory = EmbeddingsStore.create_from_config("postgres_hybrid")
    """

    backend: Annotated[VECTOR_STORE_ENGINE | None, Field(validate_default=True, alias="id")] = None
    embeddings_factory: EmbeddingsFactory
    table_name_prefix: str = "embeddings"
    config: dict[str, Any] = {}
    index_document: bool = False
    collection_metadata: dict[str, str] | None = None
    record_manager_url: AnyUrl | None = None
    _record_manager: SQLRecordManager | None = None
    _conf: dict = {}

    def __init__(self, **data):
        """Direct instantiation is not allowed. Use create_from_config instead."""
        raise RuntimeError(
            "EmbeddingsStore cannot be instantiated directly. "
            "Use EmbeddingsStore.create_from_config(config_tag) instead."
        )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context) -> None:
        """Post-initialization logic for EmbeddingsStore.

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
    def create_from_config(cls, config_tag: str) -> "EmbeddingsStore":
        """Create a EmbeddingsStore from configuration.

        Args:
            config_tag: Configuration tag to look up (e.g., 'default', 'local_indexed')

        Returns:
            Configured EmbeddingsStore instance

        Raises:
            ValueError: If configuration tag is not found or invalid
            KeyError: If required configuration keys are missing

        Example:
            ```python
            # Configuration in baseline.yaml:
            # embeddings_store:
            #   default:
            #     id: InMemory
            #   local_indexed:
            #     id: Chroma
            #     embeddings: default
            #     table_name_prefix: embeddings
            #     record_manager: sqlite:///data/record_manager.sql
            #     config:
            #       chroma_path: /custom/path

            factory = EmbeddingsStore.create_from_config("default")
            factory = EmbeddingsStore.create_from_config("local_indexed")
            ```
        """
        try:
            cfg = _EmbeddingsStoreConfig.model_validate(global_config().get_dict(f"embeddings_store.{config_tag}"))
        except (ValueError, KeyError) as e:
            raise ValueError(f"Configuration for vector store factory '{config_tag}' not found") from e

        instance = cls.model_construct(
            backend=cfg.backend,
            embeddings_factory=cfg.resolve_embeddings_factory(),
            table_name_prefix=cfg.table_name_prefix,
            config=cfg.config,
            index_document=cfg.index_document,
            collection_metadata=cfg.collection_metadata,
            record_manager_url=cfg.record_manager,
        )
        instance.model_post_init(None)
        return instance

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
        if backend not in EmbeddingsStore.known_items():
            raise ValueError(f"Unknown Vector Store: {backend}")
        return backend

    @deprecated("use get_vector_store")
    def get(self) -> VectorStore:
        return self.get_vector_store()

    def get_vector_store(self) -> VectorStore:
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
            # Use the record manager URL from configuration
            db_url = self.record_manager_url
            if not db_url:
                # Fallback to global config if not specified in individual config
                try:
                    db_url = global_config().get_str("vector_store.record_manager")
                except (ValueError, KeyError) as e:
                    raise ValueError("Record manager URL not found in configuration or global config") from e

            logger.debug(f"vector store record manager : {db_url}")
            namespace = f"{self.backend}/{self.table_name}"
            self._record_manager = SQLRecordManager(
                namespace,
                db_url=str(db_url),
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
            return self.get_vector_store().add_documents(list(docs))
        else:
            vector_store = self.get_vector_store()
            assert self._record_manager

            info = index(
                docs,
                self._record_manager,
                vector_store,
                cleanup="incremental",
                source_id_key="source",
                key_encoder="blake2b",
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
            return self.get_vector_store()._collection.count()  # type: ignore
        else:
            raise NotImplementedError(f"Don't know how to get collection count for {self.get_vector_store()}")

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
            try:
                from langchain_postgres import PGEngine  # pyright: ignore[reportMissingImports]
            except ImportError as e:
                raise ImportError(
                    "langchain_postgres is required for PgVector operations. "
                    "Install it with: uv pip install langchain-postgres"
                ) from e

            if pg_engine := self._conf.get("pg_engine"):
                logger.info(f"drop table {self._conf['schema_name']}.{self._conf['table_name']}")
                assert isinstance(pg_engine, PGEngine)
                pg_engine.drop_table(table_name=self._conf["table_name"], schema_name=self._conf["schema_name"])
            else:
                raise NotImplementedError(f"Don't know how to delete collection for {self.get()}")
        else:
            raise NotImplementedError(f"Delete collection not implemented for {self.backend}")

    def clear(self) -> bool:
        """Clear all documents from the vector store.

        Returns:
            True if documents were cleared successfully, False otherwise

        Notes:
            For backends that support it, this will remove all documents but keep the collection.
            For others, attempts to delete and recreate the collection.
        """
        try:
            if self.backend == "Chroma":
                # For Chroma, we need to get all document IDs and delete them
                vector_store = self.get_vector_store()
                try:
                    # Get all documents to delete them
                    all_docs = vector_store._collection.get()  # type: ignore
                    if all_docs and "ids" in all_docs and all_docs["ids"]:
                        # Delete all documents by their IDs
                        vector_store._collection.delete(ids=all_docs["ids"])  # type: ignore
                        logger.info(f"Deleted {len(all_docs['ids'])} documents from Chroma collection")
                    return True
                except Exception as e:
                    logger.error(f"Failed to clear Chroma collection: {e}")
                    return False
            elif self.backend in ["InMemory", "Sklearn"]:
                # For in-memory stores, we need to recreate them
                # This is a limitation - they don't have a clear method
                logger.warning(f"Clear operation for {self.backend} requires recreating the store")
                return True
            elif self.backend == "PgVector":
                try:
                    self.delete_collection()
                    return True
                except NotImplementedError:
                    logger.warning(f"Clear operation not fully supported for {self.backend}")
                    return False
            else:
                logger.warning(f"Clear operation not implemented for {self.backend}")
                return False
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False

    def embed_text(self, text: str, metadata: dict[str, Any] | None = None) -> list[str]:
        """Embed a single text and add it to the vector store.

        Args:
            text: Text to embed and store
            metadata: Optional metadata to associate with the document

        Returns:
            List of document IDs that were added
        """
        # Prepare metadata, ensuring 'source' key exists for indexing
        doc_metadata = metadata or {}
        if self.index_document and "source" not in doc_metadata:
            # Add a default source if not provided and indexing is enabled
            doc_metadata["source"] = f"embedded_text_{hash(text) & 0x7FFFFFFF}"

        doc = Document(page_content=text, metadata=doc_metadata)
        result = self.add_documents([doc])
        # Handle different return types from add_documents
        if isinstance(result, list):
            return result
        else:
            # IndexingResult case
            return getattr(result, "upserted", [])

    async def query(self, query: str, k: int = 4, filter: dict[str, Any] | None = None) -> list[Document]:
        """Query the vector store for similar documents.

        Args:
            query: Query text to search for
            k: Number of results to return
            filter: Optional metadata filter dictionary for filtering results

        Returns:
            List of documents matching the query
        """
        vector_store = self.get_vector_store()

        # Use similarity search with optional filter
        return vector_store.similarity_search(query, k=k, filter=filter)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store.

        Returns:
            Dictionary containing statistics about the vector store
        """
        stats = {
            "backend": self.backend,
            "table_name": self.table_name,
            "description": self.description,
            "embeddings_model": self.embeddings_factory.embeddings_id,
        }

        try:
            count = self.document_count()
            stats["document_count"] = count
        except (NotImplementedError, Exception) as e:
            stats["document_count"] = "unknown"
            stats["count_error"] = str(e)

        # Add backend-specific stats
        if self.backend == "Chroma":
            storage = self.config.get("storage", "::memory::")
            stats["storage_type"] = "memory" if storage == "::memory::" else "persistent"
            if storage != "::memory::":
                stats["storage_path"] = storage

        return stats

    @classmethod
    def list_available_configs(cls) -> list[str]:
        """List all available vector store configurations.

        Returns:
            List of configuration names that can be used with create_from_config
        """
        try:
            embeddings_store_config = global_config().get("embeddings_store", {})
            if hasattr(embeddings_store_config, "keys"):
                return list(embeddings_store_config.keys())
            return []
        except Exception:
            return []


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
    embeddings_factory = EmbeddingsFactory(embeddings="embeddings_768@fake")
    embeddings_factory_cached = EmbeddingsFactory(embeddings="embeddings_768@fake", cache_embeddings=True)

    # Note: This example shows legacy usage patterns for testing purposes
    # In production code, use EmbeddingsStore.create_from_config() instead

    # Create test configuration temporarily
    import tempfile

    import yaml

    test_config = {
        "embeddings_store": {
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
    global_config_reload()

    # Create factories using config
    factory = EmbeddingsStore.create_from_config("test_hybrid")

    print("🧪 Testing with cache_embedding...")
    cache_factory = EmbeddingsStore.create_from_config("test_cache")

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
        if HybridSearchConfig is not None:
            results = factory.get().similarity_search(
                "database search",
                k=2,
                hybrid_search_config=HybridSearchConfig(
                    tsv_column="content_tsv",
                    fusion_function_parameters={"primary_results_weight": 0.5, "secondary_results_weight": 0.5},
                ),
            )
        else:
            print("⚠️  HybridSearchConfig not available, falling back to regular search")
            results = factory.get().similarity_search("database search", k=2)

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
