"""Factory for creating and managing key-value stores with LangChain storage backends.

Provides a factory pattern for creating ByteStore instances with support for
different storage backends like local file storage and PostgreSQL.

Supports configuration-based store creation with multiple named configurations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from langchain.storage import LocalFileStore
from langchain_core.stores import ByteStore
from pydantic import BaseModel, Field

from genai_tk.utils.config_mngr import global_config


class KvStoreConfig(BaseModel, ABC):
    """Base configuration for key-value stores.
    
    Attributes:
        type: The type of store (LocalFileStore, SQLStore, etc.)
    """
    
    type: str
    
    @abstractmethod
    def create_store(self, namespace: str = "") -> ByteStore:
        """Create a ByteStore instance based on this configuration.
        
        Args:
            namespace: Optional namespace or root directory for the store
            
        Returns:
            ByteStore: Configured ByteStore instance
        """
        pass


class LocalFileStoreConfig(KvStoreConfig):
    """Configuration for local file-based key-value store.
    
    Attributes:
        type: Always "LocalFileStore"
        path: Path to the storage directory
    """
    
    type: str = Field(default="LocalFileStore")
    path: str | Path
    
    def create_store(self, namespace: str = "") -> ByteStore:
        """Create a LocalFileStore instance.
        
        Args:
            namespace: Optional subdirectory within the store path
            
        Returns:
            LocalFileStore: Configured LocalFileStore instance
        """
        from genai_tk.utils.config_mngr import global_config
        
        # Use the path directly - template expansion should happen at config level
        expanded_path = self.path
            
        base_path = Path(expanded_path)
        if namespace:
            store_path = base_path / namespace
        else:
            store_path = base_path
            
        # Create directory if it doesn't exist
        store_path.mkdir(parents=True, exist_ok=True)
        
        return LocalFileStore(str(store_path))


class SQLStoreConfig(KvStoreConfig):
    """Configuration for SQL-based key-value store.
    
    Attributes:
        type: Always "SQLStore"
        path: Database connection URL
    """
    
    type: str = Field(default="SQLStore")
    path: str
    
    def create_store(self, namespace: str = "") -> ByteStore:
        """Create a SQLStore instance.
        
        Args:
            namespace: Optional namespace for the store
            
        Returns:
            SQLStore: Configured SQLStore instance
        """
        from langchain_community.storage import SQLStore
        
        store = SQLStore(namespace=namespace, db_url=self.path)
        store.create_schema()
        return store


class KvStoreRegistry(BaseModel):
    """Registry for creating key-value stores with configurable backends.
    
    This registry supports multiple named store configurations, allowing different
    store types and settings to be used based on configuration.
    
    Examples:
        ```python
        # Get default store
        registry = KvStoreRegistry()
        store = registry.get()
        
        # Get specific named store
        store = registry.get(store_id="postgres")
        
        # Get store with custom namespace
        store = registry.get(namespace="my_data")
        ```
    """
    
    def _get_store_config(self, store_id: str) -> KvStoreConfig:
        """Get store configuration for the specified store ID.
        
        Args:
            store_id: The store configuration ID
            
        Returns:
            KvStoreConfig: The store configuration
            
        Raises:
            ValueError: If store_id is not found or type is unsupported
        """
        # Try new configuration format first
        try:
            config_dict = global_config().get_dict(f"kv_store.{store_id}")
            
            if "type" in config_dict:
                # New format with explicit type
                store_type = config_dict["type"]
                
                if store_type == "LocalFileStore":
                    # Use raw path - we'll create the directory in create_store
                    return LocalFileStoreConfig(**config_dict)
                elif store_type == "SQLStore":
                    # Resolve DSN using config manager
                    if "path" in config_dict:
                        config_dict["path"] = global_config().get_dsn(f"kv_store.{store_id}.path", driver=None)
                    return SQLStoreConfig(**config_dict)
                else:
                    raise ValueError(f"Unsupported store type: '{store_type}'")
            else:
                # Legacy format - try to infer type from store_id or content
                if store_id == "default":
                    # Default case - check for legacy "file" configuration
                    try:
                        file_config = global_config().get_dict("kv_store.file")
                        if "path" in file_config:
                            return LocalFileStoreConfig(type="LocalFileStore", **file_config)
                    except:
                        pass
                
                # Could be a legacy direct field config or other pattern
                if "path" in config_dict:
                    # Infer type based on store_id and path content
                    path_value = config_dict["path"]
                    if store_id == "sql" or (isinstance(path_value, str) and (path_value.startswith("postgresql://") or path_value.startswith("sqlite://"))):
                        # SQL store based on store_id or DSN pattern
                        config_dict["path"] = global_config().get_dsn(f"kv_store.{store_id}.path", driver=None)
                        return SQLStoreConfig(type="SQLStore", **config_dict)
                    else:
                        # Assume LocalFileStore for other cases - use raw path
                        return LocalFileStoreConfig(type="LocalFileStore", **config_dict)
                else:
                    raise ValueError(f"Store configuration '{store_id}' must specify a 'type' field or 'path' field")
        except Exception as e:
            # Check for global legacy configuration patterns only for "default" store_id
            if store_id == "default":
                try:
                    # Check for legacy "engine" configuration
                    engine = global_config().get_str("kv_store.engine", default=None)
                    if engine == "memory":
                        # For tests - create in-memory file store
                        import tempfile
                        temp_dir = tempfile.mkdtemp()
                        return LocalFileStoreConfig(type="LocalFileStore", path=temp_dir)
                    elif engine:
                        raise ValueError(f"Unsupported legacy kv_store engine: '{engine}'")
                except:
                    pass
            
            # Re-raise the original error if no legacy pattern works
            raise ValueError(f"Store configuration '{store_id}' not found: {e}")
    
    def get(self, store_id: str = "default", namespace: str = "") -> ByteStore:
        """Create and return a ByteStore instance based on configuration.
        
        Args:
            store_id: The store configuration ID to use (default: "default")
            namespace: Optional namespace or root directory for the storage
            
        Returns:
            ByteStore: A configured ByteStore instance
            
        Raises:
            ValueError: If store_id is not found or configuration is invalid
            
        Examples:
            ```python
            registry = KvStoreRegistry()
            
            # Get default store
            store = registry.get()
            
            # Get postgres store with namespace
            store = registry.get(store_id="postgres", namespace="user_data")
            ```
        """
        config = self._get_store_config(store_id)
        return config.create_store(namespace)
    
    @staticmethod
    def get_available_stores() -> list[str]:
        """Get list of available store configurations.
        
        Returns:
            list[str]: List of available store IDs
        """
        kv_config = global_config().get_dict("kv_store")
        return list(kv_config.keys())


def get_kv_store(store_id: str = "default", namespace: str = "") -> ByteStore:
    """Create a configured ByteStore instance.
    
    This is a convenience function that creates a KvStoreRegistry and returns
    the requested store.
    
    Args:
        store_id: The store configuration ID to use (default: "default")
        namespace: Optional namespace or root directory for the storage
        
    Returns:
        ByteStore: A configured ByteStore instance
        
    Examples:
        ```python
        # Get default store
        store = get_kv_store()
        
        # Get postgres store
        store = get_kv_store(store_id="postgres")
        
        # Get store with namespace
        store = get_kv_store(store_id="default", namespace="cache")
        ```
    """
    registry = KvStoreRegistry()
    return registry.get(store_id=store_id, namespace=namespace)
