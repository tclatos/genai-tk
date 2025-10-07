# WIP
# Taken from : https://colab.research.google.com/drive/1Lsn6W7rsKTG3a_LzM5u6SLwcw9g-6pvo?usp=sharing#scrollTo=YhcwvAcXB8CD


import asyncio
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.callbacks.manager import AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field, validator
from zeroentropy import AsyncZeroEntropy, ConflictError


class ZeroEntropyConfig(BaseModel):
    """Configuration model for ZeroEntropy retriever.

    This configuration follows ZeroEntropy SDK specifications while
    maintaining LangChain compatibility.
    """

    collection_name: str = Field(description="Name of the ZeroEntropy collection to query")

    k: int = Field(default=5, ge=1, le=2048, description="Number of results to retrieve (1-2048 for documents)")

    retrieval_type: str = Field(
        default="documents", description="Retrieval granularity: 'documents', 'pages', or 'snippets'"
    )

    include_metadata: bool = Field(default=True, description="Include metadata in document results")

    include_content: bool = Field(
        default=True, description="Include full content for pages/snippets (not applicable for documents)"
    )

    filter_criteria: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata filters using ZeroEntropy filter syntax"
    )

    latency_mode: str = Field(
        default="low", description="Latency/accuracy tradeoff: 'low' (faster) or 'high' (more accurate)"
    )

    precise_responses: bool = Field(default=False, description="Enable precise mode for snippet retrieval")

    reranker: Optional[str] = Field(
        default=None, description="Optional reranker model (e.g., 'zerank-1-small', 'zerank-1-large')"
    )

    @validator("retrieval_type")
    def validate_retrieval_type(cls, v):
        """Ensure retrieval type is valid."""
        valid_types = ["documents", "pages", "snippets"]
        if v not in valid_types:
            raise ValueError(f"retrieval_type must be one of {valid_types}")
        return v

    @validator("latency_mode")
    def validate_latency_mode(cls, v):
        """Ensure latency mode is valid."""
        if v not in ["low", "high"]:
            raise ValueError('latency_mode must be "low" or "high"')
        return v

    @validator("k")
    def validate_k_limits(cls, v, values):
        """Validate k based on retrieval type limits."""
        retrieval_type = values.get("retrieval_type", "documents")
        max_limits = {"documents": 2048, "pages": 1024, "snippets": 128}
        max_k = max_limits.get(retrieval_type, 2048)
        if v > max_k:
            raise ValueError(f"k must be <= {max_k} for {retrieval_type} retrieval")
        return v


class ZeroEntropyRetriever(BaseRetriever):
    """LangChain retriever for ZeroEntropy SDK.

    This retriever implements the standard LangChain BaseRetriever interface
    while providing full access to ZeroEntropy's advanced retrieval capabilities.

    Attributes:
        config: ZeroEntropyConfig object containing all retriever settings

    Examples:
        Basic usage:
        ```python
        retriever = ZeroEntropyRetriever(collection_name="my_docs")
        docs = retriever.invoke("What is machine learning?")
        ```

        Advanced configuration:
        ```python
        retriever = ZeroEntropyRetriever(
            collection_name="technical_docs",
            k=10,
            retrieval_type="snippets",
            reranker="zerank-1-large",
            filter_criteria={"category": "ml", "year": {"$gte": 2020}}
        )
        ```

        Runtime configuration with LangChain:
        ```python
        retriever = ZeroEntropyRetriever(collection_name="docs")
        configurable_retriever = retriever.configurable_fields(
            k=ConfigurableField(id="k", name="Results Count"),
            reranker=ConfigurableField(id="reranker", name="Reranker Model")
        )
        configured = configurable_retriever.with_config(
            configurable={"k": 15, "reranker": "zerank-1-small"}
        )
        ```
    """

    config: ZeroEntropyConfig
    """Configuration object containing all ZeroEntropy retriever parameters."""

    _client: Optional[AsyncZeroEntropy] = None
    """Private client instance for lazy initialization."""

    def __init__(
        self,
        collection_name: str,
        k: int = 5,
        retrieval_type: str = "documents",
        include_metadata: bool = True,
        include_content: bool = True,
        filter_criteria: Optional[Dict[str, Any]] = None,
        latency_mode: str = "low",
        precise_responses: bool = False,
        reranker: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the ZeroEntropy retriever.

        Args:
            collection_name: Name of the ZeroEntropy collection to query
            k: Number of results to retrieve (default: 5)
            retrieval_type: Type of retrieval - 'documents', 'pages', or 'snippets' (default: 'documents')
            include_metadata: Whether to include metadata in results (default: True)
            include_content: Whether to include content for pages/snippets (default: True)
            filter_criteria: Optional metadata filters using ZeroEntropy syntax
            latency_mode: 'low' for faster or 'high' for more accurate retrieval (default: 'low')
            precise_responses: Use precise mode for snippet retrieval (default: False)
            reranker: Optional reranker model name (e.g., 'zerank-1-small')
            **kwargs: Additional arguments passed to BaseRetriever

        Raises:
            ValueError: If configuration parameters are invalid
        """
        super().__init__(**kwargs)

        self.config = ZeroEntropyConfig(
            collection_name=collection_name,
            k=k,
            retrieval_type=retrieval_type,
            include_metadata=include_metadata,
            include_content=include_content,
            filter_criteria=filter_criteria,
            latency_mode=latency_mode,
            precise_responses=precise_responses,
            reranker=reranker,
        )

    @property
    def client(self) -> AsyncZeroEntropy:
        """Lazy initialization of ZeroEntropy client.

        The client uses ZEROENTROPY_API_KEY from environment variables.
        """
        if self._client is None:
            self._client = AsyncZeroEntropy()
        return self._client

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Synchronous document retrieval (required by LangChain).

        Args:
            query: Search query string
            run_manager: LangChain callback manager for retriever events

        Returns:
            List of LangChain Document objects
        """
        try:
            # Use asyncio.run for sync compatibility as per LangChain patterns
            return asyncio.run(self._aget_relevant_documents_internal(query, run_manager))
        except Exception as e:
            if run_manager:
                run_manager.on_retriever_error(e)
            # Return empty list on error to maintain chain execution
            return []

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronous document retrieval (required by LangChain).

        Args:
            query: Search query string
            run_manager: Async callback manager for retriever events

        Returns:
            List of LangChain Document objects
        """
        return await self._aget_relevant_documents_internal(query, run_manager)

    async def _aget_relevant_documents_internal(self, query: str, run_manager=None) -> List[Document]:
        """Internal async implementation using ZeroEntropy SDK.

        This method handles the actual API calls to ZeroEntropy based on
        the configured retrieval type.
        """
        try:
            if self.config.retrieval_type == "documents":
                # Retrieve at document level
                response = await self.client.queries.top_documents(
                    collection_name=self.config.collection_name,
                    query=query,
                    k=self.config.k,
                    filter=self.config.filter_criteria,
                    include_metadata=self.config.include_metadata,
                    latency_mode=self.config.latency_mode,
                    reranker=self.config.reranker,
                )

                documents = []
                for result in response.results:
                    metadata = {
                        "path": result.path,
                        "score": result.score,
                        "retrieval_type": "document",
                        "collection": self.config.collection_name,
                        "source": "zeroentropy",
                    }
                    # Add any additional metadata from the result
                    if result.metadata:
                        metadata.update(result.metadata)

                    documents.append(Document(page_content=f"Document: {result.path}", metadata=metadata))
                return documents

            elif self.config.retrieval_type == "pages":
                # Retrieve at page level
                response = await self.client.queries.top_pages(
                    collection_name=self.config.collection_name,
                    query=query,
                    k=self.config.k,
                    filter=self.config.filter_criteria,
                    include_content=self.config.include_content,
                    latency_mode=self.config.latency_mode,
                    reranker=self.config.reranker,
                )

                documents = []
                for result in response.results:
                    # Use content if available, otherwise create summary
                    content = result.content if result.content else f"Page {result.page_index} from {result.path}"

                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "path": result.path,
                                "page_index": result.page_index,
                                "score": result.score,
                                "retrieval_type": "page",
                                "collection": self.config.collection_name,
                                "source": "zeroentropy",
                            },
                        )
                    )
                return documents

            elif self.config.retrieval_type == "snippets":
                # Retrieve at snippet level
                response = await self.client.queries.top_snippets(
                    collection_name=self.config.collection_name,
                    query=query,
                    k=self.config.k,
                    filter=self.config.filter_criteria,
                    precise_responses=self.config.precise_responses,
                    reranker=self.config.reranker,
                )

                documents = []
                for result in response.results:
                    documents.append(
                        Document(
                            page_content=result.content or "",
                            metadata={
                                "path": result.path,
                                "start_index": result.start_index,
                                "end_index": result.end_index,
                                "page_span": result.page_span,
                                "score": result.score,
                                "retrieval_type": "snippet",
                                "collection": self.config.collection_name,
                                "source": "zeroentropy",
                            },
                        )
                    )
                return documents

        except ConflictError as e:
            # Handle collection or resource conflicts
            error_msg = f"ZeroEntropy collection conflict: {e}"
            if run_manager and hasattr(run_manager, "on_retriever_error"):
                run_manager.on_retriever_error(e)

    @classmethod
    def from_collection(cls, collection_name: str, **kwargs) -> "ZeroEntropyRetriever":
        """Create retriever from a collection name (factory method).

        This follows LangChain conventions for creating retrievers.

        Args:
            collection_name: Name of the ZeroEntropy collection
            **kwargs: Additional configuration parameters

        Returns:
            Configured ZeroEntropyRetriever instance

        Example:
            ```python
            retriever = ZeroEntropyRetriever.from_collection(
                "my_documents",
                k=10,
                reranker="zerank-1-small"
            )
            ```
        """
        return cls(collection_name=collection_name, **kwargs)

    def with_config(self, **config_updates) -> "ZeroEntropyRetriever":
        """Create a new retriever with updated configuration.

        This method follows LangChain patterns for runtime configuration.

        Args:
            **config_updates: Configuration parameters to update

        Returns:
            New ZeroEntropyRetriever instance with updated configuration

        Example:
            ```python
            new_retriever = retriever.with_config(
                k=20,
                retrieval_type="snippets"
            )
            ```
        """
        # Create new config with updates
        new_config = self.config.copy(update=config_updates)

        # Return new instance with updated config
        return ZeroEntropyRetriever(
            collection_name=new_config.collection_name,
            k=new_config.k,
            retrieval_type=new_config.retrieval_type,
            include_metadata=new_config.include_metadata,
            include_content=new_config.include_content,
            filter_criteria=new_config.filter_criteria,
            latency_mode=new_config.latency_mode,
            precise_responses=new_config.precise_responses,
            reranker=new_config.reranker,
        )

    def get_lc_namespace(self) -> List[str]:
        """Get LangChain namespace for serialization.

        Returns:
            List defining the namespace hierarchy
        """
        return ["zeroentropy", "retrievers"]

    @property
    def lc_serializable(self) -> bool:
        """Indicate this retriever is serializable by LangChain.

        Returns:
            True to enable LangChain serialization
        """
        return True

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        """Get attributes for LangChain serialization.

        Returns:
            Dictionary of serializable attributes
        """
        return {
            "collection_name": self.config.collection_name,
            "k": self.config.k,
            "retrieval_type": self.config.retrieval_type,
            "include_metadata": self.config.include_metadata,
            "include_content": self.config.include_content,
            "filter_criteria": self.config.filter_criteria,
            "latency_mode": self.config.latency_mode,
            "precise_responses": self.config.precise_responses,
            "reranker": self.config.reranker,
        }


# ============= HELPER UTILITIES =============


class ZeroEntropyCollectionManager:
    """Utility class for managing ZeroEntropy collections.

    This helper provides convenient methods for setting up and managing
    collections outside of the retrieval flow.
    """

    def __init__(self):
        """Initialize the collection manager.

        Loads environment variables and creates client instance.
        """
        load_dotenv()
        self.client = AsyncZeroEntropy()

    async def create_collection(self, collection_name: str, description: Optional[str] = None) -> bool:
        """Create a new collection.

        Args:
            collection_name: Name for the new collection
            description: Optional collection description

        Returns:
            True if successful, False if already exists
        """
        try:
            await self.client.collections.add(collection_name=collection_name, description=description)
            return True
        except ConflictError:
            # Collection already exists
            return False

    async def add_document(
        self,
        collection_name: str,
        path: str,
        content: Union[str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> bool:
        """Add a document to a collection.

        Args:
            collection_name: Target collection name
            path: Document path/identifier
            content: Document content (text or structured)
            metadata: Optional document metadata
            overwrite: Whether to overwrite existing document

        Returns:
            True if successful
        """
        try:
            # Format content for ZeroEntropy
            if isinstance(content, str):
                formatted_content = {"type": "text", "text": content}
            else:
                formatted_content = content

            await self.client.documents.add(
                collection_name=collection_name,
                path=path,
                content=formatted_content,
                metadata=metadata or {},
                overwrite=overwrite,
            )
            return True
        except Exception as e:
            print(f"Error adding document: {e}")
            return False

    async def add_documents_batch(
        self, collection_name: str, documents: List[Dict[str, Any]], overwrite: bool = False
    ) -> int:
        """Add multiple documents to a collection.

        Args:
            collection_name: Target collection name
            documents: List of document dictionaries with 'path', 'content', and optional 'metadata'
            overwrite: Whether to overwrite existing documents

        Returns:
            Number of successfully added documents
        """
        success_count = 0
        for doc in documents:
            success = await self.add_document(
                collection_name=collection_name,
                path=doc["path"],
                content=doc["content"],
                metadata=doc.get("metadata"),
                overwrite=overwrite,
            )
            if success:
                success_count += 1
        return success_count

    async def get_collection_status(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get status information for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Status dictionary or None if not found
        """
        try:
            status = await self.client.status.get(collection_name=collection_name)
            return status
        except Exception:
            return None
