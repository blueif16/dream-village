# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Abstract base class for vector database clients.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# memo-bench port (1b): name suffix of the per-user consolidation collection (vectors embedded with the
# sentence-similarity prefix); the base collection holds the retrieval vectors.
CONSOLIDATION_SUFFIX = "__consol"


class VectorDBClient(ABC):
    """Abstract base class for vector database clients."""

    @abstractmethod
    def get_or_create_collection(self, collection_name: str, metadata: Dict[str, Any]):
        """Get or create a collection in the vector database."""
        pass

    @abstractmethod
    def upsert(
        self,
        collection,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ):
        """Insert or update records in the collection."""
        pass

    @abstractmethod
    def query(
        self,
        collection,
        query_texts: str,
        n_results: int,
        where: Optional[Any] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Query the collection for similar vectors."""
        pass

    @abstractmethod
    def get(
        self,
        collection,
        ids: Optional[List[str]] = None,
        where: Optional[Any] = None,
        include: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get records from the collection."""
        pass

    @abstractmethod
    def delete(self, collection, ids: List[str]):
        """Delete records from the collection."""
        pass

    @abstractmethod
    def count(self, collection) -> int:
        """Get the count of records in the collection."""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str):
        """Delete an entire collection."""
        pass
