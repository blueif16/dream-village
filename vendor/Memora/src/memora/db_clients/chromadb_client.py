# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
ChromaDB implementation of the vector database client.
"""
import os
import threading
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig
import chromadb
from chromadb.api.types import Where
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.types import normalize_embeddings, validate_embeddings

from memora.db_clients.base import CONSOLIDATION_SUFFIX, VectorDBClient
from memora.utils.embedding import BaseEmbeddingModel


class ChromaDBEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function for ChromaDB."""

    def __init__(self, cfg: DictConfig, document_kind: str = "document", query_kind: str = "query"):
        self.cfg = cfg
        self.embedding_model = BaseEmbeddingModel(cfg)
        self.document_kind = document_kind
        self.query_kind = query_kind

    def __call__(self, input: Documents) -> Embeddings:
        """Embed the documents using our embedding model."""
        return self.embedding_model.generate_embeddings(input, kind=self.document_kind)

    def embed_query(self, input: Documents) -> Embeddings:
        """Chroma calls this for collection.query(query_texts=...)."""
        embeddings = self.embedding_model.generate_embeddings(input, kind=self.query_kind)
        return validate_embeddings(normalize_embeddings(embeddings))


class ChromaDBClient(VectorDBClient):
    """ChromaDB implementation of the vector database client."""

    # Thread-safe singleton cache for PersistentClient instances per path
    _clients: Dict[str, Any] = {}
    _lock = threading.Lock()

    @classmethod
    def _get_or_create_client(cls, persist_path: str):
        """Get or create a shared PersistentClient for the given path (thread-safe)."""
        path_key = os.path.abspath(persist_path)
        if path_key not in cls._clients:
            with cls._lock:
                if path_key not in cls._clients:
                    cls._clients[path_key] = chromadb.PersistentClient(path=path_key)
        return cls._clients[path_key]

    def __init__(self, cfg: DictConfig):
        """
        Initialize ChromaDB client.

        Args:
            cfg: Configuration object containing database settings
        """
        self.cfg = cfg
        persist_path = cfg.memory.persist_path
        self.client = self._get_or_create_client(persist_path)
        self.embedding_function = ChromaDBEmbeddingFunction(cfg)
        # memo-bench port (1b): `<name>__consol` collections embed docs and queries symmetrically
        self.consolidation_embedding_function = ChromaDBEmbeddingFunction(
            cfg, document_kind="consolidation", query_kind="consolidation"
        )

    def get_or_create_collection(self, collection_name: str, metadata: Dict[str, Any]):
        """
        Get or create a ChromaDB collection.

        Args:
            collection_name: Name of the collection
            metadata: Collection metadata (e.g., distance metric)

        Returns:
            ChromaDB collection object
        """
        if collection_name.endswith(CONSOLIDATION_SUFFIX):
            embedding_function = self.consolidation_embedding_function
        else:
            embedding_function = self.embedding_function
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata=metadata,
            embedding_function=embedding_function,
        )

    def upsert(
        self,
        collection,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ):
        """
        Insert or update records in the ChromaDB collection.

        Args:
            collection: ChromaDB collection object
            ids: List of record IDs
            documents: List of document texts
            metadatas: List of metadata dicts
            embeddings: Optional pre-computed embeddings
        """
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(
        self,
        collection,
        query_texts: str,
        n_results: int,
        where: Optional[Where] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Query the ChromaDB collection for similar vectors.

        Args:
            collection: ChromaDB collection object
            query_texts: Query string
            n_results: Number of results to return
            where: Filter conditions
            include: Fields to include in results

        Returns:
            Query results dictionary
        """
        return collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            include=include,
        )

    def get(
        self,
        collection,
        ids: Optional[List[str]] = None,
        where: Optional[Where] = None,
        include: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get records from the ChromaDB collection.

        Args:
            collection: ChromaDB collection object
            ids: Optional list of IDs to retrieve
            where: Filter conditions
            include: Fields to include in results
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            Records dictionary
        """
        return collection.get(
            ids=ids,
            where=where,
            include=include,
            limit=limit,
            offset=offset,
        )

    def delete(self, collection, ids: List[str]):
        """
        Delete records from the ChromaDB collection.

        Args:
            collection: ChromaDB collection object
            ids: List of record IDs to delete
        """
        collection.delete(ids=ids)

    def count(self, collection) -> int:
        """
        Get the count of records in the ChromaDB collection.

        Args:
            collection: ChromaDB collection object

        Returns:
            Number of records
        """
        return collection.count()

    def delete_collection(self, collection_name: str):
        """
        Delete an entire ChromaDB collection.

        Args:
            collection_name: Name of the collection to delete
        """
        self.client.delete_collection(collection_name)
