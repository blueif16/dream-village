# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
ChromaDB Browser

Direct browser for ChromaDB collections with interactive exploration capabilities.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import sys

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

logger = logging.getLogger(__name__)


@dataclass
class ChromaDocument:
    """
    Represents a document stored in ChromaDB.
    """
    id: str
    content: str
    metadata: Dict[str, Any]
    embeddings: Optional[List[float]] = None
    distance: Optional[float] = None


@dataclass
class ChromaStats:
    """
    Statistics about ChromaDB collection.
    """
    total_documents: int
    collection_name: str
    metadata_keys: List[str]
    unique_metadata_values: Dict[str, int]
    content_stats: Dict[str, Any]


class ChromaBrowser:
    """
    Interactive browser for ChromaDB collections.
    
    Provides direct access to ChromaDB collections with search, filtering,
    and analysis capabilities.
    """
    
    def __init__(self, db_path: str, collection_name: str = None):
        """
        Initialize ChromaDB browser.
        
        Args:
            db_path: Path to ChromaDB database directory
            collection_name: Name of collection to browse (if None, will list available)
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not available. Please install with: pip install chromadb")
            
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize ChromaDB client
        self.client = None
        self.collection = None
        self._initialize_client()
        
        # Cache for documents
        self._document_cache: List[ChromaDocument] = []
        self._cache_valid = False
        
    def _initialize_client(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Create ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # List available collections
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            
            if not collection_names:
                raise ValueError(f"No collections found in database: {self.db_path}")
                
            # Select collection
            if self.collection_name:
                if self.collection_name not in collection_names:
                    raise ValueError(f"Collection '{self.collection_name}' not found. Available: {collection_names}")
                self.collection = self.client.get_collection(self.collection_name)
            else:
                # Use first collection if none specified
                self.collection_name = collection_names[0]
                self.collection = self.client.get_collection(self.collection_name)
                
            self.logger.info(f"Connected to ChromaDB collection: {self.collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise
            
    def list_collections(self) -> List[str]:
        """
        List all available collections in the database.
        
        Returns:
            List[str]: Collection names
        """
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            self.logger.error(f"Failed to list collections: {str(e)}")
            return []
            
    def switch_collection(self, collection_name: str) -> bool:
        """
        Switch to a different collection.
        
        Args:
            collection_name: Name of collection to switch to
            
        Returns:
            bool: True if successful
        """
        try:
            self.collection = self.client.get_collection(collection_name)
            self.collection_name = collection_name
            self._cache_valid = False  # Invalidate cache
            self.logger.info(f"Switched to collection: {collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to switch collection: {str(e)}")
            return False
            
    def get_all_documents(
        self, 
        limit: Optional[int] = None,
        include_embeddings: bool = False
    ) -> List[ChromaDocument]:
        """
        Get all documents from the collection.
        
        Args:
            limit: Maximum number of documents to return
            include_embeddings: Whether to include embedding vectors
            
        Returns:
            List[ChromaDocument]: Documents from the collection
        """
        try:
            # Get documents from ChromaDB
            include_list = ["documents", "metadatas"]
            if include_embeddings:
                include_list.append("embeddings")
                
            results = self.collection.get(
                limit=limit,
                include=include_list
            )
            
            # Convert to ChromaDocument objects
            documents = []
            ids = results.get("ids", [])
            contents = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            embeddings = results.get("embeddings", []) if include_embeddings else [None] * len(ids)
            
            for i, doc_id in enumerate(ids):
                content = contents[i] if i < len(contents) else ""
                metadata = metadatas[i] if i < len(metadatas) else {}
                embedding = embeddings[i] if embeddings[i] is not None else None
                
                doc = ChromaDocument(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    embeddings=embedding
                )
                documents.append(doc)
                
            self._document_cache = documents
            self._cache_valid = True
            
            self.logger.info(f"Retrieved {len(documents)} documents from collection")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to get documents: {str(e)}")
            return []
            
    def search_documents(
        self, 
        query: str, 
        n_results: int = 10,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[ChromaDocument]:
        """
        Search documents using semantic similarity.
        
        Args:
            query: Search query
            n_results: Number of results to return
            where_filter: Metadata filter conditions
            
        Returns:
            List[ChromaDocument]: Matching documents with distances
        """
        try:
            # Perform semantic search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to ChromaDocument objects
            documents = []
            if results["ids"] and len(results["ids"]) > 0:
                ids = results["ids"][0]
                contents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                
                for i, doc_id in enumerate(ids):
                    content = contents[i] if i < len(contents) else ""
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    distance = distances[i] if i < len(distances) else None
                    
                    doc = ChromaDocument(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                        distance=distance
                    )
                    documents.append(doc)
                    
            self.logger.info(f"Found {len(documents)} matching documents")
            return documents
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return []
            
    def get_collection_stats(self) -> ChromaStats:
        """
        Get statistics about the collection.
        
        Returns:
            ChromaStats: Collection statistics
        """
        try:
            # Get all documents if not cached
            if not self._cache_valid:
                self.get_all_documents()
                
            documents = self._document_cache
            
            # Basic stats
            total_documents = len(documents)
            
            # Metadata analysis
            metadata_keys = set()
            metadata_values = {}
            content_lengths = []
            
            for doc in documents:
                # Collect metadata keys
                metadata_keys.update(doc.metadata.keys())
                
                # Count unique metadata values
                for key, value in doc.metadata.items():
                    if key not in metadata_values:
                        metadata_values[key] = set()
                    metadata_values[key].add(str(value))
                    
                # Content length
                content_lengths.append(len(doc.content))
                
            # Calculate content stats
            content_stats = {}
            if content_lengths:
                content_stats = {
                    "total_characters": sum(content_lengths),
                    "average_length": sum(content_lengths) / len(content_lengths),
                    "min_length": min(content_lengths),
                    "max_length": max(content_lengths)
                }
                
            # Count unique metadata values
            unique_metadata_counts = {
                key: len(values) for key, values in metadata_values.items()
            }
            
            return ChromaStats(
                total_documents=total_documents,
                collection_name=self.collection_name,
                metadata_keys=list(metadata_keys),
                unique_metadata_values=unique_metadata_counts,
                content_stats=content_stats
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {str(e)}")
            return ChromaStats(
                total_documents=0,
                collection_name=self.collection_name,
                metadata_keys=[],
                unique_metadata_values={},
                content_stats={}
            )
            
    def filter_documents(
        self, 
        where_filter: Dict[str, Any],
        limit: Optional[int] = None
    ) -> List[ChromaDocument]:
        """
        Filter documents by metadata conditions.
        
        Args:
            where_filter: Filter conditions
            limit: Maximum results to return
            
        Returns:
            List[ChromaDocument]: Filtered documents
        """
        try:
            results = self.collection.get(
                where=where_filter,
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            # Convert to ChromaDocument objects
            documents = []
            ids = results.get("ids", [])
            contents = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            
            for i, doc_id in enumerate(ids):
                content = contents[i] if i < len(contents) else ""
                metadata = metadatas[i] if i < len(metadatas) else {}
                
                doc = ChromaDocument(
                    id=doc_id,
                    content=content,
                    metadata=metadata
                )
                documents.append(doc)
                
            self.logger.info(f"Filtered to {len(documents)} documents")
            return documents
            
        except Exception as e:
            self.logger.error(f"Filter failed: {str(e)}")
            return []
            
    def export_documents(
        self, 
        documents: List[ChromaDocument],
        file_path: str,
        format: str = "json"
    ) -> None:
        """
        Export documents to file.
        
        Args:
            documents: Documents to export
            file_path: Output file path
            format: Export format (json, csv, txt)
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if format.lower() == "json":
                self._export_json(documents, file_path)
            elif format.lower() == "csv":
                self._export_csv(documents, file_path)
            elif format.lower() == "txt":
                self._export_txt(documents, file_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
            self.logger.info(f"Exported {len(documents)} documents to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Export failed: {str(e)}")
            raise
            
    def _export_json(self, documents: List[ChromaDocument], file_path: str) -> None:
        """Export documents to JSON format."""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "collection_name": self.collection_name,
            "total_documents": len(documents),
            "documents": [asdict(doc) for doc in documents]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    def _export_csv(self, documents: List[ChromaDocument], file_path: str) -> None:
        """Export documents to CSV format."""
        import csv
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'id', 'content_length', 'content_preview', 'metadata'
            ])
            
            # Data
            for doc in documents:
                content_preview = doc.content[:100] + '...' if len(doc.content) > 100 else doc.content
                metadata_str = json.dumps(doc.metadata)
                
                writer.writerow([
                    doc.id,
                    len(doc.content),
                    content_preview,
                    metadata_str
                ])
                
    def _export_txt(self, documents: List[ChromaDocument], file_path: str) -> None:
        """Export documents to text format."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"ChromaDB Export - {datetime.now().isoformat()}\n")
            f.write(f"Collection: {self.collection_name}\n")
            f.write(f"Total Documents: {len(documents)}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, doc in enumerate(documents, 1):
                f.write(f"Document {i}:\n")
                f.write(f"  ID: {doc.id}\n")
                f.write(f"  Length: {len(doc.content)} characters\n")
                f.write(f"  Metadata: {json.dumps(doc.metadata, indent=4)}\n")
                f.write(f"  Content: {doc.content[:500]}{'...' if len(doc.content) > 500 else ''}\n")
                f.write("-" * 40 + "\n\n")