# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Local persistent memory store implementation using ChromaDB.
"""
import logging

import json
from collections import OrderedDict
from typing import Any, Dict, List, Optional, TypeVar, Union
import time
import threading

from omegaconf import DictConfig
from chromadb.api.types import Where
from rank_bm25 import BM25Okapi

from memora.core.base import MemoryStoreBase
from memora.core.memory_entry import MemoryEntry
from memora.db_clients import VectorDBClient, create_vector_db_client
from memora.db_clients.base import CONSOLIDATION_SUFFIX
from memora.utils.embedding import BaseEmbeddingModel
from memora.utils.misc import index_to_id, extract_user_id_from_where

# Initialize logger
logger = logging.getLogger(__name__)

T = TypeVar("T")
OneOrMany = Union[T, List[T]]


class LocalMemoryStore(MemoryStoreBase):
    """
    Memory store implementation using ChromaDB PersistentClient for local file storage.
    
    This class provides persistent storage of memory records in local files using ChromaDB.
    All data is stored locally on the filesystem and persisted between sessions.
    """

    # Class-level lock management for per-user synchronization
    _user_locks = {}  # Dict[str, threading.RLock] - locks per user_id
    _locks_lock = threading.RLock()  # Lock for managing the locks dictionary

    @classmethod
    def _get_user_lock(cls, user_id: str) -> threading.RLock:
        """
        Get or create a lock for the specified user_id.
        This ensures all instances with the same user_id share the same lock.
        """
        with cls._locks_lock:
            if user_id not in cls._user_locks:
                cls._user_locks[user_id] = threading.RLock()
            return cls._user_locks[user_id]

    def __init__(self, cfg: DictConfig, user_id: str):
        """
        Initialize the local persistent memory store.
        
        Args:
            cfg: Configuration object containing memory settings
            user_id: Optional user identifier for collection isolation.
        """
        self.cfg = cfg
        self.user_id = user_id

        persist_path = cfg.memory.persist_path
        distance = cfg.memory.distance

        print("Vector database path:", persist_path)

        # Initialize vector database client based on configuration
        # Supports ChromaDB (default) and Redis - see db_type in config
        self.db_client: VectorDBClient = create_vector_db_client(cfg)
        self.embedding_model = BaseEmbeddingModel(cfg)

        self._embedding_cache = OrderedDict()
        self._cache_max_size = 300

        # Get the shared lock for this user_id (shared across all instances with same user_id)
        self._lock = self._get_user_lock(user_id)

        # Create or get user-specific collection
        # Extract email alias (part before @) for cleaner collection names
        user_alias = user_id.split('@')[0] if '@' in user_id else user_id
        self.collection_name = f"{cfg.memory.collection_name}_{user_alias}"
        self.collection = self._get_or_create_collection(self.collection_name)

        # memo-bench port (1b): factual abstractions are mirrored into a second collection whose vectors
        # use the consolidation (sentence-similarity) prefix; it serves only the consolidation candidate search.
        self.consolidation_collection_name = f"{self.collection_name}{CONSOLIDATION_SUFFIX}"
        self.consolidation_collection = self._get_or_create_collection(self.consolidation_collection_name)

        # memo-bench port (1c): "value" embeds and BM25-indexes a factual entry's value instead of its index
        self.embed_field = cfg.memory.get("embed_field", "index")
        if self.embed_field not in ("index", "value"):
            raise ValueError(f"memory.embed_field must be 'index' or 'value', got {self.embed_field!r}")

        # BM25 index per user
        self._bm25_indices = {}  # user_id -> BM25Okapi index
        self._bm25_doc_ids = {}  # user_id -> list of document IDs

    def _get_or_create_collection(self, collection_name: str):
        """Create a new collection for the user if it doesn't exist."""
        logger.info(f"Getting or creating collection: {collection_name}")
        collection = self.db_client.get_or_create_collection(
            collection_name=collection_name,
            metadata={"hnsw:space": self.cfg.memory.distance},
        )
        return collection

    @staticmethod
    def _is_factual_abstraction(metadata: Dict[str, Any]) -> bool:
        """A primary factual entry (not a cue anchor, not an episodic memory)."""
        return metadata.get("memory_type") == "factual" and not metadata.get("linked_memory")

    def _embedded_text(self, index: str, value: str, metadata: Dict[str, Any]) -> str:
        """Text that is embedded for a record (memory.embed_field ablation applies to factual entries only)."""
        if self.embed_field == "value" and self._is_factual_abstraction(metadata):
            return value
        return index

    def _get_cached_embedding(self, index: str) -> Optional[List[float]]:
        """Get embedding from cache if available, updating LRU order."""
        if index in self._embedding_cache:
            embedding = self._embedding_cache.pop(index)
            self._embedding_cache[index] = embedding
            return embedding
        return None

    def _cache_embedding(self, index: str, embedding: List[float]) -> None:
        """Cache an embedding with LRU eviction when cache is full."""
        if index in self._embedding_cache:
            del self._embedding_cache[index]

        self._embedding_cache[index] = embedding

        if len(self._embedding_cache) > self._cache_max_size:
            self._embedding_cache.popitem(last=False)

    def upsert(
        self,
        index: str,
        value: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Insert or update a memory record identified by 'index'.
        Uses in-memory embedding cache to avoid recomputing embeddings.

        Args:
            index: Natural language index to be embedded
            value: Value/content to store
            metadata: Additional metadata to store

        Returns:
            Record ID (derived from index)
        """
        with self._lock:
            rid = index_to_id(index)

            meta = {"index": index, "value": value}
            if metadata:
                # Serialize image_urls list to JSON if present
                if "image_urls" in metadata and isinstance(
                    metadata["image_urls"], list
                ):
                    metadata = metadata.copy()  # Don't modify original
                    metadata["image_urls"] = json.dumps(metadata["image_urls"])
                meta = {**meta, **metadata}

            cached_embedding = self._get_cached_embedding(index)
            document = self._embedded_text(index, value, meta)

            if cached_embedding is not None:
                self.db_client.upsert(
                    collection=self.collection,
                    ids=[rid],
                    documents=[document],
                    metadatas=[meta],
                    embeddings=[cached_embedding],
                )
            else:
                self.db_client.upsert(
                    collection=self.collection,
                    ids=[rid],
                    documents=[document],
                    metadatas=[meta],
                )

            # mirror factual abstractions into the consolidation collection (cue anchors / episodes stay out)
            if self._is_factual_abstraction(meta):
                self.db_client.upsert(
                    collection=self.consolidation_collection,
                    ids=[rid],
                    documents=[document],
                    metadatas=[meta],
                )

            return rid

    def query(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Where] = None,
        include: Optional[List[str]] = None,
        consolidation: bool = False,
    ) -> List[MemoryEntry]:
        """
        Vector search by context information to find similar memories.
        Includes retry logic to handle ChromaDB indexing delays.
        
        Args:
            query: query context string to search for
            top_k: Number of results to return
            where: Filter conditions for metadata-based filtering
            include: Fields to include in results
            consolidation: search the consolidation collection (factual abstractions,
                sentence-similarity vectors) instead of the retrieval collection
            
        Returns:
            List of MemoryEntry objects
        """
        with self._lock:
            include = include or ["metadatas", "distances"]
        collection = self.consolidation_collection if consolidation else self.collection

        # Retry logic to handle vector database indexing delays
        max_retries = 3
        retry_delay = 0.1
        entries = []
        for attempt in range(max_retries):
            try:
                result = self.db_client.query(
                    collection=collection,
                    query_texts=query,
                    n_results=top_k,
                    where=where,
                    include=include,
                )

                for metadata, distance in zip(result["metadatas"][0], result["distances"][0]):
                    metadata["score"] = 1 - distance
                    entry = MemoryEntry.from_dict(metadata)
                    entries.append(entry)
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Query attempt {attempt + 1}/{max_retries} failed, retrying in {retry_delay}s: {str(e)[:100]}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Query failed after {max_retries} attempts: {e}")
                    raise
        return entries

    def keyword_search(
        self,
        keywords: List[str],
        top_k: int = 10,
        where: Optional[Where] = None,
    ) -> List[MemoryEntry]:
        """
        Keyword-based search that matches against both index and value fields.
        Processes keywords from longest to shortest, stopping early when enough results found.
        
        For longer keywords (3+ words), allows substring matching (e.g., "sarah birthday cake" matches "birthday cake").
        For shorter keywords (1-2 words), requires exact phrase matching to avoid overly broad matches.
        
        Args:
            keywords: List of keywords/phrases to search for
            top_k: Maximum number of results to return
            where: Filter conditions for metadata-based filtering
            
        Returns:
            List of MemoryEntry objects with scores based on keyword length
        """
        with self._lock:
            # Get all documents for text-based search
            result = self.db_client.get(
                collection=self.collection,
                where=where,
                include=["metadatas"]
            )

            if not result["metadatas"]:
                return []

        # Convert keywords into lowercase for matching
        keywords_lower = [kw.lower() for kw in keywords]

        # Expand longer keywords (3+ words) to include subphrases, then deduplicate
        phrase_to_score = {}
        max_word_count = 0

        for keyword in keywords_lower:
            keyword_word_count = len(keyword.split())
            max_word_count = max(max_word_count, keyword_word_count)

            # Add the original keyword with its full score
            if keyword not in phrase_to_score or phrase_to_score[keyword] < keyword_word_count:
                phrase_to_score[keyword] = keyword_word_count

            # For longer keywords (3+ words), generate substring variants for partial matching
            if keyword_word_count >= 3:
                keyword_words = keyword.split()

                # Add all contiguous subphrases of 2+ words (but less than full length)
                for start_idx in range(len(keyword_words)):
                    for end_idx in range(start_idx + 2, len(keyword_words) + 1):
                        if end_idx - start_idx < len(keyword_words):  # Don't duplicate full phrase
                            subphrase = " ".join(keyword_words[start_idx:end_idx])
                            subphrase_word_count = end_idx - start_idx
                            # Subphrases get score based on their word count, unless already tracked with higher score
                            if subphrase not in phrase_to_score or phrase_to_score[subphrase] < subphrase_word_count:
                                phrase_to_score[subphrase] = subphrase_word_count

        # Sort all unique phrases by word count first, then by character length (both descending)
        # This prioritizes longer, more specific phrases
        phrases_sorted = sorted(
            phrase_to_score.items(), 
            key=lambda x: (len(x[0].split()), len(x[0])), 
            reverse=True
        )

        # Track matched documents with their best scores (maintains insertion order)
        matched_docs = OrderedDict()  # record_id -> (entry, score)

        # Process phrases from longest to shortest
        for phrase, phrase_score in phrases_sorted:
            phrase_word_count = len(phrase.split())

            # Search through all documents for this phrase
            for i, metadata in enumerate(result["metadatas"]):
                # Create unique record ID
                index = metadata.get("index", "")
                record_id = index_to_id(index)

                # Skip if we already found this document with a better/equal score
                if record_id in matched_docs and matched_docs[record_id][1] >= phrase_score:
                    continue

                value = metadata.get("value", "")
                searchable_text = f"{index} {value}".lower()

                # Check if phrase matches
                if phrase in searchable_text:
                    # Create or update entry with this phrase's score
                    if record_id not in matched_docs or matched_docs[record_id][1] < phrase_score:
                        metadata_copy = metadata.copy()
                        metadata_copy["score"] = float(phrase_score)
                        entry = MemoryEntry.from_dict(metadata_copy)
                        matched_docs[record_id] = (entry, phrase_score)

            # Early stopping: if we have enough high-quality results from longer phrases, stop
            if phrase_word_count <= 2 and len(matched_docs) >= top_k:
                break

        # Scale scores to be below semantic search threshold
        # Get threshold from config (default to 0.4 if not set)
        semantic_threshold = self.cfg.memory.get("query_score_threshold", 0.4)

        # Scale the scores of matched documents
        for record_id, (entry, raw_score) in matched_docs.items():
            scaled_score = (raw_score / (max_word_count)) * semantic_threshold
            entry.score = scaled_score
            matched_docs[record_id] = (entry, scaled_score)

            # Extract entries maintaining insertion order (already in priority order)
            matches = [entry for entry, _ in matched_docs.values()]

            return matches[:top_k]

    def get(self, key: str) -> MemoryEntry:
        """
        Retrieve a single record by its natural-language key.
        
        Args:
            key: Natural language key to retrieve
            
        Returns:
            Dict with id, metadata, document fields or None if not found
        """
        result = None
        with self._lock:
            record_id = index_to_id(key)

            result = self.db_client.get(
                collection=self.collection,
                ids=[record_id],
                include=["metadatas"]
            )

        # Check if result is valid
        if not result or not result["ids"]:
            return None

        # convert to MemoryEntry
        entry = MemoryEntry.from_dict(result["metadatas"][0])
        return entry

    def delete(self, key: str) -> None:
        """Delete a record by its natural-language key."""

        record_id = index_to_id(key)
        with self._lock:
            self.db_client.delete(
                collection=self.collection,
                ids=[record_id]
            )
            self.db_client.delete(
                collection=self.consolidation_collection,
                ids=[record_id]
            )

    def list_memories(self, limit: int = 20) -> List[MemoryEntry]:
        """List memories in the collection."""
        result = self.db_client.get(
            collection=self.collection,
            include=["documents", "metadatas"],
            limit=limit,
            offset=0
        )

        entries = []
        for metadata in result["metadatas"]:
            entry = MemoryEntry.from_dict(metadata)
            entries.append(entry)
        return entries

    def count(self) -> int:
        """Get the number of records in the collection."""
        return self.db_client.count(collection=self.collection)

    def clear(self) -> None:
        """Clear all records in the collection and reset embedding cache."""

        # Delete the entire collection and recreate it
        with self._lock:
            self.db_client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection(self.collection_name)
            self.db_client.delete_collection(self.consolidation_collection_name)
            self.consolidation_collection = self._get_or_create_collection(self.consolidation_collection_name)

        # clear the embedding cache
        self._embedding_cache.clear()

        # clear BM25 index for this user
        if self.user_id in self._bm25_indices:
            del self._bm25_indices[self.user_id]
        if self.user_id in self._bm25_doc_ids:
            del self._bm25_doc_ids[self.user_id]

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the embedding cache."""
        return {
            "cache_size": len(self._embedding_cache),
            "max_cache_size": self._cache_max_size,
            "cache_keys": list(self._embedding_cache.keys())[-10:],
        }

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenizer for BM25.
        Converts text to lowercase and splits on whitespace and basic punctuation.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and replace common punctuation with spaces
        text = text.lower()
        for char in ".,!?;:()[]{}\"'":
            text = text.replace(char, " ")
        # Split on whitespace and filter empty tokens
        tokens = [token for token in text.split() if token]
        return tokens

    def build_bm25_index(self, user_id: Optional[str] = None) -> None:
        """
        Build BM25 index for a specific user from their documents in the collection.
        This should be called after memories are added and before querying.
        Skips building if collection is empty.
        
        Args:
            user_id: User ID to build index for. If None, uses self.user_id.
        
        Note: Only stores tokenized corpus and document IDs per user. Actual metadata is
        fetched on-demand from ChromaDB during search to avoid duplication.
        """
        # Use provided user_id or default to self.user_id
        target_user_id = user_id or self.user_id

        # Check if collection is empty
        if self.db_client.count(self.collection) == 0:
            logger.info(f"Collection is empty, skipping BM25 index build for user {target_user_id}")
            return

        logger.info(f"Building BM25 index for user {target_user_id} in collection {self.collection_name}")

        # Get all documents from the collection
        result = self.db_client.get(
            collection=self.collection,
            include=["metadatas"]
        )

        if not result["metadatas"]:
            logger.info(f"No documents found, skipping BM25 index build for user {target_user_id}")
            return

        # Prepare tokenized corpus and track document IDs for this user
        tokenized_corpus = []
        doc_ids = []

        for metadata in result["metadatas"]:
            # Combine index and value for searchable text
            index = metadata.get("index", "")
            value = metadata.get("value", "")
            if self.embed_field == "value" and self._is_factual_abstraction(metadata):
                searchable_text = value  # 1c ablation: the index text is not indexed
            else:
                searchable_text = f"{index} {value}"

            # Tokenize and add to corpus
            tokens = self._tokenize(searchable_text)
            tokenized_corpus.append(tokens)

            # Store the document ID for later retrieval
            doc_id = index_to_id(index)
            doc_ids.append(doc_id)

        # Build BM25 index with tokenized corpus for this user
        self._bm25_indices[target_user_id] = BM25Okapi(tokenized_corpus)
        self._bm25_doc_ids[target_user_id] = doc_ids

        logger.info(f"BM25 index built for user {target_user_id} with {len(tokenized_corpus)} documents")

    def bm25_search(
        self,
        query: str,
        top_k: int = 10,
        where: Optional[Where] = None,
        score_threshold: float = 0.0,
    ) -> List[MemoryEntry]:
        """
        BM25-based search that ranks documents by relevance score for a specific user.
        
        Args:
            query: Query string to search for
            top_k: Maximum number of results to return
            where: Filter conditions for metadata-based filtering (applied post-ranking)
            score_threshold: Minimum BM25 score threshold for filtering results (default: 0.0)
            
        Returns:
            List of MemoryEntry objects with raw BM25 scores
        """
        # Extract user_id from where clause, or use self.user_id as fallback
        target_user_id = extract_user_id_from_where(where) or self.user_id

        # Check if BM25 index exists for this user, build it if not
        if target_user_id not in self._bm25_indices:
            logger.info(f"BM25 index not found for user {target_user_id}, building now...")
            self.build_bm25_index(user_id=target_user_id)

            # Check again if index was successfully built
            if target_user_id not in self._bm25_indices:
                logger.warning(f"Failed to build BM25 index for user {target_user_id}")
                return []

        # Get the user's BM25 index and doc IDs
        bm25_index = self._bm25_indices[target_user_id]
        doc_ids = self._bm25_doc_ids[target_user_id]

        # Tokenize the query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores for all documents of this user
        scores = bm25_index.get_scores(tokenized_query)

        # Create list of (doc_id, score) tuples and sort by score descending
        doc_scores = [(doc_ids[i], scores[i]) for i in range(len(scores))]
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Get top-k document IDs that meet the threshold
        top_doc_ids = [doc_id for doc_id, score in doc_scores[:top_k] if score >= score_threshold]

        if not top_doc_ids:
            return []

        # Fetch metadata from vector database for top documents
        result = self.db_client.get(
            collection=self.collection,
            ids=top_doc_ids,
            where=where,
            include=["metadatas"]
        )

        if not result["metadatas"]:
            return []

        # Create a mapping of doc_id to metadata for quick lookup
        id_to_metadata = {doc_id: metadata for doc_id, metadata in zip(result["ids"], result["metadatas"])}

        # Build result entries maintaining score order with deduplication
        matches = []
        seen_indices = set()  # Track added memory indices to avoid duplicates

        for doc_id, bm25_score in doc_scores:
            # Stop if we've collected enough results
            if len(matches) >= top_k:
                break

            # Skip if document was filtered out by where clause
            if doc_id not in id_to_metadata:
                continue

            # Use raw BM25 score directly
            metadata = id_to_metadata[doc_id].copy()
            metadata["score"] = float(bm25_score)

            entry = MemoryEntry.from_dict(metadata)

            # Resolve cue indices to their primary memories
            if entry.is_cue_index():
                # Get linked primary memories
                for primary_index in entry.get_linked_memories():
                    if primary_index in seen_indices:
                        continue  # Skip if already added

                    primary_entry = self.get(primary_index)
                    if primary_entry:
                        # Filter out episodic memories
                        if primary_entry.memory_type == "episodic":
                            continue

                        # Keep the BM25 score from the cue that matched
                        primary_entry.score = float(bm25_score)
                        matches.append(primary_entry)
                        seen_indices.add(primary_index)
            else:
                # Filter out episodic memories
                if entry.memory_type == "episodic":
                    continue

                if entry.index not in seen_indices:
                    matches.append(entry)
                    seen_indices.add(entry.index)

        return matches
