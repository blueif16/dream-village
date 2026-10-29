# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Agent Memory Facade - Provides a unified interface for different memory storage backends.
"""

# Standard library imports
import enum
from typing import Any, Dict, List, Optional, Union

# Third-party imports
from omegaconf import DictConfig
from chromadb.api.types import Where

# Local imports - Core components
from memora.core.base import MemoryBase
from memora.core.local_memory_store import LocalMemoryStore
from memora.core.memory_entry import MemoryEntry
from memora.core.memory_filter import MemoryFilter
from memora.core.query_generator import QueryGenerator

# Local imports - Utilities
from memora.core.query_generator import QueryGenerator
from memora.utils.llm import ChatCompletionModel
from memora.utils.log import log_memory_operation
from memora.utils.memory import combine_list
from memora.utils.misc import context_to_str, extract_user_id_from_where
from memora.utils.misc import context_to_str, index_to_id
import logging

# Initialize module logger
logger = logging.getLogger(__name__)


class QueryMode(enum.Enum):
    ORIGINAL = 1
    PRIMARY_ONLY = 2
    CUE_ONLY = 3
    BOTH = 4


class AgentMemory(MemoryBase):
    """
    Facade class that provides a unified interface for different memory storage backends.

    This class abstracts the choice between local persistent storage and HTTP-based storage,
    allowing users to switch between backends via configuration without changing their code.

    Supported backends:
    - 'local': Local persistent storage using ChromaDB
    - 'http': Remote storage via HTTP API calls
    """

    def __init__(self, cfg: DictConfig, user_id: str):
        """
        Initialize the memory facade with the specified backend and components.

        This constructor sets up all necessary components for memory operations:
        - Storage backend (local ChromaDB or HTTP)
        - Query generator for enhanced search capabilities
        - Memory builder for extracting memories and handling update logic
        - Similarity thresholds for intelligent memory management

        Args:
            cfg: Configuration object containing memory settings including:
                - memory.backend: Storage backend type ('local' or 'http')
                - memory.query_score_threshold: Minimum similarity for query results
                - memory.update_score_threshold: Minimum similarity for update consideration
                - memory.multimodal_support: Whether to enable multimodal processing (default: True)
                - Additional backend-specific configuration

        Raises:
            ValueError: If unsupported backend type is specified
        """
        self.cfg = cfg

        # initialize user id for this memory
        self.user_id = user_id

        # Maximum allowed context size in tokens
        self.MAX_CONTEXT_TOKENS = 128000  # gpt-4o-mini maximum context size

        # Initialize core components for memory operations
        self.query_generator = QueryGenerator(cfg)
        self.memory_filter = MemoryFilter(cfg)  # LLM-based memory filtering

        # Read multimodal support setting from config, default to True
        self.multimodal_support = cfg.memory.get("multimodal_support", True)

        self._llm_client = ChatCompletionModel(cfg)  # LLM for update decision making

        # Use local persistent storage with ChromaDB
        self._store = LocalMemoryStore(cfg, user_id)

        # Configure similarity thresholds for intelligent memory management
        # These thresholds control when memories are considered for updates vs new additions
        self.QUERY_SCORE_THRESHOLD = (
            cfg.memory.query_score_threshold
        )  # Filter query results

    def get_user_id(self) -> str:
        """
        Get the user ID associated with this memory facade.

        Returns:
            str: User ID
        """
        return self.user_id

    def _query_result(
        self,
        queries: List[str],
        top_k: int,
        where: Optional[Where] = None,
        query_mode: QueryMode = QueryMode.ORIGINAL,
        include: Optional[List[str]] = None,
        return_history: bool = False,
        consolidation: bool = False,
    ):
        memory_results = []
        extracted = set()  # Track already included memories to prevent duplicates

        factual_memory_condition = {"memory_type": {"$eq": "factual"}}
        
        if query_mode == QueryMode.CUE_ONLY:
            # Only search cue index memories
            # query_condition = {"linked_memory": {"$ne": ""}}
            # where = {"$and": [query_condition, factual_memory_condition]}
            # cue_indices do not have memory_type = "factual"
            where = {"linked_memory": {"$ne": ""}}
        elif query_mode == QueryMode.PRIMARY_ONLY:
            # Only search primary memories
            query_condition = {"linked_memory": {"$eq": ""}}
            where = {"$and": [query_condition, factual_memory_condition]}

        # Execute search for each generated query variant
        for query in queries:
            # Perform vector similarity search using the underlying storage backend
            results: List[MemoryEntry] = self._store.query(query, top_k, where, include, consolidation=consolidation)

            # Convert distances to similarity scores (1 - distance = similarity)
            scores = [entry.score for entry in results]

            # Combine results with scores and sort by relevance
            merged = list(zip(results, scores))
            merged_sorted = sorted(merged, key=lambda x: x[1], reverse=True)

            # Filter results by similarity threshold and deduplication
            for entry, score in merged_sorted:

                # stop processing if score below threshold
                if score < self.QUERY_SCORE_THRESHOLD:
                    break

                if entry.is_cue_index():
                    # get the linked memory entry
                    for primary_index in entry.get_linked_memories():
                        primary_entry = self._store.get(primary_index)

                        if not primary_entry:
                            logger.warning(f"Primary memory entry cannot found: {primary_index}")
                            continue

                        # update the entry and value to the linked memory
                        value = primary_entry.get_memory_value(
                            return_history=return_history
                        )
                        if value in extracted:
                            continue

                        memory_results.append(primary_entry) # append the primary entry itself, not the cue entry
                        extracted.add(value)
                else:
                    # primary memory entry
                    value = entry.get_memory_value(return_history=return_history)

                    if value in extracted:
                        continue

                    memory_results.append(entry)
                    extracted.add(value)

        return memory_results[:top_k]

    def _perform_hybrid_search(
        self,
        context: str,
        where: Optional[Where] = None,
    ) -> List[MemoryEntry]:
        """
        Perform hybrid search (keyword or BM25) based on configuration.
        
        Args:
            context: Search query string
            where: Filter conditions for metadata-based filtering
            
        Returns:
            List of MemoryEntry objects from hybrid search
        """
        # Get hybrid search method from config (default to 'bm25')
        hybrid_method = self.cfg.memory.get("hybrid_search_method", "bm25")
        hybrid_top_k = self.cfg.memory.get("hybrid_top_k", 10)

        # Build BM25 index if using BM25 method
        if hybrid_method == "bm25":
            # Extract user_id from where clause for index building
            target_user_id = extract_user_id_from_where(where) or self.user_id
            
            # Build index if it doesn't exist for this user
            if target_user_id and target_user_id not in self._store._bm25_indices:
                logger.info(f"Building BM25 index for user {target_user_id} before first search")
                self._store.build_bm25_index(user_id=target_user_id)

        # Perform hybrid search based on method
        hybrid_results = []

        if hybrid_method == "bm25":
            bm25_threshold = self.cfg.memory.get("bm25_score_threshold", 0.4)
            hybrid_results = self._store.bm25_search(context, hybrid_top_k, where, bm25_threshold)

        elif hybrid_method == "keyword":
            keywords = self.query_generator.extract_keywords(context)
            if keywords:
                hybrid_results = self._store.keyword_search(keywords, hybrid_top_k, where)

        else:
            raise ValueError(f"Unsupported hybrid search method: {hybrid_method}")
        
        return hybrid_results

    def _merge_results_with_rrf(
        self,
        result_lists: List[List[MemoryEntry]],
        weights: Optional[List[float]] = None,
        k: int = 60,
    ) -> List[MemoryEntry]:
        """
        Merge multiple search result lists using weighted Reciprocal Rank Fusion (RRF).
        
        RRF combines ranked lists without relying on raw scores, using position-based scoring:
        RRF_score(d) = sum(weight_i * (1 / (k + rank_i(d)))) for all lists where d appears
        
        Args:
            result_lists: List of result lists from different search sources
            weights: List of weights for each result list (default: [2.0, 1.0, 1.0] for [primary, cue, hybrid])
            k: RRF constant (default 60) to prevent division by zero and reduce top-rank impact
            
        Returns:
            Combined and re-ranked list of MemoryEntry objects with weighted RRF scores
        """
        # Default weights: primary semantic (2.0), cue index (1.0), hybrid (1.0)
        if weights is None:
            weights = [2.0] * len(result_lists)
        
        if len(weights) != len(result_lists):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of result lists ({len(result_lists)})")
        
        rrf_scores = {}
        all_entries = {}
        
        # Calculate weighted RRF scores for each result list
        for result_list, weight in zip(result_lists, weights):
            for rank, entry in enumerate(result_list, start=1):
                record_id = index_to_id(entry.index)
                
                # Store the entry (only need one copy)
                if record_id not in all_entries:
                    all_entries[record_id] = entry
                
                # Add weighted RRF contribution from this list
                if record_id in rrf_scores:
                    rrf_scores[record_id] += weight * (1.0 / (k + rank))
                else:
                    rrf_scores[record_id] = weight * (1.0 / (k + rank))
        
        # Normalize RRF scores to 0-1 range
        if rrf_scores:
            max_score = max(rrf_scores.values())
            min_score = min(rrf_scores.values())
            score_range = max_score - min_score
            
            # Normalize to 0-1 range (handle edge case where all scores are the same)
            if score_range > 0:
                rrf_scores = {
                    record_id: (score - min_score) / score_range
                    for record_id, score in rrf_scores.items()
                }
            else:
                # All scores are the same, set them all to 1.0
                rrf_scores = {record_id: 1.0 for record_id in rrf_scores}
        
        # Update scores with normalized RRF scores
        for record_id, rrf_score in rrf_scores.items():
            if record_id in all_entries:
                all_entries[record_id].score = rrf_score
        
        # Sort by RRF score and return
        return sorted(all_entries.values(), key=lambda x: x.score, reverse=True)

    def query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        top_k: int = 5,
        where: Optional[Where] = None,
        query_mode: QueryMode = QueryMode.ORIGINAL,
        include: Optional[List[str]] = None,
        enhance_query: bool = True,
        return_history: bool = False,
        enable_hybrid_search: bool = False,
        enable_llm_filter: bool = False,
        latency_tracker = None,
        consolidation: bool = False,
    ):
        """
        Perform intelligent semantic search to find relevant memories.

        This method implements sophisticated memory retrieval using:
        - Vector similarity search for semantic matching
        - Optional keyword-based search for exact term matching (hybrid search)
        - Optional LLM-powered query enhancement for better results
        - Similarity threshold filtering to ensure relevance
        - Deduplication to avoid returning the same memory multiple times
        - Metadata-based filtering for user/context-specific results

        The query process:
        1. Convert input context to standardized string format
        2. Optionally enhance query using LLM to generate multiple search variants
        3. If hybrid search enabled, extract keywords for keyword-based search
        4. Execute vector search and/or keyword search against the memory store
        5. Merge and deduplicate results from both search methods
        6. Filter results by similarity threshold
        7. Return ranked, relevant memories with metadata

        Args:
            context: Search context in flexible formats:
                - str: Natural language query text
                - List[str]: Multiple query strings to search
                - List[Dict[str, str]]: Structured context with key-value pairs
            top_k: Maximum number of results to return per query
            where: ChromaDB filter conditions for metadata-based filtering
                Example: {"user_id": "user-1", "timestamp": {"$gt": "2023-01-01"}}
            include: Fields to include in results ["metadatas", "distances", "documents"]
            filtering: Whether to apply similarity threshold filtering
            enhance_query: Whether to use LLM to generate enhanced query variants
            return_history: Whether to include memory update history in results
            enable_hybrid_search: Whether to enable hybrid search combining semantic and keyword search
            enable_llm_filter: Whether to use LLM to filter irrelevant memories
            query_mode: decide if we want to search with cue index (ORIGINAL, PRIMARY_ONLY, CUE_ONLY, BOTH)
            latency_tracker: Optional LatencyTracker for performance measurement
            consolidation: vector-search the consolidation collection instead of the retrieval
                collection (used by the consolidation candidate search)

        Returns:
            List[Dict]: Ranked list of relevant memories, each containing:
                - memory: The memory content/value
                - metadata: Associated metadata (query, index, timestamp, etc.)
                - score: Similarity score (0-1, higher = more similar)

        Note:
            Uses deduplication to prevent the same memory from appearing multiple times
            when enhanced queries overlap in their results or when combining semantic
            and keyword search results.
        """

        # Normalize input context to consistent string format
        context = context_to_str(context)

        # initialize results list
        memory_results = []

        if enhance_query:
            # Use LLM to generate multiple query variants for comprehensive search
            # This improves recall by capturing different ways to express the same intent
            queries = self.query_generator.generate_queries(context)
        else:
            # Use the original context as a single query for faster, direct search
            queries = [context]

        # Collect results from different sources for RRF merging
        primary_results = []
        cue_results = []
        hybrid_results = []

        # Query based on mode
        if query_mode == QueryMode.ORIGINAL:
            if latency_tracker:
                with latency_tracker.track("search_primary"):
                    primary_results = self._query_result(
                        queries, top_k, where, QueryMode.ORIGINAL, include, return_history, consolidation=consolidation
                    )
            else:
                primary_results = self._query_result(
                    queries, top_k, where, QueryMode.ORIGINAL, include, return_history, consolidation=consolidation
                )
            memory_results = primary_results
        elif query_mode == QueryMode.PRIMARY_ONLY:
            if latency_tracker:
                with latency_tracker.track("search_primary"):
                    primary_results = self._query_result(
                        queries, top_k, where, QueryMode.PRIMARY_ONLY, include, return_history, consolidation=consolidation
                    )
            else:
                primary_results = self._query_result(
                    queries, top_k, where, QueryMode.PRIMARY_ONLY, include, return_history, consolidation=consolidation
                )
            memory_results = primary_results
        elif query_mode == QueryMode.CUE_ONLY:
            if latency_tracker:
                with latency_tracker.track("search_cue"):
                    cue_results = self._query_result(
                        queries, self.cfg.memory.cue_top_k, where, QueryMode.CUE_ONLY, include, return_history, consolidation=consolidation
                    )
            else:
                cue_results = self._query_result(
                    queries, self.cfg.memory.cue_top_k, where, QueryMode.CUE_ONLY, include, return_history, consolidation=consolidation
                )
            memory_results = cue_results
        elif query_mode == QueryMode.BOTH:
            # Get results from both primary and cue index searches separately
            if latency_tracker:
                with latency_tracker.track("search_primary"):
                    primary_results = self._query_result(
                        queries, top_k, where, QueryMode.PRIMARY_ONLY, include, return_history, consolidation=consolidation
                    )
                with latency_tracker.track("search_cue"):
                    cue_results = self._query_result(
                        queries,
                        self.cfg.memory.cue_top_k,
                        where,
                        QueryMode.CUE_ONLY,
                        include,
                        return_history,
                        consolidation=consolidation,
                    )
            else:
                primary_results = self._query_result(
                    queries, top_k, where, QueryMode.PRIMARY_ONLY, include, return_history, consolidation=consolidation
                )
                cue_results = self._query_result(
                    queries,
                    self.cfg.memory.cue_top_k,
                    where,
                    QueryMode.CUE_ONLY,
                    include,
                    return_history,
                    consolidation=consolidation,
                )

        # If hybrid search is enabled, perform additional search and merge all sources with RRF
        if enable_hybrid_search:
            try:
                # Perform hybrid search (keyword or BM25)
                if latency_tracker:
                    with latency_tracker.track("search_hybrid"):
                        hybrid_results = self._perform_hybrid_search(context, where)
                else:
                    hybrid_results = self._perform_hybrid_search(context, where)

                # Prepare result lists and weights for RRF merging
                result_lists = []
                weights = []

                # Add primary results if available
                if primary_results:
                    result_lists.append(primary_results)
                    weights.append(2.0)  # Higher weight for primary semantic search

                # Add cue index results if available
                if cue_results:
                    result_lists.append(cue_results)
                    weights.append(1.0)  # Medium weight for cue index search

                # Add hybrid results if available
                if hybrid_results:
                    result_lists.append(hybrid_results)
                    weights.append(1.0)  # Medium weight for hybrid search

                # Merge all sources using RRF
                if len(result_lists) > 1:
                    if latency_tracker:
                        with latency_tracker.track("search_rrf_merge"):
                            memory_results = self._merge_results_with_rrf(result_lists, weights)
                    else:
                        memory_results = self._merge_results_with_rrf(result_lists, weights)
                elif len(result_lists) == 1:
                    memory_results = result_lists[0]
                else:
                    memory_results = []

            except Exception as e:
                # If hybrid search fails, log and continue with semantic results only
                logger.warning(f"Hybrid search failed: {e}. Falling back to semantic search only.")
                # Fallback: merge primary and cue results if both exist
                if query_mode == QueryMode.BOTH and primary_results and cue_results:
                    if latency_tracker:
                        with latency_tracker.track("search_rrf_merge"):
                            memory_results = self._merge_results_with_rrf(
                                [primary_results, cue_results],
                                [2.0, 1.0]
                            )
                    else:
                        memory_results = self._merge_results_with_rrf(
                            [primary_results, cue_results],
                            [2.0, 1.0]
                        )
                elif primary_results:
                    memory_results = primary_results
                elif cue_results:
                    memory_results = cue_results
        else:
            # No hybrid search - just merge primary and cue if in BOTH mode
            if query_mode == QueryMode.BOTH and primary_results and cue_results:
                if latency_tracker:
                    with latency_tracker.track("search_rrf_merge"):
                        memory_results = self._merge_results_with_rrf(
                            [primary_results, cue_results],
                            [2.0, 1.0]
                        )
                else:
                    memory_results = self._merge_results_with_rrf(
                        [primary_results, cue_results],
                        [2.0, 1.0]
                    )

        # Apply LLM-based filtering if explicitly enabled
        if enable_llm_filter and memory_results:
            if latency_tracker:
                with latency_tracker.track("search_llm_filter"):
                    memory_results = self.memory_filter.filter_memory(
                        query=context,
                        memory_results=memory_results,
                    )
            else:
                memory_results = self.memory_filter.filter_memory(
                    query=context,
                    memory_results=memory_results,
                )
            return memory_results

        return memory_results[:top_k]

    def get_episodic_memories_for_results(
        self, memory_results: List[MemoryEntry]
    ) -> Dict[str, MemoryEntry]:
        """
        Retrieve episodic memories linked to the given factual memory results.
        
        This method:
        1. Extracts all episodic_memory_ids from the factual memories
        2. Deduplicates them (since multiple factual memories may share the same episode)
        3. Fetches the actual episodic memory entries
        
        Args:
            memory_results: List of factual memory entries from query results
            
        Returns:
            Dict mapping episodic_memory_id to MemoryEntry for all linked episodes
        """
        # Collect all unique episodic memory IDs
        episodic_ids = set()
        for entry in memory_results:
            if entry.episodic_memory_ids:
                episodic_ids.update(entry.episodic_memory_ids)
        
        # Fetch episodic memories
        episodic_memories = {}
        for episodic_id in episodic_ids:
            episodic_entry = self._store.get(episodic_id)
            if episodic_entry:
                episodic_memories[episodic_id] = episodic_entry
            else:
                logger.warning(f"Episodic memory not found: {episodic_id}")
        
        return episodic_memories

    def get(self, key: str) -> MemoryEntry:
        """
        Retrieve a single record by its natural-language key.

        Args:
            key: Natural language key to retrieve

        Returns:
            Dict with id, metadata, document fields or None if not found
        """
        return self._store.get(key)

    def add(self, entry: MemoryEntry):
        """
        Add a single memory entry to the store.

        Args:
            entry: MemoryEntry object to add

        Returns:
            Record ID of the added memory entry
        """
        # you cannot add a cue index manually
        assert (
            entry.is_primary_index()
        ), "Only primary memory entries can be added directly."

        exist_entry = self._store.get(entry.index)
        
        # Handle duplicate indices based on memory type
        if exist_entry is not None:
            if entry.memory_type == "episodic":
                # For episodic memories, append sequential number to make index unique
                # Each episode is distinct even if the summary is similar
                original_index = entry.index
                counter = 2
                while self._store.get(f"{original_index} ({counter})") is not None:
                    counter += 1
                entry.index = f"{original_index} ({counter})"
                logger.info(
                    f"Episodic memory index already exists. "
                    f"Renamed '{original_index}' to '{entry.index}'"
                )
            else:
                # For factual memories, this is an error
                raise AssertionError(f"Memory entry {entry.index} already exists.")

        log_memory_operation("Add", entry, user_id=self.user_id)

        # add primary memory
        self._store.upsert(
            index=entry.index, value=entry.value, metadata=entry.get_metadata()
        )

        # Add cue index entries
        for cue_index in entry.get_cue_indices():

            # skip the cue index in two cases:
            # 1. the cue index is already a primary index
            # 2. the cue index is the same as the current primary index
            cue_entry = self._store.get(cue_index)
            if (cue_entry and cue_entry.is_primary_index()) or cue_index == entry.index:
                # remove the cue index since it's already a primary index
                entry.delete_cue_index(cue_index)
                continue

            # check if it exists
            linked_memory = entry.index
            if cue_entry and cue_entry.is_cue_index():
                # combine linked memory
                linked_memory = combine_list(linked_memory, cue_entry.linked_memory)

            # add or update cue index entry
            self._store.upsert(
                index=cue_index,
                value="",
                metadata={
                    "linked_memory": linked_memory,
                },
            )

    def _delete_cue_index(self, entry: MemoryEntry) -> None:
        """
        Delete a cue index and update linked primary memories.

        Args:
            key: Cue index key to delete
        """
        # get all the primary memories linked to this cue index
        linked_memories = entry.get_linked_memories()
        for primary_index in linked_memories:
            primary_entry = self._store.get(primary_index)
            assert (
                primary_entry is not None
            ), f"Primary entry {primary_index} not found."

            # delete the cue index in the primary memory
            primary_entry.cue_indices = "||".join(
                [ci for ci in primary_entry.get_cue_indices() if ci != entry.index]
            )
            self._store.upsert(
                index=primary_entry.index,
                value=primary_entry.value,
                metadata=primary_entry.get_metadata(),
            )
        # finally delete the memory entry itself
        self._store.delete(entry.index)

    def _delete_primary_memory(self, entry: MemoryEntry) -> None:
        """
        Delete a primary memory and all associated cue indices.

        Args:
            key: Primary memory key to delete
        """
        # delete all cue indices linked to this primary memory
        cue_indices = entry.get_cue_indices()
        for cue_index in cue_indices:
            cue_entry = self._store.get(cue_index)

            # Handle edge case: cue entry doesn't exist
            if cue_entry is None:
                raise AssertionError(
                    f"Cue entry '{cue_index}' not found. This may indicate a data consistency issue."
                )

            # Check if it's a valid cue index
            if cue_entry.is_cue_index():
                # Process normally - delete the cue index
                pass
            elif cue_entry.is_primary_index():
                # Handle expected edge case: cue index has been converted to a primary index
                # This can happen when a cue index phrase is later used as a primary index
                # for another memory (similar to the check in add() method)
                logger.info(
                    f"Skipping cue index '{cue_index}' during deletion: "
                    f"converted to primary index"
                )
                continue
            else:
                # Unexpected case: entry is neither cue nor primary index
                raise AssertionError(
                    f"Cue entry '{cue_index}' is in an invalid state: "
                    f"not a cue index (linked_memory='{cue_entry.linked_memory}') "
                    f"and not a primary index. This requires investigation."
                )

            # delete the cue index from the primary index
            linked_memories = cue_entry.get_linked_memories()
            linked_memories = [lm for lm in linked_memories if lm != entry.index]
            if linked_memories:
                # update the linked memories
                self._store.upsert(
                    index=cue_index,
                    value="",
                    metadata={
                        "linked_memory": "||".join(linked_memories),
                    },
                )
            else:
                # delete the cue index if no linked memories left
                self._store.delete(cue_index)
        # finally delete the memory entry itself
        self._store.delete(entry.index)

    def delete(self, key: str) -> None:
        """
        Delete a record by its natural-language key.

        Args:
            key: Natural language key to delete
        """
        entry = self._store.get(key)

        if entry.is_cue_index():
            # delete the cue index
            self._delete_cue_index(entry)
        elif entry.is_primary_index():
            # delete the primary memory and all associated cue indices
            self._delete_primary_memory(entry)

    def list_memories(self, limit: int = 20) -> List[MemoryEntry]:
        """
        List memories in the collection.

        Args:
            limit: Max number of records to return

        Returns:
            Dict containing memory records
        """
        return self._store.list_memories(limit)

    def count(self) -> int:
        """
        Get the number of records in the collection.

        Returns:
            Number of records in the collection
        """
        return self._store.count()

    def get_backend_type(self) -> str:
        """
        Get the current backend type being used.

        Returns:
            Backend type ('local' or 'http')
        """
        return self.backend_type

    def clear(self) -> None:
        """
        Clear all records in the collection.
        """
        self._store.clear()
