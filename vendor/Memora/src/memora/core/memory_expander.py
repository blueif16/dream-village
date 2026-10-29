# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Memory Expander

This module provides functionality to expand retrieved memories by building
a frontier of linked and related memories that can be explored.
"""

from typing import Dict, List, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from memora.core.memory_entry import MemoryEntry
from memora.core.memory import AgentMemory, QueryMode

class MemoryExpander:
    """
    Expands retrieved memories by building a frontier of linked memories.
    
    The Expander class analyzes a working set of memories and identifies
    potential expansion candidates (frontier) by following memory links.
    This enables iterative retrieval where the system can progressively
    explore related memories.
    
    Features:
    - Build frontier from memory links
    - Avoid duplicates with working set
    - Track visited memories
    
    Example:
        expander = Expander()
        frontier = expander.build_frontier(retrieved_memories)
        for memory_id, memory in frontier.items():
            print(f"Frontier candidate: {memory.index}")
    """
    
    def __init__(self, memory_client: Optional[AgentMemory] = None, 
                 enable_relaxed_frontier: bool = False,
                 relaxed_frontier_top_k: int = 4,
                 relaxed_frontier_threshold: float = 0.85,
                 max_cues_to_expand: int = 30,
                 max_workers: int = 5):
        """Initialize the Expander.
        
        Args:
            memory_client: AgentMemory instance
            enable_relaxed_frontier: Whether to expand frontier via similar cue indices
            relaxed_frontier_top_k: Top-k similar cues to retrieve (stricter than normal)
            relaxed_frontier_threshold: Minimum similarity score for relaxed expansion
            max_cues_to_expand: Maximum number of cues to expand (picks from highest-scoring memories)
            max_workers: Number of parallel workers for cue expansion
        """
        self.visited_ids: Set[str] = set()
        self.memory_client = memory_client
        self.enable_relaxed_frontier = enable_relaxed_frontier
        self.relaxed_frontier_top_k = relaxed_frontier_top_k
        self.relaxed_frontier_threshold = relaxed_frontier_threshold
        self.max_cues_to_expand = max_cues_to_expand
        self.max_workers = max_workers

    
    def set_memory_client(self, memory_client: AgentMemory):
        """
        Set the memory client for fetching linked memories.
        Args:
            memory_client: AgentMemory instance
        """
        self.memory_client = memory_client
    
    def build_frontier(
            self,
            frontier: Dict[str, MemoryEntry],
            memories: List[MemoryEntry]
    ) -> Dict[str, MemoryEntry]:
        """
        Build a frontier dictionary from the list of memories.
        
        The frontier consists of memories that are linked to the current
        working set but not yet retrieved. This allows for efficient
        expansion of the memory search space.
        
        Args:
            frontier: Existing frontier dict to expand (can be empty {})
            memories: List of MemoryEntry objects in the working set
        
        Returns:
            Updated frontier dictionary mapping memory IDs to MemoryEntry objects
        """
        
        if self.memory_client is None:
            raise ValueError("memory_client must be set before calling build_frontier()")
        
        print("=="*40)
        print('\n')
        print("Building frontier")
        print("=="*40)

        # Build working set: all memories we already have
        working_set = {memory.index for memory in memories}

        # Step 1: Collect all direct cues from memories (excluding visited)
        # Track which memories contributed which cues for scoring
        cue_to_memory_score = {}  # cue -> best memory score
        direct_cues = set()
        
        for memory in memories:
            if memory.index in self.visited_ids:
                continue
            self.visited_ids.add(memory.index)
            
            # Get memory score (default to 1.0 if not available, e.g., during EXPAND)
            memory_score = memory.score if memory.score is not None else 1.0
            
            for cue_index in memory.get_cue_indices():
                if cue_index not in self.visited_ids:
                    direct_cues.add(cue_index)
                    # Track the highest score for each cue
                    if cue_index not in cue_to_memory_score or memory_score > cue_to_memory_score[cue_index]:
                        cue_to_memory_score[cue_index] = memory_score
        
        # Step 2: Find similar cues if relaxed frontier is enabled (only for top cues)
        all_cues = set(direct_cues)
        
        if self.enable_relaxed_frontier and direct_cues:
            # Sort cues by their memory scores and only expand the top ones
            sorted_cues = sorted(cue_to_memory_score.items(), key=lambda x: x[1], reverse=True)
            cues_to_expand = [cue for cue, score in sorted_cues[:self.max_cues_to_expand]]
                        
            # Parallelize similarity searches
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all searches
                future_to_cue = {
                    executor.submit(self._find_similar_cues, cue_index): cue_index 
                    for cue_index in cues_to_expand
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_cue):
                    similar_cues = future.result()
                    for similar_cue in similar_cues:
                        if similar_cue.index not in self.visited_ids:
                            all_cues.add(similar_cue.index)
                
        # Step 3: Process each unique cue once
        for cue_id in all_cues:
            if cue_id in self.visited_ids:
                continue
            self.visited_ids.add(cue_id)
            
            # Get the cue entry and its linked primary memories
            cue_entry = self.memory_client.get(cue_id)
            if not cue_entry:
                continue
            
            # Add each linked primary memory to frontier if novel
            for linked_index in cue_entry.get_linked_memories():
                # Skip if already in working set or frontier
                if linked_index in working_set or linked_index in frontier:
                    continue
                
                # Fetch and add the primary memory
                linked_entry = self.memory_client.get(linked_index)
                if linked_entry:
                    frontier[linked_entry.index] = linked_entry
                                  
        return frontier
    
    def _find_similar_cues(self, cue_index: str) -> List[MemoryEntry]:
        """
        Find similar cue indices using semantic search.
        Uses stricter top_k and threshold to avoid going too broad.
        
        Args:
            cue_index: The cue index text (e.g., "Beaches near Barcelona")
            
        Returns:
            List of similar MemoryEntry objects (cues only)
        """
        try:
            similar_cues = self.memory_client.query(
                cue_index,  # Search with the cue text directly
                top_k=self.relaxed_frontier_top_k,
                enable_hybrid_search=False,
                query_mode=QueryMode.CUE_ONLY  # Only search cue indices
            )
            
            # Filter by threshold and exclude the original cue
            filtered = [
                cue for cue in similar_cues
                if cue.score >= self.relaxed_frontier_threshold 
                and cue.index != cue_index
            ]
            
            return filtered
            
        except Exception as e:
            print(f"Error finding similar cues for '{cue_index}': {e}")
            return []
    
    def reset(self):
        """Reset the visited IDs tracker."""
        self.visited_ids.clear()
