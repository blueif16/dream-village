# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Memory Entry Data Structure

This module defines the MemoryEntry class which serves as a standardized
return object for memory query and build operations in the Memora system.
"""

import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator


class MemoryMetadata(BaseModel):
    """
    Metadata associated with a memory entry.
    
    This class encapsulates all metadata fields that can be associated with
    a memory, including search context, indexing information, timestamps,
    and linkage data.
    """
    pass


class MemoryEntry(BaseModel):
    """
    Standardized memory entry object for query and build operations.
    
    This class provides a unified interface for memory entries returned from
    both query operations (semantic search) and build operations (memory extraction).
    It encapsulates the memory content, associated metadata, relevance scoring,
    and temporal information.
    
    Usage:
        # From query results
        entry = MemoryEntry.from_query_result(query_dict)
        
        # From build results  
        entry = MemoryEntry.from_build_result(build_dict)
        
        # Direct creation
        entry = MemoryEntry(
            value="User prefers dark theme",
            index="user_interface_preference",
            score=0.85
        )
    """

    # Core memory content
    value: str = Field(
        description="The main memory content/text"
    )

    original_text: Optional[str] = Field(
        default="",
        description="The original unprocessed content (for episodic memories)"
    )

    index: Optional[str] = Field(
        default="",
        description="Memory index/key for retrieval and identification"
    )

    # History and versioning support
    history: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Historical versions of this memory entry"
    )

    # type of the memory
    memory_type: Optional[str] = Field(
        default="",
        description="Type of memory (e.g., 'factual', 'procedural', 'episodic')"
    )

    # Link to episodic memories (for factual memories)
    episodic_memory_ids: Optional[List[str]] = Field(
        default_factory=list,
        description="List of episodic memory IDs that provide context for this factual memory"
    )

    # Relevance and scoring information
    score: Optional[float] = Field(
        default=0.0,
        description="Relevance/similarity score"
    )

    # Temporal information
    timestamp: Optional[str] = Field(
        default="",
        description="Timestamp when the memory event occurred"
    )

    # original query that led to this memory
    query: Optional[str] = Field(
        default="",
        description="The original query that retrieved this memory"
    )

    # creation date time
    creation_time: Optional[str] = Field(
        default="",
        description="Timestamp when the memory was originally created"
    )

    # linked memory index if the memory entry is only for indexing purpose
    linked_memory: Optional[str] = Field(
        default="",
        description="Reference to linked memory entries"
    )

    # cue indices if the memory entry is linked to multiple indices
    cue_indices: Optional[str] = Field(
        default="", description="Indices linked to this memory for enhanced retrieval"
    )

    # image URLs associated with this memory
    image_urls: Optional[List[str]] = Field(
        default_factory=list,
        description="List of image URLs associated with this memory"
    )

    # source: Optional[str] = Field(
    #     default="",
    #     description="Source of the memory (e.g., document name, conversation ID)"
    # )

    # entities: Optional[List[str]] = Field(
    #     default_factory=list,
    #     description="List of entities (people, places, organizations) mentioned in the memory"
    # )

    # tags: Optional[List[str]] = Field(
    #     default_factory=list,
    #     description="Tags associated with the memory for organization"
    # )
    
    def is_cue_index(self) -> bool:
        """Check if the memory entry is a cue index."""
        return self.linked_memory != ""

    def is_primary_index(self) -> bool:
        """Check if the memory entry is a primary index."""
        return not self.linked_memory

    def get_cue_indices(self) -> List[str]:
        """Get the list of cue indices."""
        if not self.cue_indices:
            return []
        return [cue.strip() for cue in self.cue_indices.split("||") if cue.strip()]

    def delete_cue_index(self, cue_index: str):
        """Delete a cue index from the memory entry."""
        cue_indices = self.get_cue_indices()
        cue_indices = [ci for ci in cue_indices if ci != cue_index]
        self.cue_indices = "||".join(cue_indices)

    def get_linked_memories(self) -> List[str]:
        """Get the list of linked memories."""
        if not self.linked_memory:
            return []
        return [lm.strip() for lm in self.linked_memory.split("||") if lm.strip()]

    def get_memory_value(self, return_history: bool=False, use_original_text: bool=False) -> str:
        """Get the memory value.

        Args:
            return_history: If True, return concatenated history
            use_original_text: If True and original_text exists, return original_text instead of processed value
        """

        # Return original text if requested and available
        if use_original_text and self.original_text:
            return self.original_text

        # load history if existing
        if not self.history or not return_history:
            return self.value

        # concatenate history entries
        value = "\n".join(
            [f"[{h['timestamp']}] {h['value']}" for h in self.history]
        )
        return value

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """
        Create a MemoryEntry from a dictionary.
        
        Args:
            data: Dictionary containing memory entry data
            
        Returns:
            MemoryEntry: Standardized memory entry object
        """
        data["history"] = json.loads(data.get("history", "[]"))
        data["image_urls"] = json.loads(data.get("image_urls", "[]"))
        data["episodic_memory_ids"] = json.loads(data.get("episodic_memory_ids", "[]"))
        return cls(**data)

    def get_metadata(self) -> Dict[str, Any]:
        """
        Convert MemoryEntry to dictionary format.
        
        Returns:
            Dict: Dictionary representation of the memory entry
        """
        metadata = {
            "index": self.index,
            "history": json.dumps(self.history),
            "timestamp": self.timestamp,
            "query": self.query,
            "creation_time": self.creation_time,
            "linked_memory": self.linked_memory,
            "cue_indices": self.cue_indices,
            "image_urls": json.dumps(self.image_urls),
            "memory_type": self.memory_type,
            "episodic_memory_ids": json.dumps(self.episodic_memory_ids),
            "original_text": self.original_text,
        }
        return metadata

    def __str__(self) -> str:
        """String representation of the memory entry."""
        parts = []
        if self.index:
            parts.append(f"[{self.index}]")
        parts.append(self.value)
        if self.score > 0:
            parts.append(f"(score: {self.score:.3f})")
        return " ".join(parts)

    def __repr__(self) -> str:
        """Detailed representation of the memory entry."""
        return f"MemoryEntry(index='{self.index}', value='{self.value[:50]}...', score={self.score})"
