# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from memora.core.memory_entry import MemoryEntry
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig
from pydantic import BaseModel, Field
from memora.core.cue_index_generator import CueIndexGenerator
from memora.core.memory_entry import MemoryEntry
from memora.utils.llm import ChatCompletionModel
from memora.utils.log import log_memory_building
from memora.utils.memory import (
    MemoryUpdateDecision,
    generate_history,
    generate_metadata,
)
from memora.utils.misc import get_current_timestamp
from typing import Any, Dict, List, Optional, Union
from omegaconf import DictConfig
from pydantic import BaseModel, Field
from memora.core.cue_index_generator import CueIndexGenerator
from memora.core.memory import AgentMemory, QueryMode
from memora.utils.memory import (
    MemoryUpdateDecision,
    generate_history,
)
from memora.utils.misc import get_current_timestamp, normalize_content
import logging
from openai import BadRequestError

# Initialize module logger
logger = logging.getLogger(__name__)

PROMPT_BUILD_MEMORY = """
You are an expert fatual memory extraction assistant. Your goal is to extract factual memories from a conversation segment.

# TASK: 
Read the input conversation carefully, extract ALL factual memories that could be useful for future reference.

Produce each memory as a key-value pair in the following format:

MemIndex: memory index for retrieval
MemValue: memory value with all the details supported directly from the given text.

# GUIDELINES:
1. Content and Scope:
- Use only information explicitly mentioned in the context to create the factual memories.
- Make sure to capture ALL factual information that could be useful for future retrieval. When in doubt, create more factual memories rather than fewer. Capture more details rather than less.
- Do not include greetings, small talk, or filler in the memories.
- Be exhaustive when extracting factual memories from the conversation, do not miss any details, unless they are explicitly irrelevant and unlikely to be useful for future reference (like greetings or small talk).
- Split distinct facts into separate entries.
- Capture all details about people's identities, experiences, past or upcoming events, intentions, hobbies, preferences, states, beliefs, goals, or future plans that may be useful for answering later questions.
- Make sure to include time of events, location, or other contextual details in the MemValue if mentioned.
- If images are included in the conversation, consider the images as a part of the text. DO NOT create a memory entry solely to describe the image. Instead, extract useful facts from the image (e.g., objects, locations) and merge them naturally into relevant memories. For example, if a conversation contains an image showing a birthday cake with candles, include information about that image in the relevant memory about the birthday event, such as "MemIndex: Alice celebrated her birthday\nMemValue: Alice celebrated her birthday with her friends at home, including a birthday cake with candles".
- If the conversation is between a user and an AI assistant, focus on the user's inputs and the overall context rather than the assistant's responses.

2. Format and Style:
- The MemIndex must be a short, human-readable phrase that is self-contained and unambiguous.
- Always include the specific context (e.g., domain, or entity) from the source text in the MemIndex to avoid vague terms. For example, instead of "Vacation", use "Alice's Japan Vacation". Instead of "Mike's plans", use "Mike's summer plans to visit Europe".
- Write MemValue as one or two full factual sentences, capturing all relevant details. 
    - Ensure wording is neutral and factual.
    - Use the original wordings from the conversation when possible.
    - Replace pronouns with specific names or entities to ensure clarity.
    - Handling times and dates in the MemValue: When dates and times are mentioned in the conversation, replace relative times (e.g., "yesterday", "next week", "last year") with absolute dates based on the timestamp of the conversation. For example, if the conversation timestamp is "16 June, 2023" and it mentions something happened "last year", convert it to "2022" in the MemValue.

Timestamp of conversation: {timestamp}

Input Conversation:
{content}

Output:
"""

# LLM prompt template for memory update decisions
PROMPT_MEMORY_UPDATE_DECISION = """
You are a memory management assistant. Given a new memory entry and similar existing entries, determine whether to update an existing entry or add a new one.

NEW MEMORY ENTRY:
Index: {new_index}
Value: {new_value}

EXISTING SIMILAR ENTRIES:
{candidates_info}

INSTRUCTIONS:
1. Analyze if the new entry should update any existing entry based on semantic similarity and content overlap
2. If update is needed, determine which candidate is best to update
3. Generate the updated value that combines relevant information from both entries
4. Decide if the memory index should be updated to better reflect the combined information
"""

# Template for formatting candidate information in LLM prompts
PROMPT_CANDIDATE_FORMAT = """
Candidate {index}:
- Similarity Score: {score:.3f}
- Index: {index_text}
- Value: {value}
- Creation Time: {creation_time}
"""

class MemoryOutput(BaseModel):
    memory_type: str = Field(description="Type of memory: 'Factual' or 'Procedural'.")
    index: str = Field(
        description="Use a short, specific phrase that captures the fact clearly, with enough detail to support retrieval."
    )
    value: str = Field(
        description="A concise but complete factual statement, supported directly by the conversation."
    )

    def __str__(self) -> str:
        return f"[*{self.index}*] {self.value}"

    def __repr__(self) -> str:
        return str(self)


class MemoryOutputs(BaseModel):
    entries: list[MemoryOutput] = Field(
        description="memories extracted from the content that are factual and verifiable.",
    )


class MemoryBuilder(ABC):
    """All builders accept either text or file_path (exactly one required)."""

    def __init__(self, cfg: DictConfig, agent_memory: AgentMemory, model_client: ChatCompletionModel):
        self.cfg = cfg
        self._model_client = model_client  # initialize your LLM model here

        # Initialize AgentMemory for memory operations
        self.agent_memory = agent_memory

        # Read multimodal support setting from config, default to True
        self.multimodal_support = cfg.memory.get("multimodal_support", True)

        self.cue_index_generator = CueIndexGenerator(
            cfg=self.cfg, model_client=self._model_client
        )

        self.UPDATE_SCORE_THRESHOLD = (
            cfg.memory.update_score_threshold
        )  # Consider for updates

        # Build stats tracking (reset per build() call)
        self._build_stats = {}

    def build(
        self,
        content: Union[str, List[str], List[Dict[str, str]]],
        metadata: Optional[Dict] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None, # Passing per request
    ) -> List[MemoryEntry]:
        """
        Add multiple memory entries built from the provided context with intelligent deduplication.

        This method implements sophisticated memory management:
        1. Extracts structured memory entries from natural language context
        2. Checks for similar existing memories using semantic similarity
        3. Uses LLM to decide whether to update existing memories or create new ones
        4. Maintains history of updates for traceability
        5. Logs all operations for monitoring and debugging

        The method prevents duplicate memories by:
        - Comparing new entries against existing ones using vector similarity
        - Using configurable thresholds to determine update candidates
        - Leveraging LLM reasoning to make intelligent merge decisions
        - Maintaining update history for transparency

        Args:
            context: Context to build memory from. Flexible input types:
                - str: Natural language text to extract memories from
                - List[str]: Multiple text entries to process
                - List[Dict[str, str]]: Structured context with key-value pairs
            metadata: Additional metadata to store with memories (user_id, tags, etc.)
        Returns:
            MemoryEntries: Object containing all extracted memory entries

        Note:
            Uses timestamp-based filtering to avoid updating memories from the same batch
        """

        """
        Progress callback signature: (current_step, total_steps, message)
        
        Weighted progress:
        - Step 0-40: Extracting memories (40 percent given as LLM extraction is heavy)
        - Step 40-95: Processing individual memories (varies by count)
        - Step 95-100: Finalization
        """

        # Reset build stats for this call
        self._build_stats = {"new": 0, "update": 0, "duplicate": 0, "extracted": 0}

        # Notify: Starting extraction (0% -> 40%)
        if progress_callback:
            progress_callback(0, 100, "Extracting memories from text")

        # Convert diverse input formats to consistent format
        content = normalize_content(content, multimodal_support=self.multimodal_support)

        # Log the memory building action
        log_memory_building(content, self.agent_memory.get_user_id())

        # Generate metadata with createtion time and image URLs if any
        metadata = generate_metadata(content, metadata)

        # Step 1: Generate and store episodic memory (if enabled)
        enable_episodic = self.cfg.memory.get("enable_episodic_memory", False)
        episodic_entry = None
        
        if enable_episodic:
            segment_as_ep_mem = self.cfg.memory.get("use_segments_as_episodic", False)
            episodic_entry = self.generate_episodic_memory(content, metadata, segment_as_ep_mem)
            
            if episodic_entry:
                # Store the episodic memory
                self.agent_memory.add(episodic_entry)
                
                # Store the episodic ID in metadata for factual memories to reference
                metadata["episodic_memory_id"] = episodic_entry.index

        # Step 2: Extract structured factual memory entries from the context using LLM
        memory_entries = self.generate_memory_entries(
            content, metadata, enable_cue_index=self.cfg.memory.enable_cue_index
        )

        self._build_stats["extracted"] = len(memory_entries)

        # Phase 2: Processing (40% -> 95%)
        # After extraction at 40%
        if progress_callback:
            print("hitting here ###########################################################")
            progress_callback(40, 100, f"Processing {len(memory_entries)} memories...")


        total_entries = len(memory_entries)
        # Process each extracted memory entry individually using MemoryBuilder
        for idx, entry in enumerate(memory_entries):
            # process each memory entry with intelligent add/update logic

            # Update sub-progress for each memory
            if progress_callback:
                # sub_progress = f"Processing memory {idx + 1}/{total_entries}"
                # progress_callback(2, 3, sub_progress)
                # Calculate current progress within the 40-95 range
                if total_entries > 0:
                    progress_in_phase = int(40 + (55 * (idx + 1) / total_entries))
                    progress_callback(
                        progress_in_phase, 
                        100, 
                        f"Processing memory {idx + 1}/{total_entries}"
                    )

            self.upsert_memory_entry(entry=entry)

        # Phase 3: Completion (95% -> 100%)
        if progress_callback:
            progress_callback(100, 100, "Memories saved successfully!")

        return memory_entries

    @abstractmethod
    def generate_memory_entries(
        self, content: str, metadata: Optional[Dict], enable_cue_index: bool = False
    ) -> List[MemoryEntry]:
        """
        Abstract method to generate memory entries from content.
        Must be implemented by subclasses.
        """
        ...

    @abstractmethod
    def generate_episodic_memory(
        self, content: str, metadata: Optional[Dict]
    ) -> Optional[MemoryEntry]:
        """
        Abstract method to generate episodic memory from content.
        Must be implemented by subclasses.
        Returns None if episodic memory generation is not supported or fails.
        """
        ...

    def handle_multimodal_content(
        self,
        content: Optional[Union[str, Dict]],
        metadata: Optional[Dict],
        build_memory_prompt: str,
        response_format: Any,
    ) -> Optional[MemoryOutputs]:

        if (
            not isinstance(content, dict)
            or not self.multimodal_support
            or "image" not in content
        ):
            return None

        # Use multimodal data: send text prompt + images to LLM
        formatted_prompt = build_memory_prompt.format(content=content["text"], timestamp=metadata.get("timestamp", "") if metadata else "")

        # Create a multimodal user message with formatted prompt + images
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": formatted_prompt}]
                + content["image"],
            }
        ]

        # Try multimodal LLM call, fall back to text-only if image access fails
        try:
            memories = self._model_client.invoke(
                input=messages,
                response_format=response_format,
            )
        except BadRequestError as e:
            # Check if it's a 403 image access error
            if "403" in str(e) or "can not be accessed" in str(e):
                logger.warning(
                    f"Image URL access failed (403 error): {e}. Falling back to text-only mode."
                )
            return None
        return memories

    def _query_update_candidates(self, entry: MemoryEntry) -> List[MemoryEntry]:
        """
        Query existing memories to find candidates for update consideration.
        Returns:
            List of MemoryEntry objects that are candidates for update
        """
        # Create filter for user_id, excluding same batch and linked memories
        where = {
            "$and": [
                # Exclude memories from the same batch (same timestamp)
                {"creation_time": {"$ne": entry.creation_time}},
                # Exclude cue indices with linked memory
                {"linked_memory": {"$eq": ""}},
                # Only consider factual memories for updates (episodic memories are immutable for now)
                {"memory_type": {"$eq": "factual"}},
            ]
        }

        # memo-bench port: candidates come from the consolidation collection (sentence-similarity vectors,
        # 1b); under the 1c ablation (memory.embed_field=value) a factual entry is represented by its value.
        query_text = entry.value if self.cfg.memory.get("embed_field", "index") == "value" else entry.index

        # Search for similar existing memories using semantic similarity
        query_results: List[MemoryEntry] = self.agent_memory.query(
            query_text,
            top_k=5,
            where=where,
            query_mode=QueryMode.PRIMARY_ONLY,  # use the where condition only on the primary memory
            enhance_query=False,  # Disable enhanced queries
            return_history=False,  # Disable return history for update consideration
            consolidation=True,
        )

        # Filter candidates based on similarity threshold for update consideration
        update_candidates = []
        for cand in query_results:
            score = cand.score
            if score >= self.UPDATE_SCORE_THRESHOLD:
                update_candidates.append(cand)
        return update_candidates

    def upsert_memory_entry(
        self,
        entry: MemoryEntry,
    ) -> None:
        """
        Process a single memory entry with intelligent deduplication.

        This method handles the complex logic of:
        1. Checking for similar existing memories
        2. Making update decisions using LLM
        3. Managing memory history and metadata
        4. Logging operations

        Args:
            entry: Memory entry to process
            memory_store: The memory storage backend
            metadata: Metadata for the memory entry
            update_score_threshold: Threshold for considering updates
            query_method: Method to query existing memories
        """

        # if the same memory index exists, directly update it
        existing_entry = self.agent_memory.get(entry.index)
        if existing_entry:
            # Log diagnostic information to investigate why duplicate extraction occurred
            logger.warning(
                f"Duplicate memory index detected during upsert: '{entry.index}'\n"
                f"Existing: is_primary={existing_entry.is_primary_index()}, "
                f"is_cue={existing_entry.is_cue_index()}, "
                f"creation_time={existing_entry.creation_time}, "
                f"timestamp={existing_entry.timestamp}\n"
                f"New: is_primary={entry.is_primary_index()}, "
                f"is_cue={entry.is_cue_index()}, "
                f"creation_time={entry.creation_time}, "
                f"timestamp={entry.timestamp}\n"
                f"Values match: {existing_entry.value == entry.value}\n"
                f"This may indicate duplicate extraction or a race condition."
            )
            
            print(f"[Existing] {existing_entry.index} -> {existing_entry.value}")
            print(f"[*****New] {entry.index} -> {entry.value}")

            self._build_stats["duplicate"] += 1
            return

        # Query for similar existing memories to consider for update
        update_candidates = self._query_update_candidates(entry)

        if not update_candidates:
            # No similar memories found - add as completely new entry
            self.agent_memory.add(entry)
            self._build_stats["new"] += 1
        else:
            # Similar memories exist - use LLM to decide on update strategy
            update_decision = self._decide_memory_update(entry, update_candidates)
            # update_decision["should_update"] = (
            #     False  # TODO: remove this line to enable update
            # )
            if update_decision["should_update"]:
                self.update_memory(
                    entry,
                    update_decision,
                )
                self._build_stats["update"] += 1
            else:
                # LLM decided not to update - add as completely new memory entry
                self.agent_memory.add(entry)
                self._build_stats["new"] += 1

    def merge_memory(
        self,
        existing_entry: MemoryEntry,
        new_entry: MemoryEntry,
    ) -> MemoryEntry:
        """
        Merge two memory entries into one.

        Args:
            existing_entry: Existing memory entry
            new_entry: New memory entry to merge

        Returns:
            Merged MemoryEntry
        """
        # Safety check: only allow merging memories of the same type
        if existing_entry.memory_type != new_entry.memory_type:
            logger.warning(
                f"Attempted to merge different memory types (existing: {existing_entry.memory_type}, "
                f"new: {new_entry.memory_type}). Memories must be of the same type."
            )
            raise ValueError("Cannot merge memories of different types")
        
        merged_value = f"{existing_entry.value} {new_entry.value}"
        merged_cue_indices = list(
            set(
                existing_entry.cue_indices.split("||")
                + new_entry.cue_indices.split("||")
            )
        )
        
        # Merge episodic memory IDs
        merged_episodic_ids = []
        if existing_entry.episodic_memory_ids:
            merged_episodic_ids.extend(existing_entry.episodic_memory_ids)
        if new_entry.episodic_memory_ids:
            merged_episodic_ids.extend(new_entry.episodic_memory_ids)
        merged_episodic_ids = list(set(merged_episodic_ids))
        
        # Merge image URLs
        merged_image_urls = []
        if existing_entry.image_urls:
            merged_image_urls.extend(existing_entry.image_urls)
        if new_entry.image_urls:
            merged_image_urls.extend(new_entry.image_urls)
        merged_image_urls = list(set(merged_image_urls))
        
        merged_entry = MemoryEntry(
            memory_type=existing_entry.memory_type,
            index=existing_entry.index,
            value=merged_value,
            user_id=existing_entry.user_id,
            creation_time=existing_entry.creation_time,
            timestamp=new_entry.timestamp,
            cue_indices="||".join(merged_cue_indices),
            episodic_memory_ids=merged_episodic_ids,
            image_urls=merged_image_urls,
        )
        return merged_entry

    def update_memory(
        self,
        entry: MemoryEntry,
        update_decision: Dict[str, Any],
    ) -> None:
        """
        Handle the memory update process including history management.

        Args:
            entry: New memory entry
            update_decision: LLM decision about the update
            memory_store: The memory storage backend
            metadata: Metadata for the memory entry
            timestamp: Current timestamp
            creation_time: Creation time for the batch
        """
        best_candidate: MemoryEntry = update_decision["best_candidate"]
        
        # Safety check: only allow updating factual memories
        if best_candidate.memory_type != "factual":
            logger.warning(
                f"Attempted to update non-factual memory (type: {best_candidate.memory_type}). "
                f"Only factual memories can be updated. Adding as new entry instead."
            )
            self.agent_memory.add(entry)
            return
        
        updated_value = update_decision["updated_value"]
        updated_index = update_decision["updated_index"]

        # check whether the updated index exists
        existing_entry = self.agent_memory.get(updated_index)
        if existing_entry and updated_index != best_candidate.index:

            updated_index = f"{updated_index}. (Added on {get_current_timestamp()}]"

        # Remove the original memory entry to be updated
        self.agent_memory.delete(best_candidate.index)

        # add cue indices
        updated_cue_indices = []
        if self.cfg.memory.enable_cue_index:
            updated_cue_indices = self.cue_index_generator.generate_cue_indices(
                memory_value=updated_value,
                primary_index=updated_index,
            )

        # Build update history for traceability
        history = generate_history(entry, best_candidate)

        # Combine image URLs from both old and new memories and deduplicate
        updated_image_urls = []
        if best_candidate.image_urls:
            updated_image_urls.extend(best_candidate.image_urls)
        if entry.image_urls:
            updated_image_urls.extend(entry.image_urls)
        updated_image_urls = list(set(updated_image_urls))

        # Combine episodic memory IDs from both old and new memories and deduplicate
        updated_episodic_memory_ids = []
        if best_candidate.episodic_memory_ids:
            updated_episodic_memory_ids.extend(best_candidate.episodic_memory_ids)
        if entry.episodic_memory_ids:
            updated_episodic_memory_ids.extend(entry.episodic_memory_ids)
        updated_episodic_memory_ids = list(set(updated_episodic_memory_ids))

        new_memory_entry = MemoryEntry(
            memory_type=best_candidate.memory_type,
            index=updated_index,
            value=updated_value,
            creation_time=entry.creation_time,
            timestamp=entry.timestamp,
            cue_indices="||".join(updated_cue_indices),
            history=history,
            image_urls=updated_image_urls,
            episodic_memory_ids=updated_episodic_memory_ids,
        )
        self.agent_memory.add(new_memory_entry)

        # Log the update operation
        logger.info(
            "\n" + "-" * 60 + "\n"
            f"MEMORY STORE: Update|{entry.creation_time}|{self.agent_memory.get_user_id()}\n"
            f"Index: {best_candidate.index} -> {updated_index}\n"
            f"Value: {updated_value}\n" + "-" * 60
        )

    def _decide_memory_update(self, new_entry, update_candidates) -> Dict[str, Any]:
        """
        Use LLM to make intelligent decisions about memory updates vs new additions.

        Args:
            new_entry: New memory entry being added
            update_candidates: List of similar existing memory entries

        Returns:
            Dict containing the LLM's decision
        """
        # Prepare candidate information in structured format for LLM analysis
        candidates_info = []
        for i, candidate in enumerate(update_candidates):
            candidates_info.append(
                {
                    "index": i,
                    "score": candidate.score,
                    "index_text": candidate.index,
                    "value": candidate.value,
                    "creation_time": candidate.creation_time,
                }
            )

        # Prepare structured arguments for LLM prompt
        prompt_args = {
            "new_index": new_entry.index,
            "new_value": new_entry.value,
            "candidates_info": self._format_candidates_for_prompt(candidates_info),
        }

        try:
            # Use structured response format for reliable LLM output parsing
            decision: MemoryUpdateDecision = self._model_client.invoke(
                input=PROMPT_MEMORY_UPDATE_DECISION,
                prompt_args=prompt_args,
                response_format=MemoryUpdateDecision,
            )

            # Convert Pydantic model to dictionary and add candidate reference
            decision_dict = decision.model_dump()
            if decision.should_update and decision.best_candidate_index is not None:
                decision_dict["best_candidate"] = update_candidates[
                    decision.best_candidate_index
                ]
            else:
                decision_dict["best_candidate"] = None

            return decision_dict

        except Exception as e:
            # Robust fallback: if LLM response parsing fails, default to no update
            return {
                "should_update": False,
                "best_candidate": None,
                "updated_value": None,
                "updated_index": new_entry.index,
                "reasoning": f"LLM parsing error: {e}",
            }

    def _format_candidates_for_prompt(self, candidates_info: List[Dict]) -> str:
        """
        Format candidate memory information for inclusion in LLM prompts.

        Args:
            candidates_info: List of dictionaries containing candidate information

        Returns:
            str: Formatted string representation of all candidates
        """
        formatted = []
        for candidate in candidates_info:
            formatted.append(
                PROMPT_CANDIDATE_FORMAT.format(
                    index=candidate["index"],
                    score=candidate["score"],
                    index_text=candidate["index_text"],
                    value=candidate["value"],
                    creation_time=candidate["creation_time"],
                )
            )
        return "\n".join(formatted)


# memo-bench port: Anthropic prompt caching for the static head of each build prompt (text unchanged).
from memora.utils.llm import register_cache_prefix  # noqa: E402

register_cache_prefix("PROMPT_BUILD_MEMORY", PROMPT_BUILD_MEMORY)
