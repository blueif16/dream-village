# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations
from importlib import metadata
import logging
from typing import List, Optional, Union, Dict
from omegaconf import DictConfig
from pydantic import BaseModel, Field

from memora.core.memory_entry import MemoryEntry
from memora.utils.llm import ChatCompletionModel
from memora.utils.memory import convert_memory_output
from memora.core.memory import AgentMemory

from memora.builder.memory_builder import (
    PROMPT_BUILD_MEMORY,
    MemoryBuilder,
    MemoryOutputs,
)

# Initialize module logger
logger = logging.getLogger(__name__)

PROMPT_EPISODIC_MEMORY = """
You are an expert episodic memory generator that creates episodic memory summaries from conversation segments.

# TASK:
Your task is to generate an episodic memory with an index and a detailed summary based on the provided conversation segment.

Generate the episodic memory in the following format:
EpisodicIndex: [6-8 word summary that captures the main topic, entity or event of the episode]
EpisodicValue: [1-3 sentence of descriptive summary of the conversation]

# GUIDELINES:
1. The EpisodicIndex
- Create a short index (6-8 words) that captures the main topic or event of the episode.
- Always include the specific context (e.g., domain, or entity) from the source text in the Index to avoid vague terms.

2. The EpisodicValue
- Generate an episodic summary (1-3 sentences) that captures:
  - The main information of the conversation segment, including the main topic, theme, or event being discussed.
  - Relevant participants in the conversation, refer to the participants by their names if available.
  - Use the original wordings from the conversation when possible.
- Focus on "what happened" rather than specific details. The summary is meant to provide context for future retrieval.
- Make the summary self-contained and understandable without the original conversation.
- If images are present, consider the visual content as part of the textual context.
- Use only information present in the conversation segment to generate the summary; do not add external knowledge or infer beyond the content.
- If the conversation is between a user and an AI assistant, focus on the user's inputs and the overall context rather than the assistant's responses.

Input Conversation Segment:
{content}

Output:
"""


class EpisodicMemoryOutput(BaseModel):
    episodic_index: str = Field(
        description="A short 6-8 word summary that captures the main topic, entity or event of the episode"
    )
    episodic_value: str = Field(
        description="A detailed 1-4 sentence episodic summary providing context and narrative"
    )

class ChatMemoryBuilder(MemoryBuilder):

    def __init__(self, cfg: DictConfig, agent_memory: AgentMemory, model_client: ChatCompletionModel):
        super().__init__(cfg, agent_memory, model_client)

    def generate_memory_entries(
        self,
        content: Optional[Union[str, Dict]],
        metadata: Optional[Dict],
        enable_cue_index: bool = False,
    ) -> List[MemoryEntry]:
        """
        Build memory from content.

        Args:
            content: The content to extract memories from
            enable_cue_index: If True, generate cue indices in a separate LLM call. If False, use original format.

        Returns:
            memory entries extracted from the content
        """

        # Step 1: Always use PROMPT_BUILD_MEMORY to extract factual memories
        build_memory_prompt = PROMPT_BUILD_MEMORY
        response_format = MemoryOutputs

        # handle multimodal content (dict with text and images keys)
        memories = self.handle_multimodal_content(
            content, metadata, build_memory_prompt, response_format
        )

        timestamp = metadata.get("timestamp", "N/A") if metadata else "N/A"
        # extract date from timestamp if possible
        timestamp = timestamp.split(" on ")[-1] if " on " in timestamp else timestamp

        if not memories:
            # handle pure text content
            memories = self._model_client.invoke(
                input=build_memory_prompt,
                prompt_args={
                    "content": content,
                    "timestamp": timestamp,
                },
                response_format=response_format,
            )

        # Step 2: Convert to memory entries (without cue indices yet)
        memory_entries = convert_memory_output(memories, metadata, enable_cue_index=False)
        
        # Ensure all entries are marked as factual
        for entry in memory_entries:
            entry.memory_type = "factual"
        
        # Step 3: If cue index is enabled, generate cue indices for all memories in a single batch call
        if enable_cue_index and memory_entries:
            try:
                # Prepare batch input
                memories_batch = [
                    {"index": entry.index, "value": entry.value}
                    for entry in memory_entries
                ]
                
                # Generate cue indices for all memories in one LLM call
                cue_indices_map = self.cue_index_generator.generate_cue_indices_batch(memories_batch)
                
                # Attach cue indices to each entry
                for entry in memory_entries:
                    cue_indices = cue_indices_map.get(entry.index, [])
                    # Only set cue_indices if the list is non-empty
                    entry.cue_indices = "||".join(cue_indices) if cue_indices and len(cue_indices) > 0 else ""
                    
            except Exception as e:
                logger.warning(f"Failed to generate cue indices in batch: {e}")
                # Fall back to empty cue indices
                for entry in memory_entries:
                    entry.cue_indices = ""
        
        return memory_entries
    
    def generate_episodic_memory(
        self,
        content: Optional[Union[str, Dict]],
        metadata: Optional[Dict],
        segment_as_ep_mem: bool = True,
    ) -> Optional[MemoryEntry]:
        """
        Generate episodic memory using the original conversation segment text.
        
        Args:
            content: The conversation content (text or multimodal dict from normalize_content)
            metadata: Metadata for the episodic memory
            
        Returns:
            MemoryEntry for episodic memory using the original conversation segment
        """
        try:
            # Handle multimodal content - extract text content
            # After normalize_content, content is either a string or dict with "text" key
            if isinstance(content, dict) and "text" in content:
                content_text = content["text"]
            else:
                content_text = content
            
            # Use the segment topic as the index if available
            segment_topic = metadata.get("segment_topic", "conversation") if metadata else "conversation"
            segment_index = metadata.get("segment_index", 0) if metadata else 0
            timestamp = metadata.get("timestamp", "") if metadata else ""

            # Always store the original text for hybrid episodic format
            original_text = content_text
            
            if not segment_as_ep_mem:
                episodic_output = self._model_client.invoke(
                    input=PROMPT_EPISODIC_MEMORY,
                    prompt_args={"content": content_text},
                    response_format=EpisodicMemoryOutput,
                )

                episodic_index = f"[EPISODIC] {episodic_output.episodic_index}"
                episodic_value = episodic_output.episodic_value

                episodic_entry = MemoryEntry(
                    memory_type="episodic",
                    index=episodic_index,
                    value=episodic_value,
                    original_text=original_text,
                    creation_time=metadata.get("creation_time", ""),
                    timestamp= metadata.get("timestamp", ""),
                )
            
            else:
            # Create episodic index using segment info
                episodic_index = f"[EPISODIC] {segment_topic} (segment {segment_index})"
            
            # Use the original conversation segment text as the episodic value
                episodic_value = content_text
            
            # Create episodic memory entry using the original text
            episodic_entry = MemoryEntry(
                memory_type="episodic",
                index=episodic_index,
                value=episodic_value,
                original_text=original_text,
                creation_time=metadata.get("creation_time", "") if metadata else "",
                timestamp=timestamp,
            )
            
            return episodic_entry
                
        except Exception as e:
            logger.warning(f"Failed to generate episodic memory from segment: {e}")
            return None


# memo-bench port: Anthropic prompt caching for the static head of each build prompt (text unchanged).
from memora.utils.llm import register_cache_prefix  # noqa: E402

register_cache_prefix("PROMPT_BUILD_MEMORY", PROMPT_BUILD_MEMORY)
register_cache_prefix("PROMPT_EPISODIC_MEMORY", PROMPT_EPISODIC_MEMORY)
