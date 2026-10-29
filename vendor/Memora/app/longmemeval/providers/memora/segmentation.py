# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Conversation Segmentation Module

This module provides LLM-based conversation segmentation to identify topic boundaries
in conversation batches.
"""

import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from memora.utils.llm import ChatCompletionModel

logger = logging.getLogger(__name__)


# Pydantic models for structured output
class Episode(BaseModel):
    """Represents a single topical episode in a conversation."""
    indices: List[int] = Field(description="List of message indices (0-based) in this episode")
    topic: str = Field(description="Brief description of the topic discussed in this episode")


class SegmentationOutput(BaseModel):
    """Output format for conversation segmentation."""
    episodes: List[Episode] = Field(description="List of topical episodes identified in the conversation")


# Segmentation prompt template adapted from Nemori
SEGMENTATION_PROMPT_TEMPLATE = """You are an expert conversation segmentation specialist. Your goal is to analyze a series of messages in a conversation and segment them into coherent topical episodes.

# TASK:
Read the conversation carefully, and identify points where the topic shifts significantly. Group messages that discuss similar subject, event, or theme into a single episode.

An episode is defined as a sequence of messages that revolve around a core topic or theme. Your task is to segment the conversation into such episodes.

## Output Format
Your output should be a JSON object with the following structure:
{{
    "episodes": [
        {{ 
            "topic": "<brief topic description>", 
            "indices": [<list of message indices in this episode>] 
        }},
        ...
    ]
}}

Where each episode contains:
- topic: A brief description (a few words) summarizing the main topic of the episode
- indices: A list of 1-based indices of messages that belong to this episode


# GUIDELINES:

## Segmentation Criteria
- Topical shift: Identify topic shifts in the messages. Does it introduce a new subject, event, or theme? Break the episode there. Be sensitive to subtle shifts in topic.
- Transitions: Look for transition phrases that signal a new episode, such as "By the way", "Changing the subject", or "On another note".
- Time gaps: Significant time lapses between messages may indicate a new episode.
- Setting changes: Changes in speaker, location, or context can signal a new episode.
- Topical grouping: Group consecutive messages into the same episode if they discuss the same topic or theme.


## Episode Length
- An episode should typically contain 2-8 messages.
- Combine messages into larger episodes when they discuss the same topic. 
- Avoid having long episodes (more than 8 messages) that cover multiple sub-topics.
- Avoid treating a single message as a standalone episode unless it clearly marks a shift in topic.
- When in doubt, split into smaller episodes.


## Formatting Rules
- Use 1-based indexing for message indices (i.e., the first message is index 1).
- Ensure that all messages are included in exactly one episode (no gaps or overlaps).
- Indices within each episode should be consecutive, reflecting the order of messages in the conversation.
- Episodes should cover all messages exactly once (no gaps, no overlaps).

Example output:
{{
    "episodes": [
        {{
            "topic": "General introduction and greetings",
            "indices": [1, 2, 3, 4],
        }},
        {{
            "topic": "Discussion about vacation plans",
            "indices": [5, 6],
        }},
        {{
            "topic": "Recap of last year's events",
            "indices": [7, 8, 9, 10, 11, 12],
        }}, 
        ...
    ]
}}

Respond only with the JSON object and no additional text.

Segment the following conversation:

{messages}

Output:
"""


class ConversationSegmenter:
    """Segments conversations into topical episodes using LLM-based analysis."""

    def __init__(self, cfg):
        """Initialize the conversation segmenter.
        
        Args:
            cfg: Configuration object containing OpenAI settings
        """
        self.cfg = cfg
        self.enable_segmentation = cfg.memory.get("enable_segmentation", False)
        self.batch_size = 2  # Default batch size for non-LLM mode
        
        # Only initialize LLM if segmentation is enabled
        if self.enable_segmentation:
            logger.info("LLM-based segmentation enabled")
            self.llm = ChatCompletionModel(cfg)
        else:
            logger.info("Batch-based segmentation enabled")
            self.llm = None

    def _format_messages_for_prompt(
        self, 
        messages: List[Dict[str, Any]], 
        start_idx: int = 0,
        is_context: bool = False
    ) -> str:
        """
        Format messages into a readable prompt format.
        
        Args:
            messages: List of message dictionaries
            start_idx: Starting index for message numbering
            is_context: If True, mark as context-only messages
            
        Returns:
            Formatted string of messages
        """
        formatted = []
        prefix = "[CONTEXT] " if is_context else ""
        
        for i, msg in enumerate(messages):
            idx = start_idx + i + 1  # Use 1-based indexing in prompt
            content = msg.get("content", "")
            
            # Handle multimodal content (text + images)
            if isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                text_content = " ".join(text_parts)
                has_image = any(item.get("type") == "image_url" for item in content)
                if has_image:
                    text_content += " [Contains image]"
            else:
                text_content = content
            
            role = msg.get("role", "user")
            formatted.append(f"{prefix}{idx}. {role}: {text_content}")
        
        return "\n".join(formatted)

    def _create_segmentation_prompt(
        self,
        messages: List[Dict[str, Any]]
    ) -> str:
        """
        Create the prompt for LLM segmentation.
        
        Args:
            messages: Messages to segment
            
        Returns:
            Prompt string
        """
        formatted_messages = self._format_messages_for_prompt(messages, start_idx=0, is_context=False)
        return SEGMENTATION_PROMPT_TEMPLATE.format(messages=formatted_messages)

    def _call_llm_for_segmentation(self, prompt: str) -> SegmentationOutput:
        """Call LLM to perform segmentation using structured output.
        
        Args:
            prompt: The segmentation prompt
            
        Returns:
            SegmentationOutput with parsed episodes
        """
        try:
            result = self.llm.invoke(
                input=prompt,
                response_format=SegmentationOutput,
                source="ConversationSegmenter",
                temperature=0.0
            )
            return result
            
        except Exception as e:
            logger.error(f"LLM segmentation failed: {e}")
            raise

    def segment_conversation(
        self, 
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Segment a conversation into topical episodes.
        
        Uses LLM-based segmentation if enabled, otherwise falls back to batch-based chunking.
        Always returns valid segments - falls back to batch mode on any failure.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            List of segments, each containing:
                - indices: List of message indices in this segment
                - topic: Description of the topic (or generic for batch mode)
        """
        num_messages = len(messages)
        
        if num_messages == 0:
            return []
        
        # Use LLM-based segmentation if enabled
        if self.enable_segmentation:
            segments = self._segment_with_llm(messages)
            if segments is not None:
                return segments
            else:
                logger.warning("LLM segmentation failed, falling back to batch mode")
                return self._segment_with_batches(messages)
        else:
            return self._segment_with_batches(messages)
    
    def _segment_with_llm(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]] | None:
        """Segment using LLM analysis.
        
        Returns:
            List of valid segments if successful, None if failed (validation or LLM error)
        """
        num_messages = len(messages)
        logger.info(f"Segmenting conversation with LLM ({num_messages} messages)")
        
        try:
            prompt = self._create_segmentation_prompt(messages)
            result = self._call_llm_for_segmentation(prompt)
            
            # Convert Pydantic models to dicts and convert 1-based to 0-based indices
            segments = [{"indices": [idx - 1 for idx in ep.indices], "topic": ep.topic} for ep in result.episodes]
            
            # Validate segment consistency
            if not self._validate_segments(segments, num_messages):
                logger.error(f"LLM returned invalid segments for {num_messages} messages")
                return None
            
            logger.info(f"Created {len(segments)} LLM segments from {num_messages} messages")
            return segments
            
        except Exception as e:
            logger.error(f"LLM segmentation error: {e}")
            return None
    
    def _segment_with_batches(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Segment using simple batch-based chunking."""
        num_messages = len(messages)
        logger.info(f"Segmenting conversation with batch mode ({num_messages} messages, batch_size={self.batch_size})")
        
        segments = []
        for i in range(0, num_messages, self.batch_size):
            end_idx = min(i + self.batch_size, num_messages)
            segments.append({
                "indices": list(range(i, end_idx)),
                "topic": f"Conversation batch {i//self.batch_size + 1}"
            })
        
        logger.info(f"Created {len(segments)} batch segments")
        return segments

    def _validate_segments(self, segments: List[Dict[str, Any]], num_messages: int) -> bool:
        """Validate that segments are consistent with the conversation.
        
        Args:
            segments: List of segment dictionaries with indices and topics
            num_messages: Total number of messages in the conversation
            
        Returns:
            True if segments are valid, False otherwise
        """
        if not segments:
            logger.warning("No segments returned")
            return False
        
        # Collect all indices from segments
        all_indices = []
        for seg in segments:
            indices = seg.get("indices", [])
            if not indices:
                logger.error(f"Segment has empty indices: {seg}")
                return False
            all_indices.extend(indices)
        
        # Check 1: All indices should be in valid range [0, num_messages-1]
        for idx in all_indices:
            if idx < 0 or idx >= num_messages:
                logger.error(f"Index {idx} out of range [0, {num_messages-1}]")
                return False
        
        # Check 2: No duplicate indices
        if len(all_indices) != len(set(all_indices)):
            duplicates = [idx for idx in set(all_indices) if all_indices.count(idx) > 1]
            logger.error(f"Duplicate indices found: {duplicates}")
            return False
        
        # Check 3: All messages should be covered (no gaps)
        expected_indices = set(range(num_messages))
        actual_indices = set(all_indices)
        
        missing = expected_indices - actual_indices
        if missing:
            logger.error(f"Missing message indices: {sorted(missing)}")
            return False
        
        extra = actual_indices - expected_indices
        if extra:
            logger.error(f"Extra indices not in conversation: {sorted(extra)}")
            return False
        
        # Check 4: Indices within each segment should be consecutive
        for seg in segments:
            indices = sorted(seg["indices"])
            for i in range(len(indices) - 1):
                if indices[i+1] - indices[i] != 1:
                    logger.warning(f"Non-consecutive indices in segment '{seg['topic']}': {indices}")
                    # This is a warning, not an error - we'll allow it but log it
        
        logger.info("Segment validation passed")
        return True
