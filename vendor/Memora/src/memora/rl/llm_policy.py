# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Optional, Tuple
from dataclasses import dataclass
from omegaconf import DictConfig

from memora.utils.llm import get_aoai_chat_completion_client
from memora.core.memory_entry import MemoryEntry
from .trajectory_utils import RetrievalState, RetrievalAction, ActionType
import logging

logger = logging.getLogger(__name__)

ACTION_SELECTION_PROMPT = """You are a memory retrieval agent. Your task is to select the best action to retrieve relevant memories for answering a question.

## Current State
- Question: {query}
- Memories retrieved so far: {num_retrieved}
- Remaining budget: {budget}

## Already Retrieved Memories
{retrieved_memories}

## Available Actions
{available_actions}

## Instructions
1. Analyze which memory would be most helpful for answering the question
2. Consider diversity - avoid selecting memories too similar to what's already retrieved
3. If you have enough information to answer the question, select STOP
4. Select the action that maximizes information gain

## Output Format
Return ONLY the action number (e.g., "1") or "STOP". No explanation.

Your selection:"""

class LLMPolicy:
    """
    LLM-based policy for action selection (baseline, not trainable).  
    Uses GPT-4.1-mini with prompting to select retrieval actions.
    """
    
    def __init__(
            self,
            cfg: DictConfig,
            max_primary_actions: int = 5,
            max_cue_actions: int = 3,
        ):
        self.cfg = cfg
        self.client = get_aoai_chat_completion_client(cfg)
        self.max_primary = max_primary_actions
        self.max_cue = max_cue_actions

        # Tracking for debugging
        self.last_prompt = None
        self.last_respones = None
    
    def select_action(
        self,
        state: RetrievalState,
        primary_candidates: List[MemoryEntry],
        cue_candidates: List[MemoryEntry],
        retrieved_memories: Optional[List[dict]] = None,
    ) -> RetrievalAction:
        """
        Use LLM to select the next action.
        
        Args:
            state: Current retrieval state
            primary_candidates: Available primary memory candidates
            cue_candidates: Available cue index candidates
            retrieved_memories: Already retrieved memory contents (for context)
            
        Returns:
            Selected RetrievalAction
        """
        # Filter out already retrieved
        available_primary = [
            m for m in primary_candidates if m.index not in state.retrieved_memories
        ]
        available_cue = [
            m for m in cue_candidates if m.index not in state.retrieved_memories
        ]
        
        # Check stopping conditions
        if state.budget <= 0 or (not available_primary and not available_cue):
            return RetrievalAction(action_type=ActionType.STOP)
        
        actions_text = self._format_actions(available_primary, available_cue)
        retrieved_text = self._format_retrieved(retrieved_memories)
        
        # Build prompt
        prompt = ACTION_SELECTION_PROMPT.format(
            query=state.query,
            num_retrieved=len(state.retrieved_memories),
            budget=state.budget,
            retrieved_memories=retrieved_text,
            available_actions=actions_text,
        )

        self.last_prompt = prompt # for debugging
        
        # Call LLM to generate response
        try:
            response = self.client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            selection = response.choices[0].message.content.strip()
            self.last_respones = selection  # for debugging

            # Parse response
            return self._parse_selection(
                selection, available_primary, available_cue
            )
            
        except Exception as e:
            print(f"LLM policy error: {e}, falling back to greedy")
            return self._fallback_greedy(available_primary, available_cue)
        
    def _format_retrieved(self, retrieved_memories: Optional[List[dict]]) -> str:
        """Format already retrieved memories for context"""
        if not retrieved_memories:
            return "(None yet)"
        
        lines = []
        for i, m in enumerate(retrieved_memories[:5], 1):  # Limit to 5 for prompt size
            value = m.get("value", "")[:80]  # Truncate
            lines.append(f"  {i}. {value}...")
        
        if len(retrieved_memories) > 5:
            lines.append(f"  ... and {len(retrieved_memories) - 5} more")
        
        return "\n".join(lines)
    
    
    def _format_actions(
        self,
        primary_candidates: List[MemoryEntry],
        cue_candidates: List[MemoryEntry],
    ) -> str:
        """
        Format available actions for the prompt

        Creates a numbered list of actions for the LLM to choose from.
        
        Example output:
        "1. [PRIMARY] Jolene electricity engineering project
            Content: Jolene mentioned she is working on an electricity...
        2. [PRIMARY] Deborah career counseling
            Content: Deborah gave advice about career paths...
        3. [CUE] engineering work
            Links to memories about this topic
        STOP. Stop retrieval (enough information gathered)"
        """

        lines = []
        action_idx = 1
        
        # Primary candidates
        for m in primary_candidates[:self.max_primary]:
            # Safely get value content
            content = (m.value or "")[:100]
            lines.append(
                f"{action_idx}. [PRIMARY] {m.index}\n"
                f"   Content: {content}{'...' if len(m.value or '') > 100 else ''}"
            )
            action_idx += 1
        
        # Cue candidates
        for m in cue_candidates[:self.max_cue]:
            lines.append(
                f"{action_idx}. [CUE] {m.index}\n"
                f"   Links to memories about this topic"
            )
            action_idx += 1
        
        lines.append("STOP. Stop retrieval (enough information gathered)")
        
        return "\n".join(lines)
    
    ## see from here
    def _parse_selection(
        self,
        selection: str,
        primary_candidates: List[MemoryEntry],
        cue_candidates: List[MemoryEntry],
    ) -> RetrievalAction:
        """Parse LLM selection into RetrievalAction"""
        selection = selection.strip().upper()
        
        if "STOP" in selection:
            return RetrievalAction(action_type=ActionType.STOP)
        
        try:
            # Extract number from response
            # handle cases like "1", "1.", "Action 1" etc
            import re
            match = re.search(r'(\d+)', selection)

            if not match:
                raise ValueError("No action number found in {selection}")
        
            # Convert to 0-indexed
            idx = int(match.group()) - 1  
            
            # Map index to candidate
            num_primary = min(len(primary_candidates), self.max_primary)
            num_cue = min(len(cue_candidates), self.max_cue)
            
            if 0 <= idx < num_primary:
                m = primary_candidates[idx]
                return RetrievalAction(
                    action_type=ActionType.QUERY_PRIMARY_INDEX,
                    target_memory_index=m.index,
                    score=m.score,
                )
            
            elif idx < num_primary + num_cue:
                m = cue_candidates[idx - num_primary]
                return RetrievalAction(
                    action_type=ActionType.QUERY_CUE_INDEX,
                    target_memory_index=m.index,
                    score=m.score,
                )
            else:
                print("=="*10)
                print(f"Falling back to greedy approach due to invalid index {idx + 1}")
                print("=="*10)
                logger.warning(f"Invalid action index {idx + 1}, falling back to greedy")
                return self._fallback_greedy(primary_candidates, cue_candidates)
                
        except (ValueError, IndexError):
            return self._fallback_greedy(primary_candidates, cue_candidates)
    
    def _fallback_greedy(
        self,
        primary_candidates: List[MemoryEntry],
        cue_candidates: List[MemoryEntry],
    ) -> RetrievalAction:
        """Fallback to greedy selection if LLM fails"""
        all_candidates = (
            [(m, ActionType.QUERY_PRIMARY_INDEX) for m in primary_candidates[:self.max_primary]] +
            [(m, ActionType.QUERY_CUE_INDEX) for m in cue_candidates[:self.max_cue]]
        )
        
        if not all_candidates:
            return RetrievalAction(action_type=ActionType.STOP)
        
        # Select highest score
        best = max(all_candidates, key=lambda x: x[0].score)
        return RetrievalAction(
            action_type=best[1],
            target_memory_index=best[0].index,
            score=best[0].score,
        )